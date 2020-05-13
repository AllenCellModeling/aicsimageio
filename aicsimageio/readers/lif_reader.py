#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element

import dask.array as da
import numpy as np
from dask import delayed
from readlif import utilities
from readlif.reader import LifFile

from .. import exceptions, types
from ..buffer_reader import BufferReader
from ..constants import Dimensions
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class LifReader(Reader):
    """
    LifReader wraps readlif.reader to provide the same reading capabilities but
    abstracts the specifics of using the backend library to create a unified interface.
    This enables higher level functions to duck type the File Readers.

    Parameters
    ----------
    data: types.FileLike
        A string or path to the LIF file to be read.
    chunk_by_dims: List[str]
        The dimensions to use as the for mapping the chunks / blocks.
        Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
        Note: SpatialY and SpatialX will always be added to the list if not present.
    S: int
        If the image has different dimensions on any scene from another, the dask array
        construction will fail.
        In that case, use this parameter to specify a specific scene to construct a
        dask array for.
        Default: 0 (select the first scene)
    """

    LIF_MAGIC_BYTE = 0x70
    LIF_MEMORY_BYTE = 0x2A

    ########################################################
    #
    #  Note LifFile (the underlying library of this Reader) only allows for
    #  reading a single YX plane at a time, unlike other underlying libraries
    #  such as tiffffile or aicspylibczi that allow for reading n-dim chunk reading.
    #
    ########################################################

    @property
    def dims(self) -> str:
        """
        The dimensions for a lif file.

        Returns
        -------
        str
            "STCZYX"
        """
        return Dimensions.DefaultOrder  # forcing 6 D

    @staticmethod
    def _compute_offsets(lif: LifFile) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute the offsets for each of the YX planes so that the LifFile object
        doesn't need to be created for each YX plane read.

        Parameters
        ----------
        lif : LifFile
            The LifFile object with an open file pointer to the file.

        Returns
        -------
        List[numpy.ndarray]
            The list of numpy arrays holds the offsets and it should be accessed as
            [S][T,C,Z].
        numpy.ndarray
            The second numpy array holds the plane read length per Scene.

        """
        scene_list = []
        scene_img_length_list = []

        for s_index, img in enumerate(lif.get_iter_image()):
            pixel_type = LifReader.get_pixel_type(lif.xml_root, s_index)
            (
                x_size,
                y_size,
                z_size,
                t_size,
            ) = img.dims  # in comments in this block these correspond to X, Y, Z, T
            c_size = img.channels  # C
            img_offset, img_block_length = img.offsets
            offsets = np.zeros(shape=(t_size, c_size, z_size), dtype=np.uint64)
            t_offset = c_size * z_size
            z_offset = c_size
            seek_distance = c_size * z_size * t_size
            if img_block_length == 0:
                # In the case of a blank image, we can calculate the length from
                # the metadata in the LIF. When this is read by the parser,
                # it is set to zero initially.
                log.debug(
                    "guessing image length: LifFile assumes 1byte per pixel,"
                    " but I think this is wrong!"
                )
                image_len = seek_distance * x_size * y_size * pixel_type.itemsize
            else:  # B = bytes per pixel
                image_len = int(
                    img_block_length / seek_distance
                )  # B*X*Y*C*Z*T / C*Z*T = B*X*Y = size of an YX plane

            for t_index in range(t_size):
                t_requested = t_offset * t_index  # C*Z*t_index
                for c_index in range(c_size):
                    c_requested = c_index
                    for z_index in range(z_size):
                        z_requested = z_offset * z_index  # z_index * C
                        item_requested = (
                            t_requested + z_requested + c_requested
                        )  # the number of YX frames to jump
                        # self.offsets[0] is the offset to the beginning of the image
                        # block here we index into that block to get the offset for any
                        # YX frame in this image block
                        offsets[t_index, c_index, z_index] = np.uint64(
                            img.offsets[0] + image_len * item_requested
                        )

            scene_list.append(offsets)
            scene_img_length_list.append(image_len)

        return scene_list, np.asarray(scene_img_length_list, dtype=np.uint64)

    def __init__(
        self,
        data: types.FileLike,
        chunk_by_dims: List[str] = [
            Dimensions.SpatialZ,
            Dimensions.SpatialY,
            Dimensions.SpatialX,
        ],
        S: int = 0,
        **kwargs,
    ):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

        # Store parameters needed for _daread
        self.chunk_by_dims = chunk_by_dims
        self.specific_s_index = S
        lif = LifFile(filename=self._file)
        #  _chunk_offsets is a list of ndarrays
        # (only way I could deal with inconsistent scene shape)
        self._chunk_offsets, self._chunk_lengths = LifReader._compute_offsets(lif=lif)

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        """
        Test the file that was provided to check that the header is consistent with
        this reader.

        Parameters
        ----------
        buffer: io.BytesIO
            This is the contents of the file path the LifReader was initialized with.

        Returns
        -------
        True / False
            True if it has the right header byte structure False if it does not.

        """
        with BufferReader(buffer) as buffer_reader:
            header = buffer_reader.read_bytes(n_bytes=8)

            # If the buffer is to short return false
            if len(buffer_reader.endianness) < 2 or len(header) < 8:
                return False
            # Check for the magic byte
            if (
                buffer_reader.endianness[0] != LifReader.LIF_MAGIC_BYTE
                and header[1] != LifReader.LIF_MAGIC_BYTE
            ):
                return False
            # Check for the memory byte, if magic byte and memory byte are present
            # return True
            if header[6] == LifReader.LIF_MEMORY_BYTE:
                return True
        return False

    @staticmethod
    def _dims_shape(lif: LifFile):
        """
        Get the dimensions for the opened file from the binary data (not the metadata)

        Parameters
        ----------
        lif: LifFile

        Returns
        -------
        list[dict]
            A list of dictionaries containing Dimension / depth. If the shape is
            consistent across Scenes then the list will have only one Dictionary. If
            the shape is inconsistent the the list will have a dictionary for each
            Scene. A consistently shaped file with 3 scenes, 7 time-points
            and 4 Z slices containing images of (h,w) = (325, 475) would return
            [
             {'S': (0, 3), 'T': (0,7), 'X': (0, 475), 'Y': (0, 325), 'Z': (0, 4)}
            ].
            The result for a similarly shaped file but with different number of time
            points per scene would yield
            [
             {'S': (0, 1), 'T': (0,8), 'X': (0, 475), 'Y': (0, 325), 'Z': (0, 4)},
             {'S': (1, 2), 'T': (0,6), 'X': (0, 475), 'Y': (0, 325), 'Z': (0, 4)},
             {'S': (2, 3), 'T': (0,7), 'X': (0, 475), 'Y': (0, 325), 'Z': (0, 4)}
            ]

        """
        shape_list = [
            {
                Dimensions.Time: (0, img.nt),
                Dimensions.Channel: (0, img.channels),
                Dimensions.SpatialZ: (0, img.nz),
                Dimensions.SpatialY: (0, img.dims[1]),
                Dimensions.SpatialX: (0, img.dims[0]),
            }
            for idx, img in enumerate(lif.get_iter_image())
        ]
        consistent = all(elem == shape_list[0] for elem in shape_list)
        if consistent:
            shape_list[0][Dimensions.Scene] = (0, len(shape_list))
            shape_list = [shape_list[0]]
        else:
            for idx, lst in enumerate(shape_list):
                lst[Dimensions.Scene] = (idx, idx + 1)
        return shape_list

    @staticmethod
    def _read_dims_to_ranges(lif: LifFile, read_dims: Optional[Dict[str, int]] = None):
        """
        Convert the provided read_dims and file structure into ranges to iterate over

        Parameters
        ----------
        lif: LifFile
            The LifFile to get the ranges from
        read_dims: Dict[str: int]
            The list of locked dimensions

        Returns
        -------
        Dict[Dimension: range]
            These ranges can then be used to iterate through the specified YX images

        """
        if read_dims is None:
            read_dims = {}

        data_shape = LifReader._dims_shape(lif=lif)

        # If S is in read_dims then use the specified value and the specified dims for
        # that scene
        if Dimensions.Scene in read_dims:
            s_range = range(
                read_dims[Dimensions.Scene], read_dims[Dimensions.Scene] + 1
            )
            s_dict = data_shape[s_range[0]]
        else:
            s_range = range(*data_shape[0][Dimensions.Scene])
            s_dict = data_shape[0]

        # Map the dims over to ranges and if the dim is in read_dims make the range
        # over the single dim
        integrated_dims = {Dimensions.Scene: s_range}
        for dim in [Dimensions.Time, Dimensions.Channel, Dimensions.SpatialZ]:
            if dim in read_dims:
                integrated_dims[dim] = range(read_dims[dim], read_dims[dim] + 1)
            else:
                integrated_dims[dim] = range(*s_dict[dim])

        return integrated_dims

    @staticmethod
    def get_pixel_type(meta: Element, scene: int = 0) -> np.dtype:
        """
        This function parses the metadata to assign the appropriate numpy.dtype

        Parameters
        ----------
        meta: xml.etree.ElementTree.Element
            The root Element of the metadata etree
        scene: int
            The index of the scene, scenes could have different storage data types.

        Returns
        -------
        numpy.dtype
            The appropriate data type to construct the matrix with.

        """
        # Due to the 12 bit values being stored in a uint16 the raw data is a little
        # fussy to get the contrast correct.
        p_types = {
            8: np.uint8,
            12: np.dtype("<u2"),  # little endian uint16
            16: np.dtype("<u2"),  # little endian uint16
            32: np.dtype("<u4"),  # little endian uint32 ** untested
            64: np.dtype("<u8"),  # little endian uint64 ** untested
        }
        img_sets = meta.findall(".//Image")

        img = img_sets[scene]
        chs = img.findall(".//ChannelDescription")
        resolution = set([ch.attrib["Resolution"] for ch in chs])
        if len(resolution) != 1:
            raise exceptions.InconsistentPixelType(
                f"Metadata contains two conflicting "
                f"Resolution attributes: {resolution}"
            )

        return p_types[int(resolution.pop())]

    @staticmethod
    def _get_array_from_offset(
        im_path: Path,
        offsets: List[np.ndarray],
        read_lengths: np.ndarray,
        meta: Element,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Gets specified bitmap data from the lif file (private).

        Parameters
        ----------
        im_path: Path
            Path to the LIF file to read.
        offsets: List[numpy.ndarray]
            A List of numpy ndarrays offsets, see _compute_offsets for more details.
        read_lengths: numpy.ndarray
            A 1D numpy array of read lengths, the index is the scene index
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)

        Returns
        -------
        numpy.ndarray
            a stack of images as a numpy.ndarray
        List[Tuple[str, int]]
            The shape of the data being returned
        """
        if read_dims is None:
            read_dims = {}

        lif = LifFile(im_path)

        # Data has already been checked for consistency. The dims are either consistent
        # or S is specified selected_ranges get's the ranges for the Dimension for the
        # range unless the dim is explicitly specified
        selected_ranges = LifReader._read_dims_to_ranges(lif, read_dims)
        s_index = read_dims[Dimensions.Scene] if Dimensions.Scene in read_dims else 0
        lif_img = lif.get_image(img_n=s_index)
        x_size = lif_img.dims[0]
        y_size = lif_img.dims[1]
        pixel_type = LifReader.get_pixel_type(meta, s_index)

        # The ranged dims
        ranged_dims = [
            (dim, len(selected_ranges[dim]))
            for dim in [
                Dimensions.Scene,
                Dimensions.Time,
                Dimensions.Channel,
                Dimensions.SpatialZ,
            ]
        ]

        img_stack = []

        # Loop through the dim ranges to return the requested image stack
        with open(str(im_path), "rb") as image:
            for s_index in selected_ranges[Dimensions.Scene]:
                for t_index in selected_ranges[Dimensions.Time]:
                    for c_index in selected_ranges[Dimensions.Channel]:
                        for z_index in selected_ranges[Dimensions.SpatialZ]:
                            # Use the precalculated offset to jump to the begining of
                            # the desired YX plane
                            image.seek(offsets[s_index][t_index, c_index, z_index])
                            # Read the image data as a bytearray
                            byte_array = image.read(read_lengths[s_index])
                            # Convert the bytearray to a the type pixel_type
                            typed_array = np.frombuffer(
                                byte_array, dtype=pixel_type
                            ).reshape(x_size, y_size)
                            # LIF stores YX planes so transpose them to get YX
                            typed_array = typed_array.transpose()
                            # Append the YX plane to the image stack.
                            img_stack.append(typed_array)

        shape = [len(selected_ranges[dim[0]]) for dim in ranged_dims]
        shape.append(y_size)
        shape.append(x_size)
        ranged_dims.append((Dimensions.SpatialY, y_size))
        ranged_dims.append((Dimensions.SpatialX, x_size))
        return (
            np.array(img_stack).reshape(*shape),
            ranged_dims,
        )  # in some subset of STCZYX order

    @staticmethod
    def _read_image(
        img: Path,
        offsets: List[np.ndarray],
        read_lengths: np.ndarray,
        meta: Element,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info
        that was read.

        Parameters
        ----------
        img: Path
            Path to the LIF file to read.
        offsets: List[numpy.ndarray]
            A List of numpy ndarrays offsets, see _compute_offsets for more details.
        read_lengths: numpy.ndarray
            A 1D numpy array of read lengths, the index is the scene index
        meta: xml.etree.ElementTree.Element
            The root element of the metadata etree from the file.
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)

        Returns
        -------
        data: np.ndarray
            The data read for the dimensions provided.
        read_dimensions: List[Tuple[str, int]]]
            The dimension sizes that were returned from the read.
        """
        # Catch optional read dim
        if read_dims is None:
            read_dims = {}

        # Read image
        log.debug(f"Reading dimensions: {read_dims}")
        data, dims = LifReader._get_array_from_offset(
            img, offsets, read_lengths, meta, read_dims
        )

        # Drop dims so that the data dims match the chunk_dims for dask
        ops = []
        real_dims = []
        for i, dim_info in enumerate(dims):
            # Expand dimension info
            dim, size = dim_info

            # If the dim was provided in the read dims we know a single plane for that
            # dimension was requested so remove it
            if dim in read_dims:
                ops.append(0)
            # Otherwise just read the full slice
            else:
                ops.append(slice(None, None, None))
                real_dims.append(dim_info)

        # Convert ops and run getitem
        return data[tuple(ops)], real_dims

    @staticmethod
    def _imread(
        img: Path,
        offsets: List[np.ndarray],
        read_lengths: np.ndarray,
        meta: Element,
        read_dims: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        This function is a pass through to _read_image above
        the difference is it returns the data without the dims structure.
        Parameters
        ----------
        img: Path
            Path to the LIF file to read.
        offsets: List[numpy.ndarray]
            A List of numpy ndarrays offsets, see _compute_offsets for more details.
        read_lengths: numpy.ndarray
            A 1D numpy array of read lengths, the index is the scene index
        meta: xml.etree.ElementTree.Element
            The root element of the metadata etree from the file.
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)

        Returns
        -------
        data: np.ndarray
            The data read for the dimensions provided.
        """
        data, dims = LifReader._read_image(
            img=img,
            offsets=offsets,
            read_lengths=read_lengths,
            meta=meta,
            read_dims=read_dims,
        )
        return data

    @staticmethod
    def _daread(
        img: Path,
        offsets: List[np.ndarray],
        read_lengths: np.ndarray,
        chunk_by_dims: List[str] = [
            Dimensions.SpatialZ,
            Dimensions.SpatialY,
            Dimensions.SpatialX,
        ],
        S: int = 0,
    ) -> Tuple[da.core.Array, str]:
        """
        Read a LIF image file as a delayed dask array where certain dimensions act as
        the chunk size.

        Parameters
        ----------
        img: Path
            The filepath to read.
        offsets: List[numpy.ndarray]
            A List of numpy ndarrays offsets, see _compute_offsets for more details.
        read_lengths: numpy.ndarray
            A 1D numpy array of read lengths, the index is the scene index
        chunk_by_dims: List[str]
            The dimensions to use as the for mapping the chunks / blocks.
            Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
            Note: SpatialY and SpatialX will always be added to the list if not present.
        S: int
            If the image has different dimensions on any scene from another, the dask
            array construction will fail.
            In that case, use this parameter to specify a specific scene to construct a
            dask array for.
            Default: 0 (select the first scene)

        Returns
        -------
        img: dask.array.core.Array
            The constructed dask array where certain dimensions are chunked.
        dims: str
            The dimension order as a string.
        """
        # Get image dims indicies
        lif = LifFile(filename=img)
        image_dim_indices = LifReader._dims_shape(lif=lif)

        # Catch inconsistent scene dimension sizes
        if len(image_dim_indices) > 1:
            # Choose the provided scene
            try:
                image_dim_indices = image_dim_indices[S]
                log.info(
                    f"File contains variable dimensions per scene, "
                    f"selected scene: {S} for data retrieval."
                )
            except IndexError:
                raise exceptions.InconsistentShapeError(
                    f"The LIF image provided has variable dimensions per scene. "
                    f"Please provide a valid index to the 'S' parameter to create a "
                    f"dask array for the index provided. "
                    f"Provided scene index: {S}. Scene index range: "
                    f"0-{len(image_dim_indices)}."
                )
        else:
            # If the list is length one that means that all the scenes in the image
            # have the same dimensions
            # Just select the first dictionary in the list
            image_dim_indices = image_dim_indices[0]

        # Uppercase dimensions provided to chunk by dims
        chunk_by_dims = [d.upper() for d in chunk_by_dims]

        # Always add Y and X dims to chunk by dims because that is how LIF files work
        if Dimensions.SpatialY not in chunk_by_dims:
            log.info(
                "Adding the Spatial Y dimension to chunk by dimensions as it was not "
                "found."
            )
            chunk_by_dims.append(Dimensions.SpatialY)
        if Dimensions.SpatialX not in chunk_by_dims:
            log.info(
                "Adding the Spatial X dimension to chunk by dimensions as it was not "
                "found."
            )
            chunk_by_dims.append(Dimensions.SpatialX)

        # Setup read dimensions for an example chunk
        first_chunk_read_dims = {}
        for dim, (dim_begin_index, dim_end_index) in image_dim_indices.items():
            # Only add the dimension if the dimension isn't a part of the chunk
            if dim not in chunk_by_dims:
                # Add to read dims
                first_chunk_read_dims[dim] = dim_begin_index

        # Read first chunk for information used by dask.array.from_delayed
        sample, sample_dims = LifReader._get_array_from_offset(
            im_path=img,
            offsets=offsets,
            read_lengths=read_lengths,
            meta=lif.xml_root,
            read_dims=first_chunk_read_dims,
        )

        # Get the shape for the chunk and operating shape for the dask array
        # We also collect the chunk and non chunk dimension ordering so that we can
        # swap the dimensions after we block the dask array together.
        sample_chunk_shape = []
        operating_shape = []
        non_chunk_dimension_ordering = []
        chunk_dimension_ordering = []
        for i, dim_info in enumerate(sample_dims):
            # Unpack dim info
            dim, size = dim_info

            # If the dim is part of the specified chunk dims then append it to the
            # sample, and, append the dimension to the chunk dimension ordering
            if dim in chunk_by_dims:
                sample_chunk_shape.append(size)
                chunk_dimension_ordering.append(dim)

            # Otherwise, append the dimension to the non chunk dimension ordering, and,
            # append the true size of the image at that dimension
            else:
                non_chunk_dimension_ordering.append(dim)
                operating_shape.append(
                    image_dim_indices[dim][1] - image_dim_indices[dim][0]
                )

        # Convert shapes to tuples and combine the non and chunked dimension orders as
        # that is the order the data will actually come out of the read data as
        sample_chunk_shape = tuple(sample_chunk_shape)
        blocked_dimension_order = (
            non_chunk_dimension_ordering + chunk_dimension_ordering
        )

        # Fill out the rest of the operating shape with dimension sizes of 1 to match
        # the length of the sample chunk. When dask.block happens it fills the
        # dimensions from inner-most to outer-most with the chunks as long as the
        # dimension is size 1. Basically, we are adding empty dimensions to the
        # operating shape that will be filled by the chunks from dask
        operating_shape = tuple(operating_shape) + (1,) * len(sample_chunk_shape)

        # Create empty numpy array with the operating shape so that we can iter through
        # and use the multi_index to create the readers.
        lazy_arrays = np.ndarray(operating_shape, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims
        # dictionaries by simply zipping together the ordered dims list and the current
        # multi-index plus the begin index for that plane. We then set the value of the
        # array at the same multi-index to the delayed reader using the constructed
        # read_dims dictionary.
        dims = [d for d in Dimensions.DefaultOrder]
        begin_indicies = tuple(image_dim_indices[d][0] for d in dims)
        for i, _ in np.ndenumerate(lazy_arrays):
            # Add the czi file begin index for each dimension to the array dimension
            # index
            this_chunk_read_indicies = (
                current_dim_begin_index + curr_dim_index
                for current_dim_begin_index, curr_dim_index in zip(begin_indicies, i)
            )

            # Zip the dims with the read indices
            this_chunk_read_dims = dict(
                zip(blocked_dimension_order, this_chunk_read_indicies)
            )

            # Remove the dimensions that we want to chunk by from the read dims
            for d in chunk_by_dims:
                if d in this_chunk_read_dims:
                    this_chunk_read_dims.pop(d)

            # Add delayed array to lazy arrays at index
            lazy_arrays[i] = da.from_delayed(
                delayed(LifReader._imread)(
                    img, offsets, read_lengths, lif.xml_root, this_chunk_read_dims
                ),
                shape=sample_chunk_shape,
                dtype=sample.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array and fill the inner
        # most empty dimensions with chunks
        merged = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example being, if the original dimension ordering was "SZYX" and we want to
        # chunk by "S", "Y", and "X" we created an array with dimensions ordering "ZSYX"
        transpose_indices = []
        transpose_required = False
        for i, d in enumerate(Dimensions.DefaultOrder):
            new_index = blocked_dimension_order.index(d)
            if new_index != i:
                transpose_required = True
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        # The default case is "Z", "Y", "X", which _usually_ doesn't need to be
        # transposed because that is _usually_
        # The normal dimension order of the LIF file anyway
        if transpose_required:
            merged = da.transpose(merged, tuple(transpose_indices))

        # Because dimensions outside of Y and X can be in any order and present or not
        # we also return the dimension order string.
        return merged, "".join(dims)

    def _read_delayed(self) -> da.core.Array:
        """
        Returns
        -------
        Constructed dask array where each chunk is a delayed read from the LIF file.
        Places dimensions in the native order (i.e. "TZCYX")
        """
        dask_array, _ = LifReader._daread(
            self._file,
            self._chunk_offsets,
            self._chunk_lengths,
            chunk_by_dims=self.chunk_by_dims,
            S=self.specific_s_index,
        )
        return dask_array

    def _read_immediate(self) -> np.ndarray:
        # Get image dims indicies
        lif = LifFile(filename=self._file)
        image_dim_indices = LifReader._dims_shape(lif=lif)

        # Catch inconsistent scene dimension sizes
        if len(image_dim_indices) > 1:
            # Choose the provided scene
            log.info(
                f"File contains variable dimensions per scene, "
                f"selected scene: {self.specific_s_index} for data retrieval."
            )
            data, _ = LifReader._get_array_from_offset(
                self._file,
                self._chunk_offsets,
                self._chunk_lengths,
                self.metadata,
                {Dimensions.Scene: self.specific_s_index},
            )

        else:
            # If the list is length one that means that all the scenes in the image
            # have the same dimensions
            # Read all data in the image
            data, _ = LifReader._get_array_from_offset(
                self._file, self._chunk_offsets, self._chunk_lengths, self.metadata,
            )

        return data

    def dtype(self) -> np.dtype:
        """
        The data type of the underlying numpy ndarray, ie uint8, uint16, uint32 etc.

        Returns
        -------
        numpy.dtype
            The data format used to store the data in the Leica lif file and the read
            numpy.ndarray.
        """
        return self.dask_data.dtype

    @property
    def metadata(self) -> Element:
        """
        Load and return the metadata from the LIF file

        Returns
        -------
        The xml Element Tree of the metadata
        """
        # We can't serialize xml element trees so don't save the tree to the object
        # state
        meta_xml, header = utilities.get_xml(self._file)
        return meta_xml

    def _size_of_dimension(self, dim: str) -> int:
        if dim in self.dims:
            return self.dask_data.shape[self.dims.index(dim)]

        return 1

    def size_s(self) -> int:
        return self._size_of_dimension(Dimensions.Scene)

    def size_t(self) -> int:
        return self._size_of_dimension(Dimensions.Time)

    def size_c(self) -> int:
        return self._size_of_dimension(Dimensions.Channel)

    def size_z(self) -> int:
        return self._size_of_dimension(Dimensions.SpatialZ)

    def size_y(self) -> int:
        return self._size_of_dimension(Dimensions.SpatialY)

    def size_x(self) -> int:
        return self._size_of_dimension(Dimensions.SpatialX)

    def get_channel_names(self, scene: int = 0) -> List[str]:
        """
        Get the channel names for the scene

        Parameters
        ----------
        scene: int
            The index of the scene from which to retrieve the channel names

        Returns
        -------
        List[str]
            A list of descriptive names of the channels of the form
            "Gray--TL-BF--EMP_BF" and "Green--FLUO--GFP"
        """
        # For the remove to work the
        img_sets = self.metadata.findall(".//Image")

        img = img_sets[scene]
        scene_channel_list = []
        chs = img.findall(".//ChannelDescription")
        chs_details = img.findall(".//WideFieldChannelInfo")
        for ch in chs:
            ch_detail = next(
                x for x in chs_details if x.attrib["LUT"] == ch.attrib["LUTName"]
            )
            scene_channel_list.append(
                (
                    f"{ch_detail.attrib['LUT']}"
                    f"--{ch_detail.attrib['ContrastingMethodName']}"
                    f"--{ch_detail.attrib['FluoCubeName']}"
                )
            )
        return scene_channel_list

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        """
        Get the (X, Y, Z) pixel size. If the value is not set it returns 1.0.

        Parameters
        ----------
        scene: int
            The scene to retrieve the dimensions from

        Returns
        -------
        (X, Y, Z) in m.

        """
        # Find all the Image nodes in the xml tree. They correspond to the individual
        # scenes
        img_sets = self.metadata.findall(".//Image")

        # select the specified scene
        img = img_sets[scene]
        scene_pixel_size = [1.0, 1.0, 1.0]
        dim_list = []
        dim_list.append(img.findall(".//DimensionDescription[@DimID='1']")[0])
        dim_list.append(img.findall(".//DimensionDescription[@DimID='2']")[0])
        z_dim = img.findall(".//DimensionDescription[@DimID='3']")
        if len(z_dim) > 0:
            dim_list.append(z_dim[0])

        # Calculate and overwrite the pixel size for each X, Y, Z dim if present
        for idx, dim in enumerate(dim_list):
            # The formula for the pixel size is
            # pixel_size = Length/(NumberOfElements - 1) from Leica & ImageJ
            scene_pixel_size[idx] = abs(float(dim.attrib["Length"])) / (
                float(dim.attrib["NumberOfElements"]) - 1.0
            )
        return tuple(scene_pixel_size)

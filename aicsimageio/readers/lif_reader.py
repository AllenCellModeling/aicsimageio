#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import lxml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from readlif.reader import LifFile, LifImage
from readlif import reader, utilities
from dask import delayed
from lxml.etree import _Element

from .. import exceptions, types
from ..buffer_reader import BufferReader
from ..constants import Dimensions
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class LifReader(Reader):
    """
    LifReader wraps readlif.reader to provide the same reading capabilities but abstracts the specifics of using the
    backend library to create a unified interface. This enables higher level functions to duck type the File Readers.

    Parameters
    ----------
    data: types.FileLike
        A string or path to the CZI file to be read.
    chunk_by_dims: List[str]
        The dimensions to use as the for mapping the chunks / blocks.
        Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
        Note: SpatialY and SpatialX will always be added to the list if not present.
    S: int
        If the image has different dimensions on any scene from another, the dask array construction will fail.
        In that case, use this parameter to specify a specific scene to construct a dask array for.
        Default: 0 (select the first scene)
    """

    LIF_MAGIC_BYTE = 0x70
    LIF_MEMORY_BYTE = 0x2a

    ########################################################
    #
    #  Note the lif treats scenes as separate images in the lif file.
    #  Also once a Scene/Image is loaded the image data is retrieved
    #  2D XY plane by 2D XY plane meaning that if you have a Z stack.
    #  concretely if you have loaded an image into img and you then
    #  attempt to retrieve the XY plane with get_frame(c=0, t=0) and the
    #  image has 20 z slices get_frame will assume z=0 is the slice you want.
    #
    ########################################################

    def __init__(
        self,
        data: types.FileLike,
        chunk_by_dims: List[str] = [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX],
        S: int = 0,
        **kwargs
    ):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

        # Store parameters needed for _daread
        self.chunk_by_dims = chunk_by_dims
        self.specific_s_index = S
        lif = LifFile(filename=self._file)
        self._chunk_offsets, self._chunk_lengths = LifReader._compute_offsets(lif=lif)

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        with BufferReader(buffer) as buffer_reader:
            header = buffer_reader.read_bytes(n_bytes=8)

            if len(buffer_reader.endianness) < 2 or len(header) < 8:
                return False
            if buffer_reader.endianness[0] != LifReader.LIF_MAGIC_BYTE and header[1] != LifReader.LIF_MAGIC_BYTE:
                return False
            if header[6] == LifReader.LIF_MEMORY_BYTE:
                return True
        return False

    @staticmethod
    def _read_image(img: Path, offsets: np.ndarray, r_length: np.ndarray, read_dims: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info that was read.

        Parameters
        ----------
        img: Path
            Path to the LIF file to read.
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

        # Init czi
        # lif = LifFile(img)

        # Read image
        log.debug(f"Reading dimensions: {read_dims}")
        data, dims = LifReader._get_item_as_bitmap(img, offsets, r_length, **read_dims)

        return data, dims  # the data and dims, dimensions with fixed index have been squeezed out

    @staticmethod
    def _imread(img: Path, offsets: np.ndarray, r_length: np.ndarray, read_dims: Optional[Dict[str, str]] = None) -> np.ndarray:
        data, dims = LifReader._read_image(img=img,
                                           offsets=offsets,
                                           r_length=r_length,
                                           read_dims=read_dims
                                           )
        return data

    @staticmethod
    def _dims_shape(lif: LifFile):
        shape_list = [{'T': (0, img.nt),
                       'C': (0, img.channels),
                       'Z': (0, img.nz),
                       'Y': (0, img.dims[1]),
                       'X': (0, img.dims[0])}
                      for idx, img in enumerate(lif.get_iter_image())]
        consistent = all(elem == shape_list[0] for elem in shape_list)
        if consistent:
            shape_list[0]['S'] = (0, len(shape_list))
            shape_list = [shape_list[0]]
        else:
            for idx, lst in enumerate(shape_list):
                lst['S'] = (idx, idx+1)
        return shape_list

    @staticmethod
    def _daread(
        img: Path,
        lif: LifFile,
        offsets: np.ndarray,
        r_length: np.ndarray,
        chunk_by_dims: List[str] = [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX],
        S: int = 0
    ) -> Tuple[da.core.Array, str]:
        """
        Read a CZI image file as a delayed dask array where certain dimensions act as the chunk size.

        Parameters
        ----------
        img: Path
            The filepath to read.
        czi: CziFile
            The loaded CziFile object created from reading the filepath.
        chunk_by_dims: List[str]
            The dimensions to use as the for mapping the chunks / blocks.
            Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
            Note: SpatialY and SpatialX will always be added to the list if not present.
        S: int
            If the image has different dimensions on any scene from another, the dask array construction will fail.
            In that case, use this parameter to specify a specific scene to construct a dask array for.
            Default: 0 (select the first scene)

        Returns
        -------
        img: dask.array.core.Array
            The constructed dask array where certain dimensions are chunked.
        dims: str
            The dimension order as a string.
        """
        # Get image dims indicies
        image_dim_indices = LifReader._dims_shape(lif=lif)

        # Catch inconsistent scene dimension sizes
        if len(image_dim_indices) > 1:
            # Choose the provided scene
            try:
                image_dim_indices = image_dim_indices[S]
                log.info(f"File contains variable dimensions per scene, selected scene: {S} for data retrieval.")
            except IndexError:
                raise exceptions.InconsistentShapeError(
                    f"The LIF image provided has variable dimensions per scene. "
                    f"Please provide a valid index to the 'S' parameter to create a dask array for the index provided. "
                    f"Provided scene index: {S}. Scene index range: 0-{len(image_dim_indices)}."
                )
        else:
            # If the list is length one that means that all the scenes in the image have the same dimensions
            # Just select the first dictionary in the list
            image_dim_indices = image_dim_indices[0]

        # Uppercase dimensions provided to chunk by dims
        chunk_by_dims = [d.upper() for d in chunk_by_dims]

        # Always add Y and X dims to chunk by dims because that is how CZI files work
        if Dimensions.SpatialY not in chunk_by_dims:
            log.info(f"Adding the Spatial Y dimension to chunk by dimensions as it was not found.")
            chunk_by_dims.append(Dimensions.SpatialY)
        if Dimensions.SpatialX not in chunk_by_dims:
            log.info(f"Adding the Spatial X dimension to chunk by dimensions as it was not found.")
            chunk_by_dims.append(Dimensions.SpatialX)

        # Setup read dimensions for an example chunk
        first_chunk_read_dims = {}
        for dim, (dim_begin_index, dim_end_index) in image_dim_indices.items():
            # Only add the dimension if the dimension isn't a part of the chunk
            if dim not in chunk_by_dims:
                # Add to read dims
                first_chunk_read_dims[dim] = dim_begin_index

        # Read first chunk for information used by dask.array.from_delayed
        sample, sample_dims = LifReader._get_item_as_bitmap(img, offsets, r_length, **first_chunk_read_dims)
        # lif.read_image(**first_chunk_read_dims)

        # Get the shape for the chunk and operating shape for the dask array
        # We also collect the chunk and non chunk dimension ordering so that we can swap the dimensions after we
        # block the dask array together.
        sample_chunk_shape = []
        operating_shape = []
        non_chunk_dimension_ordering = []
        chunk_dimension_ordering = []
        for i, dim_info in enumerate(sample_dims):
            # Unpack dim info
            dim, size = dim_info

            # If the dim is part of the specified chunk dims then append it to the sample, and, append the dimension
            # to the chunk dimension ordering
            if dim in chunk_by_dims:
                sample_chunk_shape.append(size)
                chunk_dimension_ordering.append(dim)

            # Otherwise, append the dimension to the non chunk dimension ordering, and, append the true size of the
            # image at that dimension
            else:
                non_chunk_dimension_ordering.append(dim)
                operating_shape.append(image_dim_indices[dim][1] - image_dim_indices[dim][0])

        # Convert shapes to tuples and combine the non and chunked dimension orders as that is the order the data will
        # actually come out of the read data as
        sample_chunk_shape = tuple(sample_chunk_shape)
        blocked_dimension_order = non_chunk_dimension_ordering + chunk_dimension_ordering

        # Fill out the rest of the operating shape with dimension sizes of 1 to match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to outer-most with the chunks as long as
        # the dimension is size 1
        # Basically, we are adding empty dimensions to the operating shape that will be filled by the chunks from dask
        operating_shape = tuple(operating_shape) + (1, ) * len(sample_chunk_shape)

        # Create empty numpy array with the operating shape so that we can iter through and use the multi_index to
        # create the readers.
        lazy_arrays = np.ndarray(operating_shape, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims dictionaries by simply zipping together
        # the ordered dims list and the current multi-index plus the begin index for that plane. We then set the value
        # of the array at the same multi-index to the delayed reader using the constructed read_dims dictionary.
        dims = [d for d in LifReader.static_dims()]
        begin_indicies = tuple(image_dim_indices[d][0] for d in dims)
        for i, _ in np.ndenumerate(lazy_arrays):
            # Add the czi file begin index for each dimension to the array dimension index
            this_chunk_read_indicies = (
                current_dim_begin_index + curr_dim_index
                for current_dim_begin_index, curr_dim_index in zip(begin_indicies, i)
            )

            # Zip the dims with the read indices
            this_chunk_read_dims = dict(zip(blocked_dimension_order, this_chunk_read_indicies))

            # Remove the dimensions that we want to chunk by from the read dims
            for d in chunk_by_dims:
                if d in this_chunk_read_dims:
                    this_chunk_read_dims.pop(d)

            # Add delayed array to lazy arrays at index
            lazy_arrays[i] = da.from_delayed(
                delayed(LifReader._imread)(img, offsets, r_length, this_chunk_read_dims),
                shape=sample_chunk_shape,
                dtype=sample.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array and fill the inner-most empty dimensions with chunks
        merged = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example being, if the original dimension ordering was "SZYX" and we want to chunk by "S", "Y", and "X"
        # We created an array with dimensions ordering "ZSYX"
        transpose_indices = []
        transpose_required = False
        for i, d in enumerate(LifReader.static_dims()):
            new_index = blocked_dimension_order.index(d)
            if new_index != i:
                transpose_required = True
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        # The default case is "Z", "Y", "X", which _usually_ doesn't need to be transposed because that is _usually_
        # The normal dimension order of the CZI file anyway
        if transpose_required:
            merged = da.transpose(merged, tuple(transpose_indices))

        # Because dimensions outside of Y and X can be in any order and present or not
        # we also return the dimension order string.
        return merged, "".join(dims)

    @staticmethod
    def _daread_safe(
        img: Union[str, Path],
        offsets: np.ndarray,
        r_length: np.ndarray,
        chunk_by_dims: List[str] = [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX],
        S: int = 0
    ) -> Tuple[da.core.Array, str]:
        """
        Safely read a LIF image file as a delayed dask array where certain dimensions act as the chunk size.

        Parameters
        ----------
        img: Union[str, Path]
            The filepath to read.
        chunk_by_dims: List[str]
            The dimensions to use as the for mapping the chunks / blocks.
            Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
            Note: SpatialY and SpatialX will always be added to the list if not present.
        S: int
            If the image has different dimensions on any scene from another, the dask array construction will fail.
            In that case, use this parameter to specify a specific scene to construct a dask array for.
            Default: 0 (select the first scene)

        Returns
        -------
        img: dask.array.core.Array
            The constructed dask array where certain dimensions are chunked.
        dims: str
            The dimension order as a string.
        """
        # Resolve image path
        img = LifReader._resolve_image_path(img)

        # Init temp czi
        lif = LifFile(img)

        # Safely construct the dask array or catch any exception
        try:
            return LifReader._daread(img=img,
                                     lif=lif,
                                     offsets=offsets,
                                     r_length=r_length,
                                     chunk_by_dims=chunk_by_dims,
                                     S=S
                                     )

        except Exception as e:
            # A really bad way to close any connection to the CZI object
            lif._bytes = None
            lif.reader = None

            raise e

    @property
    def dask_data(self) -> da.core.Array:
        """
        Returns
        -------
        Constructed dask array where each chunk is a delayed read from the CZI file.
        Places dimensions in the native order (i.e. "TZCYX")
        """
        if self._dask_data is None:
            self._dask_data, self._dims = LifReader._daread_safe(
                self._file,
                self._chunk_offsets,
                self._chunk_lengths,
                chunk_by_dims=self.chunk_by_dims,
                S=self.specific_s_index
            )

        return self._dask_data

    @staticmethod
    def static_dims():
        return "STCZYX"

    @property
    def dims(self) -> str:
        return LifReader.static_dims()  # forcing 6 D

    def dtype(self) -> np.dtype:
        return self.dask_data.dtype

    @property
    def metadata(self) -> _Element:
        """
        Load and return the metadata from the CZI file

        Returns
        -------
        The lxml Element Tree of the metadata
        """
        # We can't serialize lxml element trees so don't save the tree to the object state
        if self._metadata:
            return self._metadata

        self._metadata, header = utilities.get_xml(self._file)
        return self._metadata

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

    @staticmethod
    def get_pixel_type(meta: _Element, scene: int = 0):
        p_types = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
        img_sets = meta.findall(".//Image")

        img = img_sets[scene]
        chs = img.findall(".//ChannelDescription")
        resolution = set([ch.attrib['Resolution'] for ch in chs])
        if len(resolution) != 1:
            raise exceptions.InconsistentPixelType(f"Metadata contains two conflicting "
                                                   f"Resolution attributes: {resolution}")

        return p_types[int(resolution.pop())]

    def get_channel_names(self, scene: int = 0):
        # for the remove to work the
        img_sets = self.metadata.findall(".//Image")

        img = img_sets[scene]
        scene_channel_list = []
        chs = img.findall(".//ChannelDescription")
        chs_details = img.findall(".//WideFieldChannelInfo")
        for ch in chs:
            ch_detail = next(x for x in chs_details if x.attrib["LUT"] == ch.attrib["LUTName"])
            scene_channel_list.append((f"{ch_detail.attrib['LUT']}--{ch_detail.attrib['ContrastingMethodName']}"
                                       f"--{ch_detail.attrib['FluoCubeName']}"))
        return scene_channel_list

    # TODO refactor this utility function into a metadata wrapper class
    def _getmetadataxmltext(self, findpath, default=None):
        ref = self.metadata.find(findpath)
        if ref is None:
            return default
        return ref.text

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        # for the remove to work the
        img_sets = self.metadata.findall(".//Image")

        img = img_sets[scene]
        scene_pixel_size = [1.0, 1.0, 1.0]
        dim_list = img.findall(".//DimensionDescription")
        for idx, dim in enumerate(dim_list):
            scene_pixel_size[idx] = (abs(1000000.0*float(dim.attrib['Length']))/(float(dim.attrib['NumberOfElements'])-1.0))
        return tuple(scene_pixel_size)

    #  bitmap reader functions

    @staticmethod
    def _get_item_as_bitmap(im_path: Path, offsets: np.ndarray, r_length: np.ndarray, **kwargs):
        """
        Gets specified item from the image set (private).
        Args:
            n (int): what item to retrieve

        Returns:
            PIL image
        """
        lif = LifFile(im_path)

        # data has already been checked for consistency. The dims are either consistent or S is specified
        selected_ranges = LifReader.kwargs_to_ranges(lif, **kwargs)
        s_index = kwargs[Dimensions.Scene] if Dimensions.Scene in kwargs.keys() else 0
        lif_img = lif.get_image(img_n=s_index)
        x_size = lif_img.dims[0]
        y_size = lif_img.dims[1]
        lir = LifReader(im_path)
        pixel_type = lir.get_pixel_type(lir.metadata, s_index)

        ranged_dims = [(dim, len(selected_ranges[dim])) for dim in [Dimensions.Scene,
                                                                    Dimensions.Time,
                                                                    Dimensions.Channel,
                                                                    Dimensions.SpatialZ]
                       ]

        img_stack = []
        with open(str(im_path), "rb") as image:
            for s_index in selected_ranges[Dimensions.Scene]:
                for t_index in selected_ranges[Dimensions.Time]:
                    for c_index in selected_ranges[Dimensions.Channel]:
                        for z_index in selected_ranges[Dimensions.SpatialZ]:
                            image.seek(offsets[s_index, t_index, c_index, z_index])
                            byte_array = image.read(r_length[s_index])
                            img_stack.append(
                                np.frombuffer(byte_array, dtype=pixel_type).reshape(x_size, y_size).transpose()
                            )

        shape = [len(selected_ranges[dim[0]]) for dim in ranged_dims]
        shape.append(y_size)
        shape.append(x_size)
        ranged_dims.append((Dimensions.SpatialY, y_size))
        ranged_dims.append((Dimensions.SpatialX, x_size))
        return np.array(img_stack).reshape(*shape), ranged_dims  # in some subset of STCZYX order

    @staticmethod
    def kwargs_to_ranges(lif: LifFile, **kwargs):

        data_shape = LifReader._dims_shape(lif=lif)

        if Dimensions.Scene in kwargs:
            s_range = range(kwargs[Dimensions.Scene], kwargs[Dimensions.Scene] + 1)
            s_dict = data_shape[s_range[0]]
        else:
            s_range = range(*data_shape[0][Dimensions.Scene])
            s_dict = data_shape[0]

        ans = {Dimensions.Scene: s_range}
        for ky in [Dimensions.Time, Dimensions.Channel, Dimensions.SpatialZ]:
            if ky in kwargs.keys():
                ans[ky] = range(kwargs[ky], kwargs[ky]+1)
            else:
                ans[ky] = range(*s_dict[ky])

        return ans

    @staticmethod
    def get_frame(img: LifImage, z: int = 0, t: int = 0, c: int = 0):
        """
        Gets the specified frame (z, t, c) from image.

        Args:
            img (LifImage): the Scene/Image within the LifFile
            z (int): z position
            t (int): time point
            c (int): channel

        Returns:
            Image as a numpy.ndarray (XY) object
        """
        t = int(t)
        c = int(c)
        z = int(z)
        if z >= img.nz:
            raise ValueError("Requested Z frame doesn't exist.")
        elif t >= img.nt:
            raise ValueError("Requested T frame doesn't exist.")
        elif c >= img.channels:
            raise ValueError("Requested channel doesn't exist.")

        total_items = img.channels * img.nz * img.nt

        t_offset = img.channels * img.nz
        t_requested = t_offset * t

        z_offset = img.channels
        z_requested = z_offset * z

        c_requested = c

        item_requested = t_requested + z_requested + c_requested
        if item_requested > total_items:
            raise ValueError("The requested item is after the end of the image")

        return LifReader._get_item_as_bitmap(img=img, n=item_requested)

    @staticmethod
    def _compute_offsets(lif: LifFile) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        lif : LifFile
            The LifFile object with an open file pointer to the file.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first numpy array holds the offsets and it's indexes are in STCZ order.
            The second numpy array holds the image read length per Scene.

        """
        s_list = []
        s_img_length_list = []
        for img in lif.get_iter_image():
            offsets = np.zeros(shape=(img.nt, img.channels, img.nz), dtype=np.uint64)
            t_offset = img.channels * img.nz
            z_offset = img.channels
            seek_distance = img.channels * img.dims[2] * img.dims[3]
            # self.offsets[1] is the length of the image
            if img.offsets[1] == 0:
                # In the case of a blank image, we can calculate the length from
                # the metadata in the LIF. When this is read by the parser,
                # it is set to zero initially.
                image_len = seek_distance * img.dims[0] * img.dims[1]
            else:
                image_len = int(img.offsets[1] / seek_distance)

            for t_index in range(img.nt):
                t_requested = t_offset * t_index
                for c_index in range(img.channels):
                    c_requested = c_index
                    for z_index in range(img.nz):
                        z_requested = z_offset * z_index
                        item_requested = t_requested + z_requested + c_requested
                        # self.offsets[0] is the offset in the file
                        offsets[t_index, c_index, z_index] = img.offsets[0] + image_len * item_requested

            s_list.append(offsets)
            s_img_length_list.append(image_len)

        return np.asarray(s_list, dtype=np.uint64), np.asarray(s_img_length_list, dtype=np.uint64)

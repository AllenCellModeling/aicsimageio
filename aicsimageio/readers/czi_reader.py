#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

import dask.array as da
import numpy as np
from aicspylibczi import CziFile
from dask import delayed

from .. import exceptions, types
from ..buffer_reader import BufferReader
from ..constants import Dimensions
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class CziReader(Reader):
    """
    CziReader wraps aicspylibczi to provide the same reading capabilities but abstracts the specifics of using the
    backend library to create a unified interface. This enables higher level functions to duck type the File Readers.

    Parameters
    ----------
    data: types.FileLike
        A string or path to the CZI file to be read.
    """
    ZEISS_2BYTE = b'ZI'             # First two characters of a czi file according to Zeiss docs
    ZEISS_10BYTE = b'ZISRAWFILE'    # First 10 characters of a well formatted czi file.

    def __init__(
        self,
        data: types.FileLike,
        chunk_by_dims: List[str] = [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX],
        S: Optional[int] = None,
        **kwargs
    ):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

        # Store parameters needed for _daread
        self.chunk_by_dims = chunk_by_dims
        self.specific_s_dim = S

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        with BufferReader(buffer) as buffer_reader:
            if buffer_reader.endianness != CziReader.ZEISS_2BYTE:
                return False
            header = buffer_reader.endianness + buffer_reader.read_bytes(8)
            return header == CziReader.ZEISS_10BYTE

    @staticmethod
    def _read_image(img: Path, read_dims: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        # Catch optional read dim
        if read_dims is None:
            read_dims = {}

        # Init czi
        czi = CziFile(img)

        # Read image
        log.debug(f"Reading dimensions: {read_dims}")
        data, dims = czi.read_image(**read_dims)

        # Drop dims that shouldn't be provided back
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
    def _imread(img: Path, read_dims: Optional[Dict[str, str]] = None) -> np.ndarray:
        print(read_dims)
        data, dims = CziReader._read_image(img=img, read_dims=read_dims)
        return data

    @staticmethod
    def _resolve_image_path(img: Union[str, Path]) -> Path:
        # Convert pathlike to CziFile
        if isinstance(img, (str, Path)):
            # Resolve path
            img = Path(img).expanduser().resolve(strict=True)

            # Check path
            if img.is_dir():
                raise IsADirectoryError(
                    f"Please provide a single file to the `img` parameter. "
                    f"Received directory: {img}"
                )

        # Check that no other type was provided
        if not isinstance(img, Path):
            raise TypeError(
                f"Please provide a path to a file as a string, or an pathlib.Path, to the "
                f"`img` parameter. "
                f"Received type: {type(img)}"
            )

        return img

    @staticmethod
    def _daread(
        img: Union[str, Path],
        chunk_by_dims: List[str] = [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX],
        S: Optional[int] = None
    ) -> da.core.Array:
        """
        Read a CZI image file as a delayed dask array where certain dimensions act as the chunk size.

        Parameters
        ----------
        img: Union[str, Path]
            The filepath to read.
        chunk_by_dims: List[str]
            The dimensions to use as the for mapping the chunks / blocks.
            Default: [Dimensions.SpatialZ, Dimensions.SpatialY, Dimensions.SpatialX]
            Note: SpatialY and SpatialX will always be added to the list if not present.
        S: Optional[int] = None
            If the image has different dimensions on any scene from another, the dask array construction will fail.
            In that case, use this parameter to specify a specific scene to construct a dask array for.

        Returns
        -------
        img: dask.array.core.Array
            The constructed dask array where certain dimensions are chunked.
        """
        # Resolve image path
        img = CziReader._resolve_image_path(img)

        # Init temp czi
        czi = CziFile(img)

        # Get image dims shape
        image_dims = czi.dims_shape()

        # Catch inconsistent scene dimension sizes
        if isinstance(image_dims, list):
            raise exceptions.InconsistentShapeError(
                f"The CZI image provided has variable dimensions per scene. "
                f"Please provide an index to the 'S' parameter to create a dask array for the index provided. "
                f"Per Scene dimension index ranges: "
                f"{image_dims}"
            )

        # Todo:
        # Handle specific S dim provided

        # Uppercase dimensions provided to chunk by dims
        chunk_by_dims = [d.upper() for d in chunk_by_dims]

        # Add Y and X dims to chunk by dims
        if Dimensions.SpatialY not in chunk_by_dims:
            log.info(f"Adding the Spatial Y dimension to chunk by dimensions as it was not found.")
            chunk_by_dims.append(Dimensions.SpatialY)
        if Dimensions.SpatialX not in chunk_by_dims:
            log.info(f"Adding the Spatial X dimension to chunk by dimensions as it was not found.")
            chunk_by_dims.append(Dimensions.SpatialX)

        # Setup read dimensions for an example chunk
        first_chunk_read_dims = {}
        for dim, dim_info in image_dims.items():
            # Only add the dimension if the dimension isn't a part of the chunk
            if dim not in chunk_by_dims:
                # Unpack dimension info
                dim_begin_index, dim_end_index = dim_info

                # Add to read dims
                first_chunk_read_dims[dim] = dim_begin_index

        # Read first chunk for information used by dask.array.from_delayed
        sample, sample_dims = czi.read_image(**first_chunk_read_dims)

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

            # If the dim is part of the specified chunk dims then append it to the sample, and,
            # append a dimension of size 1 to the operating shape
            if dim in chunk_by_dims:
                sample_chunk_shape.append(size)
                chunk_dimension_ordering.append(dim)

            # Otherwise, append it to the operating shape by getting it from the czi.size at the specified index
            else:
                non_chunk_dimension_ordering.append(dim)
                operating_shape.append(czi.size[i])

        # Convert shapes to tuples and combine the non and chunked dimension orders as that is the order the data will
        # actually come out of the read data as
        sample_chunk_shape = tuple(sample_chunk_shape)
        operating_shape = tuple(operating_shape) + (1, ) * len(sample_chunk_shape)
        blocked_dimension_order = non_chunk_dimension_ordering + chunk_dimension_ordering

        # Create empty numpy array with the operating shape so that we can iter through and use the multi_index to
        # create the readers.
        lazy_arrays = np.ndarray(operating_shape, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims dictionaries by simply zipping together
        # the ordered dims list and the current multi-index plus the begin index for that plane. We then set the value
        # of the array at the same multi-index to the delayed reader using the constructed read_dims dictionary.
        dims = [d for d in czi.dims]
        begin_indicies = tuple(image_dims[d][0] for d in dims)
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
                this_chunk_read_dims.pop(d)

            # Add array to lazy arrays at index from delayed
            lazy_arrays[i] = da.from_delayed(
                delayed(CziReader._imread)(img, this_chunk_read_dims),
                shape=sample_chunk_shape,
                dtype=sample.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array
        merged = da.block(lazy_arrays.tolist())

        # Transpose back to original dimension ordering
        transpose_indices = []
        transpose_required = False
        for i, d in enumerate(czi.dims):
            new_index = blocked_dimension_order.index(d)
            if new_index != i:
                transpose_required = True
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        if transpose_required:
            merged = da.transpose(merged, tuple(transpose_indices))

        # Because dimensions outside of Y and X can be in any order and present or not
        # we also return the dimension order string.
        return merged, "".join(dims)

    @property
    def data(self) -> da.core.Array:
        """
        Returns
        -------
        Constructed dask array where each chunk is a delayed read from the CZI file.
        Places dimensions in the native order (i.e. "TZCYX")
        """
        if self._data is None:
            self._data, self._dims = CziReader._daread(
                self._file,
                chunk_by_dims=self.chunk_by_dims,
                S=self.specific_s_dim
            )

        return self._data

    @property
    def dims(self) -> str:
        if self._dims is None:
            self._data, self._dims = CziReader._daread(
                self._file,
                chunk_by_dims=self.chunk_by_dims,
                S=self.specific_s_dim
            )

        return self._dims

    @property
    def metadata(self) -> ElementTree:
        """
        Lazy load the metadata from the CZI file

        Returns
        -------
        The xml Element Tree of the metadata
        """
        if self._metadata is None:
            # load the metadata
            self._metadata = self.czi.metadata
        return self._metadata

    def get_channel_names(self, scene: int = 0):
        chelem = self.metadata.findall("./Metadata/Information/Image/Dimensions/Channels/Channel")
        return [ch.get("Name") for ch in chelem]

    # TODO refactor this utility function into a metadata wrapper class
    def _getmetadataxmltext(self, findpath, default=None):
        ref = self.metadata.find(findpath)
        if ref is None:
            return default
        return ref.text

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        px = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='X']/Value", "1.0"))
        py = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Y']/Value", "1.0"))
        pz = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Z']/Value", "1.0"))
        return (px, py, pz)

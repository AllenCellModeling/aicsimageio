#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, List, Optional, Tuple, Type

import dask.array as da
import numpy as np

from . import transforms, types
from .constants import Dimensions
from .exceptions import (InvalidDimensionOrderingError,
                         UnsupportedFileFormatError)
from .readers import (ArrayLikeReader, CziReader, DefaultReader, OmeTiffReader,
                      TiffReader)
from .readers.reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# The order of the readers in this list is important.
# Example: if TiffReader was placed before OmeTiffReader, we would never hit the OmeTiffReader.
SUPPORTED_READERS = [ArrayLikeReader, CziReader, OmeTiffReader, TiffReader, DefaultReader]

###############################################################################


class AICSImage:
    """
    AICSImage takes microscopy image data types (files) of varying dimensions ("ZYX", "TCZYX", "CYX") and
    puts them into a consistent 6D "STCZYX" ordered dask array. The data, metadata are lazy loaded and can be
    accessed as needed. Note the dims are assumed to match "STCZYX" from right to left meaning if 4 dimensional
    data is provided then the dimensions are assigned to be "CZYX", 2 dimensional would be "YX". This guessed assignment
    is only for file types without dimension metadata (i.e. not .ome.tiff or .czi).

    Simple Example
    --------------
    img = AICSImage("my_file.tiff")
    data = img.data  # data is a 6D "STCZYX" dask array
    metadata = img.metadata  # metadata from the file, an xml.etree
    zstack_t8 = img.get_image_data("ZYX", S=0, T=8, C=0)  # returns a 3D "ZYX" numpy array
    zstack_t10 = data[0, 10, 0, :, :, :]  # access the S=0, T=10, C=0 "ZYX" dask array
    zstack_t10_np = zstack_t10.compute()  # read the data from the file and convert from dask array to numpy array

    File Examples
    -------------
    OME-TIFF
        img = AICSImage("filename.ome.tiff")
    CZI (Zeiss)
        img = AICSImage("filename.czi")
    TIFF
        img = AICSImage("filename.tiff")
    PNG / GIF / ...
        img = AICSImage("filename.png")
        img = AICSImage("filename.gif")

    dask.array.Array Example
    ------------------------
    blank = dask.array.zeros((2, 600, 900))
    img = AICSImage(blank)

    numpy.ndarray Example
    ---------------------
    blank = numpy.zeros((2, 600, 900))
    img = AICSImage(blank)
    """

    def __init__(
        self,
        data: types.ImageLike,
        known_dims: Optional[str] = None,
        **kwargs
    ):
        """
        Constructor for AICSImage class intended for providing a unified interface for dealing with
        microscopy images. To extend support to a new reader simply add a new reader child class of
        Reader ([readers/reader.py]) and add the class to SUPPORTED_READERS variable.

        Parameters
        ----------
        data: types.ImageLike
            String with path to file, numpy.ndarray, or dask.array.Array with up to six dimensions.
        known_dims: Optional[str]
            Optional string with the known dimension order. If None, the reader will attempt to parse dim order.
        """
        # Check known dims
        if known_dims is not None:
            if not all([d in Dimensions.DefaultOrder for d in known_dims]):
                raise InvalidDimensionOrderingError(
                    f"The provided dimension string to the 'known_dims' argument includes dimensions that AICSImage "
                    f"does not support. Received: '{known_dims}'. Supported dimensions: {Dimensions.DefaultOrderList}."
                )

        # Hold onto known dims until data is requested
        self._known_dims = known_dims

        # Dims should nearly always be default dim order unless explictly overridden
        self.dims = Dimensions.DefaultOrder

        # Determine reader class and create dask delayed array
        reader_class = self.determine_reader(data=data)
        self._reader = reader_class(data, **kwargs)

        # Lazy load data from reader and reformat to standard dimensions
        self._dask_data = None
        self._data = None
        self._metadata = None

    @staticmethod
    def determine_reader(data: types.ImageLike) -> Type[Reader]:
        """
        Cheaply check to see if a given file is a recognized type and return the appropriate reader for the file.
        """
        # Iterate through the ordered supported readers to find the right one
        for reader_class in SUPPORTED_READERS:
            if reader_class.is_this_type(data):
                return reader_class

        raise UnsupportedFileFormatError(data)

    @property
    def dask_data(self) -> da.core.Array:
        """
        Returns
        -------
        Returns a dask array with dimension ordering "STCZYX"
        """
        # Construct dask array if never before constructed
        if self._dask_data is None:
            reader_data = self._reader.dask_data

            # Read and reshape and handle delayed known dims reshape
            self._dask_data = transforms.reshape_data(
                data=reader_data,
                given_dims=self._known_dims or self.reader.dims,
                return_dims=self.dims,
            )

        return self._dask_data

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self.dask_data.compute()

        return self._data

    def size(self, dims: str = Dimensions.DefaultOrder) -> Tuple[int]:
        """
        Parameters
        ----------
        dims: str
            A string containing a list of dimensions being requested. The default is to return the six standard dims.

        Returns
        -------
        Returns a tuple with the requested dimensions filled in
        """
        # Ensure dims is an uppercase string
        dims = dims.upper()

        # Check that dims requested are all a part of the available dims in the package
        if not (all(d in Dimensions.DefaultOrder for d in dims)):
            raise InvalidDimensionOrderingError(f"Invalid dimensions requested: {dims}")

        # Check that the dims requested are in the image dims
        if not (all(d in self.dims for d in dims)):
            raise InvalidDimensionOrderingError(f"Invalid dimensions requested: {dims}")

        # Return the shape of the data for the dimensions requested
        return tuple([self.dask_data.shape[self.dims.index(dim)] for dim in dims])

    @property
    def size_x(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Spatial X dimension.
        """
        return self.size(Dimensions.SpatialX)[0]

    @property
    def size_y(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Spatial Y dimension.
        """
        return self.size(Dimensions.SpatialY)[0]

    @property
    def size_z(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Spatial Z dimension.
        """
        return self.size(Dimensions.SpatialZ)[0]

    @property
    def size_c(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Channel dimension.
        """
        return self.size(Dimensions.Channel)[0]

    @property
    def size_t(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Time dimension.
        """
        return self.size(Dimensions.Time)[0]

    @property
    def size_s(self) -> int:
        """
        Returns
        -------
        size: int
            The size of the Scene dimension.
        """
        return self.size(Dimensions.Scene)[0]

    @property
    def metadata(self) -> Any:
        """
        Returns
        -------
        metadata: Any
            The Metadata from the Czi, or Ome.Tiff file, or other base class type with metadata.
            For pure image files an empty string or None is returned.
        """
        if self._metadata is None:
            self._metadata = self._reader.metadata

        return self._metadata

    @property
    def reader(self) -> Reader:
        """
        This property returns the class created to read the image file type.
        The intent is that if the AICSImage class doesn't provide a raw enough
        interface then the base class can be used directly.

        Returns
        -------
        reader: Reader
            A child of Reader; CziReader OmeTiffReader, TiffReader, DefaultReader, etc.
        """
        return self._reader

    def get_image_dask_data(self, out_orientation: Optional[str] = None, **kwargs) -> da.core.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims`

        kwargs:
            C=1: specifies Channel 1
            T=3: specifies the fourth index in T
            D=n: D is Dimension letter and n is the index desired D should not be present in the out_orientation

        Returns
        -------
        data: dask array
            The read data with the dimension ordering that was specified with out_orientation.

        Note: If a requested dimension is not present in the data the dimension is added with
        a depth of 1.
        """
        # If no out orientation, simply return current data as numpy array
        if out_orientation is None:
            return self.dask_data

        # Transform and return
        return transforms.reshape_data(
            data=self.dask_data,
            given_dims=self.dims,
            return_dims=out_orientation,
            **kwargs,
        )

    def get_image_data(self, out_orientation: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Get specific dimension image data out of an image as a numpy array.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims`

        kwargs:
            C=1: specifies Channel 1
            T=3: specifies the fourth index in T
            D=n: D is Dimension letter and n is the index desired D should not be present in the out_orientation

        Returns
        -------
        data: np.ndarray
            The read data with the dimension ordering that was specified with out_orientation.

        Note: If a requested dimension is not present in the data the dimension is added with
        a depth of 1.
        """
        return self.get_image_dask_data(out_orientation=out_orientation, **kwargs).compute()

    def get_channel_names(self, scene: int = 0) -> List[str]:
        """
        Get the image's channel names.

        Parameters
        ----------
        scene: the index of the scene for which to return channel names

        Returns
        -------
        list of strings representing the channel names
        """
        try:
            names = self._reader.get_channel_names(scene)
        except AttributeError:
            names = [str(i) for i in range(self.size_c)]
        return names

    def __repr__(self) -> str:
        return f"<AICSImage [{type(self.reader).__name__}]>"


def imread_dask(data: types.ImageLike, **kwargs) -> da.core.Array:
    """
    Read image as a dask array.

    Parameters
    ----------
    data: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    kwargs: Dict[str, Any]
        Any extra arguments to passed down to AICSImage and subsequent readers.

    Returns
    -------
    data: da.core.Array
        The image read and configured as a dask array.
    """
    return AICSImage(data, **kwargs).dask_data


def imread(data: types.ImageLike, **kwargs) -> np.ndarray:
    """
    Read image as a numpy ndarray.

    Parameters
    ----------
    data: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    kwargs: Dict[str, Any]
        Any extra arguments to passed down to AICSImage and subsequent readers.

    Returns
    -------
    data: np.ndarray
        The image read and configured as a numpy ndarray.
    """
    return AICSImage(data, **kwargs).data

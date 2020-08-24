#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Optional, Tuple

import dask.array as da
import numpy as np

from . import types
from .dimensions import DEFAULT_DIMENSION_ORDER, Dimensions
from .readers.reader import Reader

###############################################################################


class AICSImage:

    # The order of the readers in this list is important.
    # Example:
    # if TiffReader was placed before OmeTiffReader,
    # we would never hit the OmeTiffReader
    # SUPPORTED_READERS = [
    #     ArrayLikeReader,
    #     CziReader,
    #     LifReader,
    #     OmeTiffReader,
    #     TiffReader,
    #     DefaultReader,
    # ]

    def __init__(
        self, image: types.ImageLike, known_dims: Optional[str] = None, **kwargs
    ):
        """
        AICSImage takes microscopy image data types (files or arrays) of varying
        dimensions ("ZYX", "TCZYX", "CYX") and puts them into a consistent 6D "STCZYX"
        ("Scene-Time-Channel-Z-Y-X") ordered array. The data and metadata are lazy
        loaded and can be accessed as needed.

        Parameters
        ----------
        image: types.ImageLike
            A string, Path, fsspec based file, or, numpy or dask array, to read.
        known_dims: Optional[str]
            Optional string with the known dimension order. If None, the reader will
            attempt to parse dim order.
        kwargs: Any
            Extra keyword arguments that will be passed down to the reader subclass.

        Examples
        --------
        Initialize an image and read the slices specified as a numpy array.

        >>> img = AICSImage("my_file.tiff")
        ... zstack_t8 = img.get_image_data("ZYX", S=0, T=8, C=0)

        Initialize an image, construct a delayed dask array for certain slices, then
        read the data.

        >>> img = AICSImage("my_file.czi")
        ... zstack_t8 = img.get_image_dask_data("ZYX", S=0, T=8, C=0)
        ... zstack_t8_data = zstack_t8.compute()

        Initialize an image with a dask or numpy array.

        >>> data = np.random.rand(100, 100)
        ... img = AICSImage(data)

        Initialize an image from S3 with s3fs.

        >>> fs = s3fs.S3FileSystem("s3://my_bucket")
        ... f = fs.open("s3://my_bucket/my_file.tiff")
        ... img = AICSImage(f)

        Initialize an image and pass arguments to the reader using kwargs.

        >>> img = AICSImage("my_file.czi", chunk_by_dims=["T", "Y", "X"])

        Notes
        -----
        Constructor for AICSImage class intended for providing a unified interface for
        dealing with microscopy images. To extend support to a new reader simply add a
        new reader child class of Reader ([readers/reader.py]) and add the class to
        SUPPORTED_READERS variable.
        """
        pass

    @staticmethod
    def determine_reader(image: types.ImageLike) -> Reader:
        """
        Cheaply check to see if a given file is a recognized type and return the
        appropriate reader for the file.
        """
        pass

    @property
    def reader(self) -> Reader:
        """
        This property returns the object created to read the image file type.
        The intent is that if the AICSImage class doesn't provide a raw enough
        interface then the base class can be used directly.
        """
        pass

    @property
    def dask_data(self) -> da.Array:
        """
        Returns the image as a dask array with dimension ordering "STCZYX".
        """
        pass

    @property
    def data(self) -> np.ndarray:
        """
        Return the image as a numpy array with dimension ordering "STCZYX".
        """
        pass

    @property
    def dtype(self) -> np.dtype:
        """
        Data-type of the image array's elements.
        """
        pass

    @property
    def shape(self) -> Tuple[int]:
        """
        Tuple of the image array's dimensions.
        """
        pass

    @property
    def dims(self) -> Dimensions:
        """
        Object with the paired dimension names and their sizes.
        """
        pass

    def get_image_dask_data(
        self, out_orientation: Optional[str] = None, **kwargs
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims.order`

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the out_orientation.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indicies
              desired. D should be present in the out_orientation.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indicies
              desired. D should be present in the out_orientation.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the out_orientation.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the out_orientation.

        Returns
        -------
        data: dask array
            The image data with the specified dimension ordering.

        Examples
        --------
        Specific index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_dask_data("ZYX", C=1)

        List of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_dask_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_dask_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_dask_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        pass

    def get_image_data(
        self, out_orientation: Optional[str] = None, **kwargs
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a numpy array.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims.order`

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the out_orientation.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indicies
              desired. D should be present in the out_orientation.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indicies
              desired. D should be present in the out_orientation.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the out_orientation.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the out_orientation.

        Returns
        -------
        data: numpy array
            The image data with the specified dimension ordering.

        Examples
        --------
        Specific index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_data("ZYX", C=1)

        List of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = AICSImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        pass

    @property
    def metadata(self) -> Any:
        """
        Passthrough to the base image reader metadata property.
        For more information, see the specific image format reader you are using
        for details on it's metadata property.
        """
        pass

    def get_channel_names(self, scene: int = 0) -> List[str]:
        """
        Attempts to use the available metadata for the image to return the channel
        names for the image.

        If no channel metadata is available, returns a list of string indicies for each
        channel in the image.

        Parameters
        ----------
        scene: int
            The scene index to return channel names for.

        Returns
        -------
        channel_names: List[str]
            List of strings representing channel names.
        """
        pass

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        """
        Attempts to use the available metadata for the image to return the physical
        pixel sizes for the image.

        If no pixel metadata is available, returns `1.0` for each spatial dimension.

        Paramaters
        ----------
        scene: int
            The scene index to return physical pixel sizes for.

        Returns
        -------
        sizes: Tuple[float]
            Tuple of floats representing physical pixel sizes for dimensions X, Y, Z
            (in that order).
        """
        pass

    def save(
        self,
        filepath: types.PathLike,
        save_dims: str = DEFAULT_DIMENSION_ORDER,
        **kwargs,
    ):
        """
        Save the image to a file.

        Parameters
        ----------
        filepath: types.FileLike
            The path to save the image and metadata to.
            The image writer is determined based off of the file extension included in
            this path.
        save_dims: str
            The selected dimensions to save out.
        kwargs: Any
            Extra keyword arguments that will be passed down to the writer subclass.
        """
        pass

    def __str__(self) -> str:
        return f"<AICSImage [{type(self.reader).__name__} -- {self.shape}]>"

    def __repr__(self) -> str:
        return str(self)


def imread_dask(image: types.ImageLike, **kwargs) -> da.Array:
    """
    Read image as a dask array.

    Parameters
    ----------
    image: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    kwargs: Any
        Extra keyword arguments that will be passed down to the reader subclass.

    Returns
    -------
    data: da.core.Array
        The image read and configured as a dask array.
    """
    return AICSImage(image, **kwargs).dask_data


def imread(image: types.ImageLike, **kwargs) -> np.ndarray:
    """
    Read image as a numpy array.

    Parameters
    ----------
    image: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    kwargs: Any
        Extra keyword arguments that will be passed down to the reader subclass.

    Returns
    -------
    data: np.ndarray
        The image read and configured as a numpy ndarray.
    """
    return AICSImage(image, **kwargs).data


def imwrite(
    data: types.ArrayLike,
    filepath: types.PathLike,
    dims: Optional[str] = None,
    **kwargs,
):
    """
    Save an array to an image file.

    Parameters
    ----------
    data: types.ArrayLike
        The numpy or dask array to save to a file.
    filepath: types.FileLike
        The path to save the image and metadata to.
        The image writer is determined based off of the file extension included in
        this path.
    dims: str
        Optional string with the dimension order.
        If None, we will guess the order based off the selected image writer.
    kwargs: Any
        Extra keyword arguments that will be passed down to the writer subclass.
    """
    return AICSImage(data, known_dims=dims).save(
        filepath=filepath, save_dims=dims, **kwargs
    )

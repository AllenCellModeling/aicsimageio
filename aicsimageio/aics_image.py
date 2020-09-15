#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Optional, Tuple

import dask.array as da
import numpy as np

from . import types
from .dimensions import Dimensions
from .readers.reader import Reader
from .types import PhysicalPixelSizes

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
        dimensions ("ZYX", "TCZYX", "CYX") and reads them as consistent 5D "TCZYX"
        ("Time-Channel-Z-Y-X") ordered array(s). The data and metadata are lazy
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
        Initialize an image then read the file and return specified slices as a numpy
        array.

        >>> img = AICSImage("my_file.tiff")
        ... zstack_t8 = img.get_image_data("ZYX", T=8, C=0)

        Initialize an image, construct a delayed dask array for certain slices, then
        read only that data.

        >>> img = AICSImage("my_file.czi")
        ... zstack_t8 = img.get_image_dask_data("ZYX", T=8, C=0)
        ... zstack_t8_data = zstack_t8.compute()

        Initialize an image with a dask or numpy array.

        >>> data = np.random.rand(100, 100)
        ... img = AICSImage(data)

        Initialize an image from S3 with s3fs.

        >>> img = AICSImage("s3://my_bucket/my_file.tiff")

        Initialize an image and pass arguments to the reader using kwargs.

        >>> img = AICSImage("my_file.czi", chunk_by_dims=["T", "Y", "X"])

        Initialize an image, change scene, read data to numpy.

        >>> img = AICSImage("my_many_scene.czi")
        ... img.set_scene(3)
        ... img.data

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
        Returns
        -------
        reader: Reader
            Cheaply check to see if a given file is a recognized type and return the
            appropriate reader for the file.

        Raises
        ------
        TypeError: Unsupported file format.
        """
        pass

    @property
    def reader(self) -> Reader:
        """
        Returns
        -------
        reader: Reader
            The object created to read the image file type.
            The intent is that if the AICSImage class doesn't provide a raw enough
            interface then the base class can be used directly.
        """
        pass

    @property
    def scenes(self) -> List[int]:
        """
        Returns
        -------
        scenes: List[int]
            A list of valid scene ids in the file.

        Notes
        -----
        Scene IDs are not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """
        pass

    @property
    def current_scene(self) -> int:
        """
        Returns
        -------
        scene: int
            The current operating scene.
        """
        pass

    def set_scene(self, id: int):
        """
        Set the operating scene.

        Parameters
        ----------
        id: int
            The scene id to set as the operating scene.
        """
        pass

    @property
    def dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The image as a dask array with dimension ordering "TCZYX".
        """
        pass

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The image as a numpy array with dimension ordering "TCZYX".
        """
        pass

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.ndtype
            Data-type of the image array's elements.
        """
        pass

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns
        -------
        shape: Tuple[int]
            Tuple of the image array's dimensions.
        """
        pass

    @property
    def dims(self) -> Dimensions:
        """
        Returns
        -------
        dims: Dimensions
            Object with the paired dimension names and their sizes.
        """
        pass

    def get_image_dask_data(
        self, dimension_order_out: Optional[str] = None, **kwargs
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: "TCZYX"

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

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
        self, dimension_order_out: Optional[str] = None, **kwargs
    ) -> da.Array:
        """
        Read the image as a numpy array then return specific dimension image data.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: "TCZYX"

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

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
        * This will preload the entire image before returning the requested data.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        pass

    @property
    def metadata(self) -> Any:
        """
        Returns
        -------
        metadata: Any
            Passthrough to the base image reader metadata property.
            For more information, see the specific image format reader you are using
            for details on its metadata property.
        """
        pass

    @property
    def channel_names(self) -> List[str]:
        """
        Returns
        -------
        channel_names: List[str]
            Using available metadata, the list of strings representing channel names.
        """
        pass

    @property
    def physical_pixel_size(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.
        """
        pass

    def __str__(self) -> str:
        return f"<AICSImage [{type(self.reader).__name__} -- {self.shape}]>"

    def __repr__(self) -> str:
        return str(self)


def imread_dask(
    image: types.ImageLike,
    scene: Optional[int] = None,
    **kwargs,
) -> da.Array:
    """
    Read image as a dask array.

    Parameters
    ----------
    image: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    scene: Optional[int]
        A scene id to create a dask array for.
        Default: first
    kwargs: Any
        Extra keyword arguments that will be passed down to the reader subclass.

    Returns
    -------
    data: da.core.Array
        The image read and configured as a dask array.
    """
    img = AICSImage(image, **kwargs)

    # Select scene
    if scene is not None:
        img.set_scene(scene)

    return img.dask_data


def imread(
    image: types.ImageLike,
    scene: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Read image as a numpy array.

    Parameters
    ----------
    image: types.ImageLike
        A filepath, in memory numpy array, or preconfigured dask array.
    scene: Optional[int]
        A scene id to read the image data for.
        Default: first
    kwargs: Any
        Extra keyword arguments that will be passed down to the reader subclass.

    Returns
    -------
    data: np.ndarray
        The image read and configured as a numpy ndarray.
    """
    img = AICSImage(image, **kwargs)

    # Select scene
    if scene is not None:
        img.set_scene(scene)

    return img.data


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

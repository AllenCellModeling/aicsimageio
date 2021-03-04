#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr

from . import dimensions, exceptions, readers, transforms, types
from .metadata import utils as metadata_utils
from .readers.reader import Reader
from .types import PhysicalPixelSizes, ReaderType

###############################################################################


class AICSImage:

    # The order of the readers in this list is important.
    # Example:
    # if TiffReader was placed before OmeTiffReader,
    # we would never hit the OmeTiffReader
    SUPPORTED_READERS = (
        readers.ArrayLikeReader,
        # readers.CziReader,
        readers.LifReader,
        readers.OmeTiffReader,
        readers.TiffReader,
        readers.DefaultReader,
    )

    @staticmethod
    def determine_reader(image: types.ImageLike, **kwargs: Any) -> ReaderType:
        """
        Cheaply check to see if a given file is a recognized type and return the
        appropriate reader for the image.

        Returns
        -------
        ReaderClass: ReaderType
            The reader that supports the provided image.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            No reader could be found that supports the provided image.
        """
        for ReaderClass in AICSImage.SUPPORTED_READERS:
            if ReaderClass.is_supported_image(image, **kwargs):  # type: ignore
                return ReaderClass

        # Construct non-URI image "paths"
        # numpy, dask, etc.
        if isinstance(image, (str, Path)):
            path = str(image)
        else:
            path = str(type(image))

        raise exceptions.UnsupportedFileFormatError("AICSImage", path)

    def __init__(self, image: types.ImageLike, **kwargs: Any):
        """
        AICSImage takes microscopy image data types (files or arrays) of varying
        dimensions ("ZYX", "TCZYX", "CYX") and reads them as consistent 5D "TCZYX"
        ("Time-Channel-Z-Y-X") ordered array(s). The data and metadata are lazy
        loaded and can be accessed as needed.

        Parameters
        ----------
        image: types.ImageLike
            A string, Path, fsspec supported URI, or arraylike to read.
        kwargs: Any
            Extra keyword arguments that will be passed down to the reader subclass.

        Examples
        --------
        Initialize an image then read the file and return specified slices as a numpy
        array.

        >>> img = AICSImage("my_file.tiff")
        ... zstack_t8 = img.get_image_data("ZYX", T=8, C=0)

        Initialize an image, construct a delayed dask array for certain slices, then
        read only the specified chunk of data.

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
        ... img.set_scene("Image:3")
        ... img.data

        Notes
        -----
        If your image is a mosaic tiled image, the tiles will be stitched together.
        """
        # Determine reader class and create dask delayed array
        ReaderClass = self.determine_reader(image, **kwargs)
        self._reader = ReaderClass(image, **kwargs)

        # Lazy load data from reader and reformat to standard dimensions
        self._xarray_dask_data: Optional[xr.DataArray] = None
        self._xarray_data: Optional[xr.DataArray] = None
        self._dims: Optional[dimensions.Dimensions] = None

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
        return self._reader

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """
        return self.reader.scenes

    @property
    def current_scene(self) -> str:
        """
        Returns
        -------
        scene: str
            The current operating scene.
        """
        return self.reader.current_scene

    @property
    def current_scene_index(self) -> int:
        """
        Returns
        -------
        scene_index: int
            The current operating scene index in the file.
        """
        return self.scenes.index(self.current_scene)

    def set_scene(self, scene_id: str) -> None:
        """
        Set the operating scene.

        Parameters
        ----------
        scene_id: str
            The scene id to set as the operating scene.

        Raises
        ------
        IndexError: the provided scene id is not found in the available scene id list
        """
        # Only need to run when the scene id is different from current scene
        if scene_id != self.reader.current_scene:

            # Validate scene id
            if scene_id not in self.scenes:
                raise IndexError(
                    f"Scene id: {scene_id} "
                    f"is not present in available image scenes: {self.scenes}"
                )

            # Update current scene on the base Reader
            # This clears the base Reader's cache
            self.reader.set_scene(scene_id)

            # Reset the data stored in the AICSImage object
            self._xarray_dask_data = None
            self._xarray_data = None
            self._dims = None

    def _transform_data_array_to_aics_image_standard(
        self,
        arr: xr.DataArray,
    ) -> xr.DataArray:
        # Determine if we include Samples dim or not
        if dimensions.DimensionNames.Samples in arr.dims:
            return_dims = dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
        else:
            return_dims = dimensions.DEFAULT_DIMENSION_ORDER

        # Pull the data with the appropriate dimensions
        data = transforms.reshape_data(
            data=arr.data,
            given_dims="".join(arr.dims),  # type: ignore
            return_dims=return_dims,
        )

        # Pull coordinate planes
        coords: Dict[str, Any] = {}
        for d in return_dims:
            if d in arr.coords:
                coords[d] = arr.coords[d]

        # Add channel coordinate plane because it is required in AICSImage
        if dimensions.DimensionNames.Channel not in coords:
            coords[dimensions.DimensionNames.Channel] = [
                metadata_utils.generate_ome_channel_id(
                    image_id=self.current_scene,
                    channel_id=0,
                )
            ]

        return xr.DataArray(
            data,
            dims=tuple([d for d in return_dims]),
            coords=coords,  # type: ignore
            attrs=arr.attrs,
        )

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed image and metadata as an annotated data array.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        """
        if self._xarray_dask_data is None:
            if dimensions.DimensionNames.MosaicTile in self.reader.dims.order:
                self._xarray_dask_data = (
                    self._transform_data_array_to_aics_image_standard(
                        self.reader.mosaic_xarray_dask_data
                    )
                )

            else:
                self._xarray_dask_data = (
                    self._transform_data_array_to_aics_image_standard(
                        self.reader.xarray_dask_data
                    )
                )

        return self._xarray_dask_data

    @property
    def xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_data: xr.DataArray
            The fully read image and metadata as an annotated data array.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        Recommended to use `xarray_dask_data` for large mosaic images.
        """
        if self._xarray_data is None:
            if dimensions.DimensionNames.MosaicTile in self.reader.dims.order:
                self._xarray_data = self._transform_data_array_to_aics_image_standard(
                    self.reader.mosaic_xarray_data
                )

            else:
                self._xarray_data = self._transform_data_array_to_aics_image_standard(
                    self.reader.xarray_data
                )

            # Remake the delayed xarray dataarray object using a rechunked dask array
            # from the just retrieved in-memory xarray dataarray
            self._xarray_dask_data = xr.DataArray(
                da.from_array(self._xarray_data.data),
                dims=self._xarray_data.dims,
                coords=self._xarray_data.coords,
                attrs=self._xarray_data.attrs,
            )

        return self._xarray_data

    @property
    def dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The image as a dask array with standard dimension ordering.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        """
        return self.xarray_dask_data.data

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The image as a numpy array with standard dimension ordering.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        Recommended to use `dask_data` for large mosaic images.
        """
        return self.xarray_data.data

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.dtype
            Data-type of the image array's elements.
        """
        return self.xarray_dask_data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.
        """
        return self.xarray_dask_data.shape

    @property
    def dims(self) -> dimensions.Dimensions:
        """
        Returns
        -------
        dims: dimensions.Dimensions
            Object with the paired dimension names and their sizes.
        """
        if self._dims is None:
            self._dims = dimensions.Dimensions(
                dims=self.xarray_dask_data.dims, shape=self.shape
            )

        return self._dims

    def get_image_dask_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: dimensions.DEFAULT_DIMENSION_ORDER (with or without Samples)

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
        data: da.Array
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
        If a requested dimension is not present in the data the dimension is
        added with a depth of 1.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None:
            return self.dask_data

        # Transform and return
        return transforms.reshape_data(
            data=self.dask_data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

    def get_image_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Read the image as a numpy array then return specific dimension image data.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: dimensions.DEFAULT_DIMENSION_ORDER (with or without Samples)

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
        data: np.ndarray
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
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None:
            return self.data

        # Transform and return
        return transforms.reshape_data(
            data=self.data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

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
        return self.reader.metadata

    @property
    def channel_names(self) -> List[str]:
        """
        Returns
        -------
        channel_names: List[str]
            Using available metadata, the list of strings representing channel names.
        """
        # Unlike the base readers, the AICSImage guarantees a Channel dim
        return list(self.xarray_dask_data[dimensions.DimensionNames.Channel].values)

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return self.reader.physical_pixel_sizes

    def __str__(self) -> str:
        return (
            f"<AICSImage ["
            f"Reader: {type(self.reader).__name__}, "
            f"Image-is-in-Memory: {self._xarray_data is not None}"
            f"]>"
        )

    def __repr__(self) -> str:
        return str(self)


def _construct_img(
    image: types.ImageLike, scene_id: Optional[str] = None, **kwargs: Any
) -> AICSImage:
    # Construct image
    img = AICSImage(image, **kwargs)

    # Select scene
    if scene_id is not None:
        img.set_scene(scene_id)

    return img


def imread_xarray_dask(
    image: types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Read image as a delayed xarray DataArray.

    Parameters
    ----------
    image: types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the DataArray with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the AICSImage and Reader subclass.

    Returns
    -------
    data: xr.DataArray
        The image read, scene selected, and returned as an AICS standard shaped delayed
        xarray DataArray.
    """
    return _construct_img(image, scene_id, **kwargs).xarray_dask_data


def imread_dask(
    image: types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> da.Array:
    """
    Read image as a delayed dask array.

    Parameters
    ----------
    image: types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the dask array with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the AICSImage and Reader subclass.

    Returns
    -------
    data: da.core.Array
        The image read, scene selected, and returned as an AICS standard shaped delayed
        dask array.
    """

    return _construct_img(image, scene_id, **kwargs).dask_data


def imread_xarray(
    image: types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Read image as an in-memory xarray DataArray.

    Parameters
    ----------
    image: types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the DataArray with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the AICSImage and Reader subclass.

    Returns
    -------
    data: xr.DataArray
        The image read, scene selected, and returned as an AICS standard shaped
        in-memory DataArray.
    """
    return _construct_img(image, scene_id, **kwargs).xarray_data


def imread(
    image: types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Read image as a numpy array.

    Parameters
    ----------
    image: types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the numpy array with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the AICSImage and Reader subclass.

    Returns
    -------
    data: np.ndarray
        The image read, scene selected, and returned as an AICS standard shaped
        np.ndarray.
    """
    return _construct_img(image, scene_id, **kwargs).data

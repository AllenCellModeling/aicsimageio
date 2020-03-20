#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

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
    def __init__(
        self,
        data: types.ImageLike,
        known_dims: Optional[str] = None,
        dask_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        """
        AICSImage takes microscopy image data types (files) of varying dimensions ("ZYX", "TCZYX", "CYX") and
        puts them into a consistent 6D "STCZYX" ordered dask array. The data, metadata are lazy loaded and can be
        accessed as needed. Note the dims are assumed to match "STCZYX" from right to left meaning if 4 dimensional
        data is provided then the dimensions are assigned to be "CZYX", 2 dimensional would be "YX". This guessed
        assignment is only for file types without dimension metadata (i.e. not .ome.tiff or .czi).

        Parameters
        ----------
        data: types.ImageLike
            String with path to file, numpy.ndarray, or dask.array.Array with up to six dimensions.
        known_dims: Optional[str]
            Optional string with the known dimension order. If None, the reader will attempt to parse dim order.
        dask_kwargs: Dict[str, Any] = {}
            A dictionary of arguments to pass to a dask cluster and or client.
        kwargs: Dict[str, Any]
            Extra keyword arguments that can be passed down to either the reader subclass or, if using the context
            manager, the LocalCluster initialization.

        Examples
        --------
        Initialize an image and read the slices specified as a numpy array.

        >>> img = AICSImage("my_file.tiff")
        ... zstack_t8 = img.get_image_data("ZYX", S=0, T=8, C=0)

        Initialize an image, construct a delayed dask array for certain slices, then read the data.

        >>> img = AICSImage("my_file.czi")
        ... zstack_t8 = img.get_image_dask_data("ZYX", S=0, T=8, C=0)
        ... zstack_t8_data = zstack_t8.compute()

        Initialize an image with a dask or numpy array.

        >>> data = np.random.rand(100, 100)
        ... img = AICSImage(data)

        Initialize an image and pass arguments to the reader using kwargs.

        >>> img = AICSImage("my_file.czi", chunk_by_dims=["T", "Y", "X"])

        Create a local dask cluster for the duration of the context manager.

        >>> with AICSImage("filename.ome.tiff") as img:
        ...     data = img.get_image_data("ZYX", S=0, T=0, C=0)

        Create a local dask cluster with arguments for the duration of the context manager.

        >>> with AICSImage("filename.ome.tiff", dask_kwargs={"nworkers": 4}) as img:
        ...     data = img.get_image_data("ZYX", S=0, T=0, C=0)

        Connect to a specific dask cluster for the duration of the context manager.

        >>> with AICSImage("filename.ome.tiff", dask_kwargs={"address": "tcp://localhost:1234"}) as img:
        ...     data = img.get_image_data("ZYX", S=0, T=0, C=0)
        ```

        Notes
        -----
        When using the AICSImage context manager, the processing machine or container must have networking capabilities
        enabled to function properly.

        Constructor for AICSImage class intended for providing a unified interface for dealing with
        microscopy images. To extend support to a new reader simply add a new reader child class of
        Reader ([readers/reader.py]) and add the class to SUPPORTED_READERS variable.
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

        # Store dask client and cluster setup
        self._dask_kwargs = dask_kwargs
        self._client = None
        self._cluster = None

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
        Returns a dask array with dimension ordering "STCZYX".
        """
        # Construct dask array if never before constructed
        if self._dask_data is None:
            reader_data = self.reader.dask_data

            # Read and reshape and handle delayed known dims reshape
            self._dask_data = transforms.reshape_data(
                data=reader_data,
                given_dims=self._known_dims or self.reader.dims,
                return_dims=self.dims,
            )

        return self._dask_data

    @property
    def data(self) -> np.ndarray:
        """
        Return the entire image as a numpy array with dimension ordering "STCZYX".
        """
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
        size: Tuple[int]
            A tuple with the requested dimensions filled in.
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
    def shape(self) -> Tuple[int]:
        """
        Returns
        -------
        shape: Tuple[int]
            A tuple with the size of all dimensions.
        """
        return self.size()

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
        # The reader can implement read optimization or not.
        return self.reader.metadata

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

    def view_napari(self, rgb: bool = False, **kwargs):
        """
        If installed, load the image in a napari viewer.

        Parameters
        ----------
        rgb: bool
            Is the image RGB / RGBA
            Default: False (is not RGB)
        **kwargs
            Extra arguments passed down to the viewer
        """
        try:
            import napari

            # Construct getitem operations tuple to select down the data in the filled dimensions
            ops = []
            selected_dims = []
            for dim in self.dims:
                if self.size(dim)[0] == 1:
                    ops.append(0)
                else:
                    ops.append(slice(None, None, None))
                    selected_dims.append(dim)

            # Actually select the down
            data = self.dask_data[tuple(ops)]

            # Convert selected_dims to string
            dims = "".join(selected_dims)

            # Create name for window
            if isinstance(self.reader, ArrayLikeReader):
                title = f"napari: {self.dask_data.shape}"
            else:
                title = f"napari: {self.reader._file.name}"

            # Handle RGB entirely differently
            if rgb:
                # Swap channel to last dimension
                new_dims = f"{dims.replace(Dimensions.Channel, '')}{Dimensions.Channel}"
                data = transforms.transpose_to_dims(
                    data=data,
                    given_dims=dims,
                    return_dims=new_dims
                )

                # Run napari
                with napari.gui_qt():
                    napari.view_image(
                        data,
                        is_pyramid=False,
                        ndisplay=3 if Dimensions.SpatialZ in dims else 2,
                        title=title,
                        axis_labels=dims.replace(Dimensions.Channel, ""),
                        rgb=rgb,
                        **kwargs
                    )

            # Handle all other images besides RGB not requested
            else:
                # Channel axis
                c_axis = dims.index(Dimensions.Channel) if Dimensions.Channel in dims else None

                # Set visible based on number of channels
                if c_axis is not None:
                    if data.shape[c_axis] > 3:
                        visible = False
                    else:
                        visible = True
                else:
                    visible = True

                # Drop channel from dims string
                dims = dims.replace(Dimensions.Channel, "") if Dimensions.Channel in dims else dims

                # Run napari
                with napari.gui_qt():
                    napari.view_image(
                        data,
                        is_pyramid=False,
                        ndisplay=3 if Dimensions.SpatialZ in dims else 2,
                        channel_axis=c_axis,
                        axis_labels=dims,
                        title=title,
                        visible=visible,
                        **kwargs
                    )

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"'napari' has not been installed. To use this function install napari with either: "
                f"'pip install napari' or 'pip install aicsimageio[interactive]'"
            )

    def get_channel_names(self, scene: int = 0) -> List[str]:
        """
        Attempts to use the image's metadata to get the image's channel names.

        Parameters
        ----------
        scene: int
            The index of the scene for which to return channel names.

        Returns
        -------
        channels_names: List[str]
            List of strings representing the channel names.
        """
        # Get channel names from reader
        channel_names = self.reader.get_channel_names(scene)

        # Unlike the readers, AICSImage objects always have a channel dimension
        # In the case the base reader returns None, return a list of "0"
        if channel_names is None:
            return [str(i) for i in range(self.size_c)]

        # Return the read channel names
        return channel_names

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        """
        Attempts to retrieve physical pixel size for the specified scene.
        If none available, returns `1.0` for each spatial dimension.

        Parameters
        ----------
        scene: int
            The index of the scene for which to return physical pixel sizes.

        Returns
        -------
        sizes: Tuple[float]
            Tuple of floats representing the pixel sizes for X, Y, Z, in that order.
        """
        return self.reader.get_physical_pixel_size(scene)

    def __repr__(self) -> str:
        return f"<AICSImage [{type(self.reader).__name__}]>"

    @property
    def cluster(self) -> Optional["distributed.LocalCluster"]:
        """
        If this object created a local Dask cluster, return it.
        """
        return self._cluster

    @property
    def client(self) -> Optional["distributed.Client"]:
        """
        If connected to a Dask cluster, return the connected Client.
        """
        return self._client

    def close(self):
        """
        Close the connection to the Dask distributed Client.
        If this object created a LocalCluster, close it down as well.
        """
        from . import dask_utils
        self._cluster, self._client = dask_utils.shutdown_cluster_and_client(self.cluster, self.client)

    def __enter__(self):
        """
        If provided an address, create a Dask Client connection.
        If not provided an address, create a LocalCluster and Client connection.
        If not provided an address, other Dask kwargs are accepted and passed down to the LocalCluster object.
        """
        # Warn of future changes to API
        warnings.warn(
            "In aicsimageio>=3.2.*, the AICSImage context manager will no longer construct and manage a distributed "
            "local dask cluster for you. If this functionality is desired for your work, please switch to explictly "
            "calling the `aicsimageio.dask_utils.cluster_and_client` context manager.",
            FutureWarning
        )

        from . import dask_utils
        self._cluster, self._client = dask_utils.spawn_cluster_and_client(**self._dask_kwargs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Always close the Dask Client connection.
        If connected to *strictly* a LocalCluster, close it down as well.
        """
        self.close()


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

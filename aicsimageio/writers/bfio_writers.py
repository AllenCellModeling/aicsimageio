from itertools import product
from typing import Any, List, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from bfio import BioWriter
from fsspec.implementations.local import LocalFileSystem
from ome_types import OME, from_xml

from .. import types
from ..utils import io_utils
from .writer import Writer


class OmeTiledTiffWriter(Writer):
    @staticmethod
    def save(  # type: ignore
        data: types.ArrayLike,
        uri: types.PathLike,
        ome_xml: Optional[Union[str, OME]] = None,
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]] = None,
        physical_pixel_sizes: Optional[
            Union[types.PhysicalPixelSizes, List[types.PhysicalPixelSizes]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write a data array to a file using tile chunking. Tiles are 1024x1024 pixels.

        Parameters
        ----------
        data: Union[List[types.ArrayLike], types.ArrayLike]
            The array of data to store. Data arrays must have 2 to 5 dimensions. Data
            should have shape [T,C,Z,Y,X], where interstitial empty dimensions must be
            present. For example, a 3-channel image that is 1024x1024 must at least have
            shape [3,1,1024,1024]. Image with shape [3,1024,1024] will be interpreted as
            an image with 3 z-slices. Sample dimension is not supported.
            If the data is a Dask array, chunked writing will be performed
        uri: types.PathLike
            The URI or local path for where to save the data.
            Note: OmeTiledTiffWriter can only write to local file systems.
        ome_xml: Optional[Union[str, OME]]
            Provided OME metadata. The metadata can be an xml string or an OME object
            from ome-types. A provided ome_xml will override any other provided
            metadata arguments.
            Default: None
            The passed-in metadata will be validated against current OME_XML schema and
            raise exception if invalid.
            The ome_xml will also be compared against the dimensions of the input data.
            If None is given, then OME-XML metadata will be generated from the data
            array and any of the following metadata arguments.
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]]
            Lists of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:channel_index"
        physical_pixel_sizes: Optional[Union[types.PhysicalPixelSizes,
                List[types.PhysicalPixelSizes]]]
            List of numbers representing the physical pixel sizes in Z, Y, X in microns
            Default: None

        Note
        ----
        This writer can perform chunked writing. When the input array is a dask array or
        an xarray DataArray backed by dask, data is written one XY plane at a time to
        conserve memory. This is especially useful when the image will not fit into
        memory. The underlying write can support further chunking in XY dimensions, so
        this functionality should be added in the future.

        Raises
        ------
        ValueError:
            Non-local file system URI provided.

        Examples
        --------
        Write a TCZYX data set to a tiled OME-Tiff

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... OmeTiledTiffWriter.save(image, "file.ome.tif")

        If an image is too large to fit into memory, a dask array can be passed in and
        chunked writing will take place. This will work if each YX plane can fit into
        memory.

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... OmeTiledTiffWriter.save(image, "file.ome.tif")
        """
        if data.ndim > 5 or data.ndim < 2:
            raise ValueError(
                "Data must have 2-5 dimensions and be in TCZYX order. "
                + "The S (sample) dimension is not allowed."
            )

        # Resolve final destination
        fs, path = io_utils.pathlike_to_fs(uri)

        # Catch non-local file system
        if not isinstance(fs, LocalFileSystem):
            raise ValueError(
                f"Cannot write to non-local file system. "
                f"Received URI: {uri}, which points to {type(fs)}."
            )

        # If metadata is attached as lists, enforce matching shape
        if not isinstance(data, (np.ndarray, da.core.Array)):
            raise TypeError(
                "Input data for the OmeTiledTiffWriter must be a numpy.ndarray"
            )

        if isinstance(ome_xml, str):
            ome_xml = from_xml(ome_xml)

        with BioWriter(path, metadata=ome_xml, backend="python") as bw:

            # Define xml if ome_xml is not present but other kwargs are
            if ome_xml is None:
                if channel_names is not None:
                    bw.channel_names = channel_names

                if physical_pixel_sizes is not None:
                    bw.ps_z = (physical_pixel_sizes[0], None)
                    bw.ps_y = (physical_pixel_sizes[1], None)
                    bw.ps_x = (physical_pixel_sizes[2], None)

                for dim, val in zip("XYZCT", reversed(data.shape)):
                    setattr(bw, dim, val)

                bw.dtype = data.dtype

            dim_order = (-2, -1) + tuple(reversed(range(len(data.shape) - 2)))

            data = data.transpose(dim_order)

            if not all(w == d for w, d in zip(bw.shape, data.shape)):
                raise ValueError("Metadata dimensions do not match input data.")

            if bw.dtype != data.dtype:
                raise ValueError("Writer data type does not match input data.")

            # If the data is not a dask array, just write the image all at once
            if isinstance(data, xr.core.dataarray.DataArray):
                data = data.data

            if not isinstance(data, da.Array):
                bw[:] = data

            # If the data is a dask array, perform chunked writing for more scalability
            else:
                for plane in product(*(range(d) for d in reversed(data.shape[2:]))):
                    index = (slice(None), slice(None)) + plane[::-1]
                    bw[index] = data[index].compute()

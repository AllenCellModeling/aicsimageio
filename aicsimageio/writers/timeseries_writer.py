#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import dask.array as da
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from imageio import get_writer

from .. import types
from ..dimensions import DimensionNames
from ..exceptions import InvalidDimensionOrderingError, UnexpectedShapeError
from ..transforms import reshape_data
from ..utils import io_utils
from .writer import Writer

try:
    from ..readers.default_reader import DefaultReader

except ImportError:
    raise ImportError(
        "Base imageio is required for this writer. "
        "Install with `pip install aicsimageio[base-imageio]`"
    )


###############################################################################


class TimeseriesWriter(Writer):
    """
    A writer for timeseries Greyscale, RGB, or RGBA image data.
    Primarily directed at formats: "gif", "mp4", "mkv", etc.

    Notes
    -----
    To use this writer, install with: `pip install aicsimageio[base-imageio]`.
    """

    _TIMEPOINT_DIMENSIONS = [
        DimensionNames.Time,
        DimensionNames.SpatialY,
        DimensionNames.SpatialX,
    ]
    _TIMEPOINT_WITH_SAMPLES_DIMENSIONS = _TIMEPOINT_DIMENSIONS + [
        DimensionNames.Samples
    ]

    DIM_ORDERS = {
        3: "".join(_TIMEPOINT_DIMENSIONS),  # Greyscale
        4: "".join(_TIMEPOINT_WITH_SAMPLES_DIMENSIONS),  # RGB / RGBA
    }

    @staticmethod
    def _write_chunks(
        f: str,
        extension: str,
        imageio_mode: str,
        fps: int,
        data: da.Array,
        dim_order: str,
    ) -> None:
        with get_writer(
            f,
            format=extension,
            mode=imageio_mode,
            fps=fps,
        ) as writer:
            # Make each chunk of the dask array be a frame
            chunks = tuple(1 if dim == DimensionNames.Time else -1 for dim in dim_order)
            data = data.rechunk(chunks)

            # Save each frame
            for block in data.blocks:
                # Need to squeeze to remove the singleton T dimension
                writer.append_data(da.squeeze(block).compute())

    @staticmethod
    def save(
        data: types.ArrayLike,
        uri: types.PathLike,
        dim_order: str = None,
        fps: int = 24,
        **kwargs: Any,
    ) -> None:
        """
        Write a data array to a file.

        Parameters
        ----------
        data: types.ArrayLike
            The array of data to store. Data must have either three or four dimensions.
        uri: types.PathLike
            The URI or local path for where to save the data.
        dim_order: str
            The dimension order of the provided data.
            Default: None. Based off the number of dimensions, will assume
            the dimensions -- three dimensions: TYX and four dimensions: TYXS.
        fps: int
            Frames per second to attach as metadata.
            Default: 24

        Examples
        --------
        Data is the correct shape and dimension order

        >>> image = dask.array.random.random((50, 100, 100))
        ... TimeseriesWriter.save(image, "file.gif")

        Data provided with current dimension order

        >>> image = numpy.random.rand(100, 3, 1024, 2048)
        ... TimeseriesWriter.save(image, "file.mkv", "TSYX")

        Save to remote

        >>> image = numpy.random.rand(300, 100, 100, 3)
        ... TimeseriesWriter.save(image, "s3://my-bucket/file.png")

        Raises
        ------
        IOError
            Cannot write FFMPEG formats to remote storage.

        Notes
        -----
        This writer can also be useful when wanting to create a timeseries image using
        a non-time dimension. For example, creating a timeseries image where each frame
        is a Z-plane from a source volumetric image as seen below.

        >>> image = AICSImageIO("some_z_stack.ome.tiff")
        ... TimeseriesWriter.save(
        ...     data=image.get_image_data("ZYX", T=0, C=0),
        ...     uri="some_z_stack.mp4",
        ...     # Overloading the Z dimension as the Time dimension
        ...     # Technically not needed as it would have been assumed due to three dim
        ...     dim_order="TYX",
        ... )

        """
        # Check unpack uri and extension
        fs, path = io_utils.pathlike_to_fs(uri)
        (
            extension,
            imageio_mode,
        ) = DefaultReader._get_extension_and_mode(path)

        # Convert to dask array to make downstream usage of data have a consistent API
        if isinstance(data, np.ndarray):
            data = da.from_array(data)

        # Shorthand num dimensions
        n_dims = len(data.shape)

        # Check num dimensions
        if n_dims not in TimeseriesWriter.DIM_ORDERS:
            raise UnexpectedShapeError(
                f"TimeseriesWriter requires that data must have either 3 or 4 "
                f"dimensions. Provided data with {n_dims} dimensions. ({data.shape})"
            )

        # Assume dim order if not provided
        if dim_order is None:
            dim_order = TimeseriesWriter.DIM_ORDERS[n_dims]

        # Uppercase dim order
        dim_order = dim_order.upper()

        # Check dimensions provided in the dim order string are all T, Y, X, or S
        if any(
            [
                dim not in TimeseriesWriter._TIMEPOINT_WITH_SAMPLES_DIMENSIONS
                for dim in dim_order
            ]
        ):
            raise InvalidDimensionOrderingError(
                f"The dim_order parameter only accepts dimensions: "
                f"{TimeseriesWriter._TIMEPOINT_WITH_SAMPLES_DIMENSIONS}. "
                f"Provided dim_order string: '{dim_order}'."
            )

        # Transpose dimensions if dim_order not ready for imageio
        if dim_order != TimeseriesWriter.DIM_ORDERS[n_dims]:
            # Actual reshape of the data
            data = reshape_data(
                data,
                given_dims=dim_order,
                return_dims=TimeseriesWriter.DIM_ORDERS[n_dims],
            )

            # Set dim order to updated order
            dim_order = TimeseriesWriter.DIM_ORDERS[n_dims]

        # Handle FFMPEG formats
        if extension in DefaultReader.FFMPEG_FORMATS:
            # FFMPEG can only handle local files
            # https://github.com/imageio/imageio-ffmpeg/issues/28#issuecomment-566012783
            if not isinstance(fs, LocalFileSystem):
                raise IOError(
                    f"Can only write to local files for formats: "
                    f"{DefaultReader.FFMPEG_FORMATS}."
                )

            # Else, write with local
            TimeseriesWriter._write_chunks(
                f=path,
                extension=extension,
                imageio_mode=imageio_mode,
                fps=fps,
                data=data,
                dim_order=dim_order,
            )

        # Handle all non-ffmpeg formats
        else:
            with fs.open(path, "wb") as open_resource:
                TimeseriesWriter._write_chunks(
                    f=open_resource,
                    extension=extension,
                    imageio_mode=imageio_mode,
                    fps=fps,
                    data=data,
                    dim_order=dim_order,
                )

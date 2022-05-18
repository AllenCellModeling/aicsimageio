#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import dask.array as da
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


class TwoDWriter(Writer):
    """
    A writer for image data is only 2 dimension with samples (RGB / RGBA) optional.
    Primarily directed at formats: "png", "jpg", etc.

    This is primarily a passthrough to imageio.imwrite.

    Notes
    -----
    To use this writer, install with: `pip install aicsimageio[base-imageio]`.
    """

    _PLANE_DIMENSIONS = [
        DimensionNames.SpatialY,
        DimensionNames.SpatialX,
    ]
    _PLANE_WITH_SAMPLES_DIMENSIONS = _PLANE_DIMENSIONS + [DimensionNames.Samples]

    DIM_ORDERS = {
        2: "".join(_PLANE_DIMENSIONS),  # Greyscale
        3: "".join(_PLANE_WITH_SAMPLES_DIMENSIONS),  # RGB / RGBA
    }

    @staticmethod
    def save(
        data: types.ArrayLike,
        uri: types.PathLike,
        dim_order: str = None,
        **kwargs: Any,
    ) -> None:
        """
        Write a data array to a file.

        Parameters
        ----------
        data: types.ArrayLike
            The array of data to store. Data must have either two or three dimensions.
        uri: types.PathLike
            The URI or local path for where to save the data.
        dim_order: str
            The dimension order of the provided data.
            Default: None. Based off the number of dimensions, will assume
            the dimensions similar to how
            aicsimageio.readers.default_reader.DefaultReader reads in
            data. That is, two dimensions: YX and three dimensions: YXS.

        Examples
        --------
        Data is the correct shape and dimension order

        >>> image = dask.array.random.random((100, 100, 4))
        ... TwoDWriter.save(image, "file.png")

        Data provided with current dimension order

        >>> image = numpy.random.rand(3, 1024, 2048)
        ... TwoDWriter.save(image, "file.png", "SYX")

        Save to remote

        >>> image = numpy.random.rand(100, 100, 3)
        ... TwoDWriter.save(image, "s3://my-bucket/file.png")
        """
        # Check unpack uri and extension
        fs, path = io_utils.pathlike_to_fs(uri)
        (
            extension,
            imageio_mode,
        ) = DefaultReader._get_extension_and_mode(path)

        # Assumption: if provided a dask array to save, it can fit into memory
        if isinstance(data, da.core.Array):
            data = data.compute()

        # Shorthand num dimensions
        n_dims = len(data.shape)

        # Check num dimensions
        if n_dims not in TwoDWriter.DIM_ORDERS:
            raise UnexpectedShapeError(
                f"TwoDWriter requires that data must have either 2 or 3 dimensions. "
                f"Provided data with {n_dims} dimensions. ({data.shape})"
            )

        # Assume dim order if not provided
        if dim_order is None:
            dim_order = TwoDWriter.DIM_ORDERS[n_dims]

        # Uppercase dim order
        dim_order = dim_order.upper()

        # Check dimensions provided in the dim order string are all Y, X, or S
        if any(
            [dim not in TwoDWriter._PLANE_WITH_SAMPLES_DIMENSIONS for dim in dim_order]
        ):
            raise InvalidDimensionOrderingError(
                f"The dim_order parameter only accepts dimensions: "
                f"{TwoDWriter._PLANE_WITH_SAMPLES_DIMENSIONS}. "
                f"Provided dim_order string: '{dim_order}'."
            )

        # Transpose dimensions if dim_order not ready for imageio
        if dim_order != TwoDWriter.DIM_ORDERS[n_dims]:
            data = reshape_data(
                data, given_dims=dim_order, return_dims=TwoDWriter.DIM_ORDERS[n_dims]
            )

        # Save image
        with fs.open(path, "wb") as open_resource:
            with get_writer(
                open_resource,
                format=extension,
                mode=imageio_mode,
            ) as writer:
                writer.append_data(data)

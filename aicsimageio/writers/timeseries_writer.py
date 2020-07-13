#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
from imageio import get_writer

from ..exceptions import InconsistentShapeError, InvalidDimensionOrderingError
from ..transforms import reshape_data
from .writer import Writer

###############################################################################


class TimeseriesWriter(Writer):
    """
    A writer for timeseries Greyscale, RGB, or RGBA image data.
    Primarily directed at formats: "gif", "mp4", "mkv", etc.
    """

    DIM_ORDERS = {
        3: "TYX",  # Greyscale
        4: "TYXC",  # RGB
    }

    @staticmethod
    def save(
        data: Union[da.Array, np.ndarray],
        filepath: Union[str, Path],
        dim_order: str = None,
        fps: int = 24,
        **kwargs,
    ):
        """
        Write a data array to a file.

        Parameters
        ----------
        data: Union[da.Array, np.ndarray]
            The array of data to store. Data must have either three or four dimensions.
        filepath: Union[str, Path]
            The path to save the data at.
        dim_order: str
            The dimension order of the provided data.
            Default: None. Based off the number of dimensions, will assume
            the dimensions -- three dimensions: TYX and four dimensions: TYXC.
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
        ... TimeseriesWriter.save(image, "file.mkv", "TCYX")

        Notes
        -----
        This writer can also be useful when wanting to create a timeseries image using
        a non-time dimension. For example, creating a timeseries image where each frame
        is a Z-plane from a source volumetric image as seen below.

        >>> image = AICSImageIO("some_z_stack.ome.tiff")
        ... TimeseriesWriter.save(
        ...     data=image.get_image_data("ZYX", S=0, T=0, C=0),
        ...     filepath="some_z_stack.mp4",
        ...     # Overloading the Z dimension as the Time dimension
        ...     # Technically not needed as it would have been assumed due to three dim
        ...     dim_order="TYX",
        ... )

        """
        # Check filepath
        # Do not fully resolve to absolute because Mac is bad with mounted drives
        filepath = Path(filepath).expanduser()

        # Convert to dask array to make downstream usage of data have a consistent API
        if isinstance(data, np.ndarray):
            data = da.from_array(data)

        # Shorthand num dimensions
        n_dims = len(data.shape)

        # Check num dimensions
        if n_dims not in TimeseriesWriter.DIM_ORDERS:
            raise InconsistentShapeError(
                f"TimeseriesWriter requires that data must have either 3 or 4 "
                f"dimensions. Provided data with {n_dims} dimensions. ({data.shape})"
            )

        # Assume dim order if not provided
        if dim_order is None:
            dim_order = TimeseriesWriter.DIM_ORDERS[n_dims]

        # Uppercase dim order
        dim_order = dim_order.upper()

        # Check dimensions provided in the dim order string are all T, C, Y, or X
        if any([dim not in ["T", "C", "Y", "X"] for dim in dim_order]):
            raise InvalidDimensionOrderingError(
                f"The dim_order parameter only accepts dimensions: "
                f"'T', 'C', 'Y', or 'X' "
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

        # Get writer
        writer = get_writer(filepath, fps=fps)

        # Make each chunk of the dask array be a frame
        chunks = tuple(1 if dim == "T" else -1 for dim in dim_order)
        data = data.rechunk(chunks)

        # Save each frame
        for block in data.blocks:
            # Need to squeeze to remove the singleton T dimension
            writer.append_data(da.squeeze(block).compute())

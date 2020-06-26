#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
from imageio import imwrite

from ..constants import Dimensions
from ..exceptions import InconsistentShapeError, InvalidDimensionOrderingError
from ..readers import DefaultReader
from ..transforms import reshape_data
from .writer import Writer

###############################################################################


class RGBWriter(Writer):
    """
    A writer for image data where the channel dimension act's as color. (RGB / RGBA)
    Primarily directed at formats: "png", "jpg", etc.

    This is primarily a passthrough to imageio.imwrite.
    """

    @staticmethod
    def save(
        data: Union[da.Array, np.ndarray],
        filepath: Union[str, Path],
        dim_order: str = None,
        **kwargs
    ):
        """
        Write a data array to a file.

        Parameters
        ----------
        data: Union[da.Array, np.ndarray]
            The array of data to store. Data must have either two or three dimensions.
        filepath: Union[str, Path]
            The path to save the data at.
        dim_order: str
            The dimension order of the provided data.
            Default: None. Based off the number of dimensions, will assume
            the dimensions similar to how aicsimageio.readers.DefaultReader reads in
            data. That is, two dimensions: YX and three dimensions: YXC.

        Examples
        --------
        Data is the correct shape and dimension order

        >>> image = dask.array.random.random((100, 100, 4))
        ... RGBWriter.save(image, "file.png")

        Data provided with current dimension order

        >>> image = numpy.random.rand(3, 1024, 2048)
        ... RGBWriter.save(image, "file.png", "CYX")
        """
        # Check filepath
        # Do not fully resolve to absolute because Mac is bad with mounted drives
        filepath = Path(filepath).expanduser()

        # Assumption: if provided a dask array to save, it can fit into memory
        if isinstance(data, da.core.Array):
            data = data.compute()

        # Check num dimensions
        if len(data.shape) not in [2, 3]:
            raise InconsistentShapeError(
                f"RGBWriter requires that data must have either 2 or 3 dimensions. "
                f"Provided data with {len(data.shape)} dimensions. ({data.shape})"
            )

        # Assume dim order if not provided
        if dim_order is None:
            if len(data.shape) == 2:
                dim_order = "YX"
            if len(data.shape) == 3:
                dim_order = "YXC"

        # Uppercase dim order
        dim_order = dim_order.upper()

        # Check dimensions provided in the dim order string are all C, Y, or X
        if any([dim not in ["C", "Y", "X"] for dim in dim_order]):
            raise InvalidDimensionOrderingError(
                f"The dim_order parameter only accepts dimensions: 'C', 'Y', or 'X' "
                f"Provided dim_order string: '{dim_order}'."
            )

        # Transpose dimensions if dim_order not ready for imageio
        if len(data.shape) == 2 and dim_order != "YX":
            data = reshape_data(data, given_dims=dim_order, return_dims="YX")
        if len(data.shape) == 3 and dim_order != "YXC":
            data = reshape_data(data, given_dims=dim_order, return_dims="YXC")

        # Save image
        imwrite(filepath, data)

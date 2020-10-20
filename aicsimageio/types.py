#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BufferedIOBase
from pathlib import Path
from typing import NamedTuple, Union

import dask.array as da
import numpy as np
import xarray as xr

# IO Types
FSSpecBased = "fsspec-based"
PathLike = Union[str, Path]
BytesLike = Union[bytes, BufferedIOBase]
FileLike = Union[PathLike, BytesLike]
ArrayLike = Union[np.ndarray, da.Array, xr.DataArray]
ImageLike = Union[FileLike, ArrayLike]


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: float
    Y: float
    X: float

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BufferedIOBase
from pathlib import Path
from typing import NamedTuple, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileOpener
from fsspec.spec import AbstractBufferedFile

# IO Types
FSSpecBased = Union[AbstractBufferedFile, LocalFileOpener]
PathLike = Union[str, Path]
BytesLike = Union[bytes, BufferedIOBase]
FileLike = Union[PathLike, BytesLike]
ArrayLike = Union[np.ndarray, da.Array, xr.DataArray]
ImageLike = Union[FileLike, ArrayLike]


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: Optional[float]
    Y: Optional[float]
    X: Optional[float]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BufferedIOBase
from pathlib import Path
from typing import NamedTuple, Union

import dask.array as da
import numpy as np
from fsspec.spec import AbstractBufferedFile

# IO Types
PathLike = Union[str, Path, AbstractBufferedFile]
BytesLike = Union[bytes, BufferedIOBase]
FileLike = Union[PathLike, BytesLike]
ArrayLike = Union[np.ndarray, da.Array]
ImageLike = Union[FileLike, ArrayLike]


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: float
    Y: float
    X: float

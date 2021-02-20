#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta
from pathlib import Path
from typing import NamedTuple, Union

import dask.array as da
import numpy as np

###############################################################################

# IO Types
PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
ImageLike = Union[PathLike, ArrayLike]

# Reader Type
ReaderType = ABCMeta


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: float
    Y: float
    X: float

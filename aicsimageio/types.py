#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import NamedTuple, Optional, Union

import dask.array as da
import numpy as np

###############################################################################

# IO Types
PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
ImageLike = Union[PathLike, ArrayLike]


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: Optional[float]
    Y: Optional[float]
    X: Optional[float]

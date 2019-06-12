#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collection of types used across multiple objects and functions.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np

# Imaging Data Types
SixDArray = np.array  # In order STCZYX

# IO Types
PathLike = Union[str, Path]
BytesLike = Union[bytes, BytesIO]

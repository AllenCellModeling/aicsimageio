#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Top-level package for AICSImageIO."""

from .aics_image import AICSImage  # noqa: F401
from .aics_image import imread  # noqa: F401
from .aics_image import imread_dask  # noqa: F401
from .aics_image import imread_xarray  # noqa: F401
from .aics_image import imread_xarray_dask  # noqa: F401

__author__ = "Jackson Maxfield Brown, Allen Institute for Cell Science"
__email__ = "jmaxfieldbrown@gmail.com, jamie.sherman@gmail.com, bowdenm@spu.edu"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "4.0.4"


def get_module_version() -> str:
    return __version__

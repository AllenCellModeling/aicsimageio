#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarks for general library operations and comparisons against other libraries.
"""

from functools import partial

from aicsimageio import imread_dask as aicsimageio_imread
from dask.array.image import imread as dask_array_imread
from dask_image.imread import imread as dask_image_imread
import pims

from .benchmark_image_containers import LOCAL_RESOURCES_DIR

###############################################################################

ACTK_OME_TIFF = str(LOCAL_RESOURCES_DIR / "actk.ome.tiff")

###############################################################################


class LibSuite:
    def time_base_import(self):
        """
        Benchmark how long it takes to import the library as a whole.
        """
        import aicsimageio  # noqa: F401


class LibCompareSuite:
    """
    Compare aicsimageio against other "just-in-time" image reading libs.
    """

    FUNC_LOOKUP = {
        "aicsimageio-default-chunks": partial(aicsimageio_imread, chunk_dims="ZYX"),
        "aicsimageio-plane-chunks": partial(aicsimageio_imread, chunk_dims="YX"),
        "dask-array-imread-default": dask_array_imread,
        "dask-image-imread-default": dask_image_imread,
    }

    params = [
        "aicsimageio-default-chunks",
        "aicsimageio-plane-chunks",
        "dask-array-imread-default",
        "dask-array-imread-pims",
        "dask-image-imread-default",
    ]

    def time_lib_config(self, func_name):
        func = self.FUNC_LOOKUP[func_name]
        func(ACTK_OME_TIFF).compute()

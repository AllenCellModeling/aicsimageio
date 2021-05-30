#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarks for general library operations and comparisons against other libraries.
"""

from dask_image.imread import imread
from aicsimageio import imread_dask

from .benchmark_image_containers import LOCAL_RESOURCES_DIR

###############################################################################

VARIANCE_CFE_OME_TIFF = str(LOCAL_RESOURCES_DIR / "variance-cfe.ome.tiff")

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

    def time_aicsimageio_chunk_zyx(self):
        imread_dask(VARIANCE_CFE_OME_TIFF, chunk_dims="ZYX").compute()

    def time_aicsimageio_chunk_yx(self):
        imread_dask(VARIANCE_CFE_OME_TIFF, chunk_dims="YX").compute()

    def time_dask_image(self):
        imread(VARIANCE_CFE_OME_TIFF).compute()

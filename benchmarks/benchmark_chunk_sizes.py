#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path

from aicsimageio import AICSImage

from .benchmark_image_containers import _ImageContainerTimeSuite

###############################################################################

# We only benchmark against local files as remote files are covered by unit tests
# and are generally slower than local but scale at a similar rate.
LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)

###############################################################################


class ChunkSuite(_ImageContainerTimeSuite):

    params = (
        [
            "YX",
            "ZYX",
            "CYX",
            "CZYX",
        ],
        sorted(
            [
                str(f)
                for f in [
                    LOCAL_RESOURCES_DIR / "actk.ome.tiff",
                    LOCAL_RESOURCES_DIR / "pipeline-4.ome.tiff",
                    LOCAL_RESOURCES_DIR / "pre-variance-cfe.ome.tiff",
                    LOCAL_RESOURCES_DIR / "variance-cfe.ome.tiff",
                ]
            ]
        ),
    )

    def setup(self, img_path, chunk_dims):
        random.seed(42)
        self.ImageContainer = AICSImage

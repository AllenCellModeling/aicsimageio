#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path

from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.readers.bioformats_reader import BioformatsReader

from .benchmark_image_containers import _ImageContainerTimeSuite

###############################################################################

# We only benchmark against local files as remote files are covered by unit tests
# and are generally slower than local but scale at a similar rate.
LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)

###############################################################################


class NativePythonCompareSuite(_ImageContainerTimeSuite):
    # This suite measures the effect of switching from Bioformats to a native Python
    # file format reader can have.
    # All readers will be different, but we do want to show the drawback of reading
    # using a Java pipe.

    params = (
        [
            str(LOCAL_RESOURCES_DIR / "pre-variance-cfe.ome.tiff"),
            str(LOCAL_RESOURCES_DIR / "variance-cfe.ome.tiff"),
        ],
        # We don't go above chunking by three dims because it would be rare
        # to do so... if you can read four-plus dims in a single chunk why can't you
        # just read in the whole image at once.
        # We also use CYX here to show that chunking with the _wrong_ dimensions can
        # result in longer processing times.
        [
            "YX",
            "ZYX",
            "CYX",
        ],
        [OmeTiffReader, BioformatsReader],
    )

    def setup(self, img_path, chunk_dims, reader):
        random.seed(42)
        self.ImageContainer = reader

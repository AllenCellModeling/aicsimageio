#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
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
    # This suite measures the effect that changing the default chunk dims
    # has on the duration of various reads.
    # We would expect that processing speed can be optimized based off of the
    # dimensions of the file and what the user is trying to do with said file.
    # i.e. If the user wants to normalize each channel and make a max projection
    # through Z, then the default of 'ZYX' is preferred over just 'YX'.
    # During this suite we not only benchmark the above example but also
    # file reading under the various chunk configurations as a monitor
    # for general read performance.

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
    )

    def time_norm_and_project(self, img_path, chunk_dims):
        """
        Benchmark how long a norm and project through Z takes
        under various chunk dims configurations.
        """
        # Init image container
        r = self.ImageContainer(img_path, chunk_dims=chunk_dims)

        # Store all delayed projections
        projs = []

        # Only run a random sample of two channels instead of all
        selected_channels = random.sample(r.channel_names, 2)
        for i, channel_name in enumerate(r.channel_names):
            if channel_name in selected_channels:
                # Select each channel
                data = r.get_image_dask_data("ZYX", C=i)

                # Get percentile norm by values
                min_px_val, max_px_val = da.percentile(
                    data.flatten(),
                    [50.0, 99.8],
                ).compute()

                # Norm
                normed = (data - min_px_val) / (max_px_val - min_px_val)

                # Clip any values outside of 0 and 1
                clipped = da.clip(normed, 0, 1)

                # Scale them between 0 and 255
                scaled = clipped * 255

                # Create max project
                projs.append(scaled.max(axis=0))

        # Compute all projections
        projs = da.stack(projs)
        projs.compute()

    def setup(self, img_path, chunk_dims):
        random.seed(42)
        self.ImageContainer = AICSImage

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path

from aicsimageio import AICSImage, readers
from aicsimageio.dimensions import DimensionNames

###############################################################################

# We only benchmark against local files as remote files are covered by unit tests
# and are generally slower than local but scale at a similar rate.
LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)

###############################################################################


class _ImageContainerMemorySuite:
    def peakmem_init(self, img_path):
        """
        Benchmark how much memory is used for just the initialized image container.
        """
        return self.ImageContainer(img_path)

    def peakmem_delayed_array(self, img_path):
        """
        Benchmark how much memory is used for the image container once the
        delayed dask array is constructed.

        Serves as a comparison against the init.
        Metadata should account for most of the memory difference.
        """
        r = self.ImageContainer(img_path)
        r.dask_data
        return r

    def peakmem_cached_array(self, img_path):
        """
        Benchmark how much memory is used for the whole image container once the
        current scene is read into memory.

        Serves as a comparison against the delayed construct and as a sanity check.
        Estimate: `r.data.size * r.data.itemsize` + some metadata and object overhead.
        """
        r = self.ImageContainer(img_path)
        r.data
        return r


class _ImageContainerTimeSuite:

    # These default chunk dimensions don't exist on every image container
    # so we have to define them here as well
    DEFAULT_CHUNK_DIMS = [
        DimensionNames.SpatialZ,
        DimensionNames.SpatialY,
        DimensionNames.SpatialX,
        DimensionNames.Samples,
    ]

    def time_init(self, img_path):
        """
        Benchmark how long it takes to validate a file and finish general setup.
        """
        self.ImageContainer(img_path)

    def time_delayed_array_construct(self, img_path):
        """
        Benchmark how long it takes to construct the delayed dask array for a file.
        """
        self.ImageContainer(img_path).dask_data

    def time_random_single_chunk_read(self, img_path):
        """
        Benchmark how long it takes to read a single chunk out of a file.

        I.E. "Pull just the Brightfield channel z-stack.
        """
        r = self.ImageContainer(img_path)

        random_index_selections = {}
        for dim, size in zip(r.dims.order, r.dims.shape):
            if dim not in self.DEFAULT_CHUNK_DIMS:
                random_index_selections[dim] = random.randint(0, size - 1)

        valid_dims_to_return = "".join(
            [d for d in r.dims.order if d in self.DEFAULT_CHUNK_DIMS]
        )
        r.get_image_dask_data(valid_dims_to_return, **random_index_selections).compute()

    def time_random_many_chunk_read(self, img_path):
        """
        Open a file, get many chunks out of the file at once.

        I.E. "Pull the DNA and Nucleus channel z-stacks, for the middle 50% timepoints".
        """
        r = self.ImageContainer(img_path)

        random_index_selections = {}
        for dim, size in zip(r.dims.order, r.dims.shape):
            if dim not in self.DEFAULT_CHUNK_DIMS:
                a = random.randint(0, size - 1)
                b = random.randint(0, size - 1)
                lower = min(a, b)
                upper = max(a, b)
                random_index_selections[dim] = slice(lower, upper, 1)

        r.get_image_dask_data(r.dims.order, **random_index_selections).compute()


###############################################################################
# ImageContainer benchmarks


class DefaultReaderSuite(_ImageContainerTimeSuite, _ImageContainerMemorySuite):
    params = [
        # We can't check any of the ffmpeg formats because asv doesn't run
        # properly with spawned subprocesses and the ffmpeg formats all
        # passthrough the request to ffmpeg...
        #
        # Because of this, these benchmarks are largely sanity checks
        sorted(
            [
                str(LOCAL_RESOURCES_DIR / "example.bmp"),
                str(LOCAL_RESOURCES_DIR / "example.jpg"),
                str(LOCAL_RESOURCES_DIR / "example.png"),
            ]
        ),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ImageContainer = readers.DefaultReader


class TiffReaderSuite(_ImageContainerTimeSuite, _ImageContainerMemorySuite):
    params = [
        sorted([str(f) for f in LOCAL_RESOURCES_DIR.glob("*.tiff")]),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ImageContainer = readers.TiffReader


class OmeTiffReaderSuite(_ImageContainerTimeSuite, _ImageContainerMemorySuite):
    params = [
        sorted([str(f) for f in LOCAL_RESOURCES_DIR.glob("*.ome.tiff")]),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ImageContainer = readers.OmeTiffReader


class LifReaderSuite(_ImageContainerTimeSuite, _ImageContainerMemorySuite):
    params = [
        sorted([str(f) for f in LOCAL_RESOURCES_DIR.glob("*.lif")]),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ImageContainer = readers.LifReader


class AICSImageSuite(_ImageContainerTimeSuite, _ImageContainerMemorySuite):
    # This suite utilizes the same suite that the base readers do.
    # In all cases, the time or peak memory used by AICSImage should
    # be minimal additional overhead from the base reader.

    params = list(
        set(
            DefaultReaderSuite.params[0]
            + TiffReaderSuite.params[0]
            + OmeTiffReaderSuite.params[0]
            + LifReaderSuite.params[0]
        )
    )

    def setup(self, img_path):
        random.seed(42)
        self.ImageContainer = AICSImage

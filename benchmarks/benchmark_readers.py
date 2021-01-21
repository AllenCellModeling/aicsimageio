#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path

from aicsimageio import readers
from aicsimageio.dimensions import DimensionNames

###############################################################################

# We only benchmark against local files as remote files are covered by unit tests
# and are generally slower than local but scale at a similar rate.
LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)

###############################################################################


class _ReaderMemorySuite:
    def peakmem_init(self, img_path):
        """
        Benchmark how much memory is used for just the initialized reader.
        """
        return self.ReaderClass(img_path)

    def peakmem_delayed_array(self, img_path):
        """
        Benchmark how much memory is used for the reader once the
        delayed dask array is constructed.

        Serves as a comparison against the init.
        Metadata should account for most of the memory difference.
        """
        r = self.ReaderClass(img_path)
        r.dask_data
        return r

    def peakmem_cached_array(self, img_path):
        """
        Benchmark how much memory is used for the whole reader once the
        current scene is read into memory.

        Serves as a comparison against the delayed construct and as a sanity check.
        Estimate: `r.data.size * r.data.itemsize` + some metadata and object overhead.
        """
        r = self.ReaderClass(img_path)
        r.data
        return r


class _ReaderTimeSuite:

    # These default chunk dimensions don't exist on every reader
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
        self.ReaderClass(img_path)

    def time_delayed_array_construct(self, img_path):
        """
        Benchmark how long it takes to construct the delayed dask array for a file.
        """
        self.ReaderClass(img_path).dask_data

    def time_random_single_chunk_read(self, img_path):
        """
        Benchmark how long it takes to read a single chunk out of a file.

        I.E. "Pull just the Brightfield channel z-stack.
        """
        r = self.ReaderClass(img_path)

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
        r = self.ReaderClass(img_path)

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
# Reader benchmarks


class DefaultReaderSuite(_ReaderTimeSuite, _ReaderMemorySuite):
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
        self.ReaderClass = readers.DefaultReader


class TiffReaderSuite(_ReaderTimeSuite, _ReaderMemorySuite):
    params = [
        sorted([str(f) for f in LOCAL_RESOURCES_DIR.glob("*.tiff")]),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ReaderClass = readers.TiffReader


class OmeTiffReaderSuite(_ReaderTimeSuite):
    params = [
        sorted([str(f) for f in LOCAL_RESOURCES_DIR.glob("*.ome.tiff")]),
    ]

    def setup(self, img_path):
        random.seed(42)
        self.ReaderClass = readers.OmeTiffReader

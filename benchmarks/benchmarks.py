# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import random
from pathlib import Path

from aicsimageio import readers
from aicsimageio.dimensions import DimensionNames

###############################################################################

LOCAL = "LOCAL"
REMOTE = "REMOTE"

LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)
REMOTE_RESOURCES_DIR = (
    "s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources"
)


def get_resource_full_path(host, filename):
    if host is LOCAL:
        return LOCAL_RESOURCES_DIR / filename
    elif host is REMOTE:
        return f"{REMOTE_RESOURCES_DIR}/{filename}"


###############################################################################


class _ImageSuite:
    # Benchmark utils
    def _init_reader(self, host, fname):
        return self.ReaderClass(get_resource_full_path(host, fname))


class _ReaderMemorySuite(_ImageSuite):
    def mem_init(self, host, fname):
        """
        How much memory do we use for the fully delayed object.
        """
        return self._init_reader(host, fname)

    def mem_delayed_image(self, host, fname):
        """
        Over the init, not much additional memory should be used.
        Metadata should account for most of the difference.
        """
        r = self._init_reader(host, fname)
        r.dask_data
        return r

    def mem_cached_image(self, host, fname):
        """
        Sanity benchmark for us.
        Serves as a comparison against the array construct.
        """
        r = self._init_reader(host, fname)
        r.data
        return r


class _ReaderTimeSuite(_ImageSuite):
    def time_init(self, host, fname):
        self._init_reader(host, fname)

    def time_array_construct(self, host, fname):
        self._init_reader(host, fname).dask_data

    def time_random_plane_read(self, host, fname):
        r = self._init_reader(host, fname)

        random_index_selections = {}
        for dim, size in zip(r.dims.order, r.dims.shape):
            if dim not in [DimensionNames.SpatialY, DimensionNames.SpatialX]:
                random_index_selections[dim] = random.randint(0, size - 1)

        r.get_image_dask_data("YX", **random_index_selections).compute()

    def time_random_chunk_read(self, host, fname):
        r = self._init_reader(host, fname)

        random_index_selections = {}
        for dim, size in zip(r.dims.order, r.dims.shape):
            if dim not in [DimensionNames.SpatialY, DimensionNames.SpatialX]:
                a = random.randint(0, size - 1)
                b = random.randint(0, size - 1)
                lower = min(a, b)
                upper = max(a, b)
                random_index_selections[dim] = slice(lower, upper, 1)

        r.get_image_dask_data(r.dims.order, **random_index_selections)

    def time_numpy_read(self, host, fname):
        self._init_reader(host, fname).data


###############################################################################
# Reader benchmarks


class DefaultReaderSuit(_ReaderTimeSuite):
    params = [
        # host params
        [LOCAL],
        # fname params
        [
            "example_valid_frame_count.mp4",
            "example.bmp",
            "example.gif",
            "example.jpg",
            "example.png",
        ],
    ]

    def setup(self, host, fname):
        random.seed(666)
        self.ReaderClass = readers.DefaultReader


class TiffReaderSuite(_ReaderTimeSuite):
    params = [
        # host params
        [LOCAL],
        # fname params
        sorted([f.name for f in LOCAL_RESOURCES_DIR.glob("*.tiff")]),
    ]

    def setup(self, host, fname):
        random.seed(666)
        self.ReaderClass = readers.TiffReader


class OmeTiffReaderSuite(_ReaderTimeSuite):
    params = [
        # host params
        [LOCAL],
        # fname params
        sorted([f.name for f in LOCAL_RESOURCES_DIR.glob("*.ome.tiff")]),
    ]

    def setup(self, host, fname):
        random.seed(666)
        self.ReaderClass = readers.OmeTiffReader

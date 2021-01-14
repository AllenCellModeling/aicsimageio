# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import random
from pathlib import Path

from aicsimageio.dimensions import DimensionNames
from aicsimageio.readers import TiffReader

###############################################################################

LOCAL = "LOCAL"
REMOTE = "REMOTE"

LOCAL_RESOURCES_DIR = (
    Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
)
REMOTE_RESOURCES_DIR = (
    "s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources"
)


def get_resource_full_path(filename, host):
    if host is LOCAL:
        return LOCAL_RESOURCES_DIR / filename
    elif host is REMOTE:
        return f"{REMOTE_RESOURCES_DIR}/{filename}"


###############################################################################


class TiffReaderSuite:
    params = [
        # tiff_path params
        sorted([f.name for f in LOCAL_RESOURCES_DIR.glob("*.tiff")]),
        # host params
        [LOCAL],
    ]

    def setup(self, tiff_path, host):
        random.seed(1612)

    def time_init(self, tiff_path, host):
        TiffReader(get_resource_full_path(tiff_path, host))

    def time_array_construct(self, tiff_path, host):
        TiffReader(get_resource_full_path(tiff_path, host)).dask_data

    def time_first_plane_read(self, tiff_path, host):
        r = TiffReader(get_resource_full_path(tiff_path, host))
        r.get_image_dask_data("YX").compute()

    def time_random_plane_read(self, tiff_path, host):
        r = TiffReader(get_resource_full_path(tiff_path, host))

        random_index_selections = {}
        for dim, size in zip(r.dims.order, r.dims.shape):
            if dim not in [DimensionNames.SpatialY, DimensionNames.SpatialX]:
                random_index_selections[dim] = random.randint(0, size - 1)

        r.get_image_dask_data("YX", **random_index_selections).compute()

    def time_numpy_read(self, tiff_path, host):
        TiffReader(get_resource_full_path(tiff_path, host)).data

    def time_dask_read(self, tiff_path, host):
        TiffReader(get_resource_full_path(tiff_path, host)).dask_data.compute()

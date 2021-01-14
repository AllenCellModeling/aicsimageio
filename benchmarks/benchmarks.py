# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from pathlib import Path

from aicsimageio.readers import TiffReader

###############################################################################

RESOURCES_DIR = Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"

###############################################################################


class LocalTiffReaderSuite:
    tiff_paths = list(RESOURCES_DIR.glob("*.tiff"))

    def setup(self):
        self.files = {}

    def time_init(
        self,
        tiff_path,
    ):
        self.files[tiff_path] = TiffReader(tiff_path)

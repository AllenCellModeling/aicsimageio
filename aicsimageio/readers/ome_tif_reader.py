import logging
import io
import re
import tifffile

from ..vendor import omexml
from .reader import Reader
from .tiff_reader import TiffReader
from .. import types


log = logging.getLogger(__name__)


class OmeTifReader(Reader):
    """Opening and processing the contents of an OME Tiff file
    """

    def __init__(self, file: types.Union[types.PathLike, types.BufferLike]):
        super().__init__(file)
        try:
            self.tif = tifffile.TiffFile(self._bytes)
        except Exception:
            log.error("tiffile could not parse this input")
            raise

    def _lazy_init_metadata(self):
        if self._metadata is None and self.tif.is_ome:
            d = self.tif.pages[0].description.strip()
            assert d.startswith("<?xml version=") and d.endswith("</OME>")
            self._metadata = omexml.OMEXML(d)
        return self._metadata

    def _lazy_init_data(self):
        if self._data is None:
            # load the data
            self._data = self.tif.asarray()
        return self._data

    @staticmethod
    def _is_this_type(byte_io: io.BytesIO) -> bool:
        is_tif = TiffReader._is_this_type(byte_io)
        if is_tif:
            buf = TiffReader.get_image_description(byte_io)
            if buf[0:5] != b"<?xml":
                return False
            match = re.search(
                b'<(\\w*)(:?)OME [^>]*xmlns\\2\\1="http://www.openmicroscopy.org/Schemas/[Oo][Mm][Ee]/',
                buf,
            )
            if match is None:
                return False
            return True
        return False

    @property
    def data(self) -> types.SixDArray:
        return self._lazy_init_data()

    @property
    def dims(self) -> str:
        self._lazy_init_metadata()
        dimension_order = self._metadata.image().Pixels.DimensionOrder
        # reverse the string
        dimension_order = dimension_order[::-1]
        return dimension_order

    @property
    def metadata(self) -> omexml.OMEXML:
        return self._lazy_init_metadata()

    def load_slice(self, slice_index=0):
        data = self.tif.asarray(key=slice_index)
        return data

    def size_z(self):
        return self.metadata.image().Pixels.SizeZ

    def size_c(self):
        return self.metadata.image().Pixels.SizeC

    def size_t(self):
        return self.metadata.image().Pixels.SizeT

    def size_x(self):
        return self.metadata.image().Pixels.SizeX

    def size_y(self):
        return self.metadata.image().Pixels.SizeY

    def dtype(self):
        return self.tif.pages[0].dtype

    def is_ome(self):
        return OmeTifReader._is_this_type(self._bytes)

import io
import logging
import numpy as np
from typing import Optional
import warnings
import xml.etree

from aicsimageio import types
from .reader import Reader
from ..buffer_reader import BufferReader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ..vendor import czifile

log = logging.getLogger(__name__)
CZI_NATIVE_ARRAY = np.ndarray


class CziReader(Reader):
    """This class is used primarily for opening and processing the contents of a CZI file

    Example:
        reader = cziReader.CziReader(path="file.czi")
        file_image = reader.load()
        file_slice = reader.load_slice(t=1, z=2, c=3)

        with cziReader.CziReader(path="file2.czi") as reader:
            file2_image = reader.load()
            file2_slice = reader.load_slice(t=1, z=2, c=3)

        # Convert a CZI file into OME Tif.
        reader = cziReader.CziReader(path="file3.czi")
        writer = omeTifWriter.OmeTifWriter(path="file3.ome.tif")
        writer.save(reader.load())

    The load() function gathers all the slices into a single 5d array with dimensions TZCYX.
    This should be used when the entire image needs to be processed or transformed in some way.

    The load_slice() function takes a single 2D slice with dimensions YX out of the 5D image.
    This should be used when only a few select slices need to be processed
    (e.g. printing out the middle slice for a thumbnail image)

    This class has a similar interface to OmeTifReader.

    In order to better understand the inner workings of this class, it is necessary to
    know that CZI files can be abstracted as an n-dimensional array.

    CZI files contain an n-dimensional array.
    If t = 1, then the array will be 6 dimensional 'BCZYX0' (czifile.axes)
    Otherwise, the array will be 7 dimensional 'BTCZYX0' (czifile.axes)
    'B' is block acquisition from the CZI memory directory
    'T' is time
    'C' is the channel
    'Z' is the index of the slice in the image stack
    'X' and 'Y' correspond to the 2D slices
    '0' is the numbers of channels per pixel (always =zero for our data)
    """

    @staticmethod
    def _is_this_type(buffer: io.BufferedIOBase) -> bool:
        is_czi = False
        with BufferReader(buffer) as buffer_reader:
            if buffer_reader.endianness == b"ZI":
                magic = buffer_reader.endianness + bytearray(
                    buffer_reader.buffer.read(8)
                )
                # Per spec: CZI files are little-endian
                is_czi = magic == b"ZISRAWFILE"
                if is_czi:
                    buffer_reader.endianness = buffer_reader.INTEL_ENDIAN
        return is_czi

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        the data from the czi file with the native order (i.e. "TZCYX")
        """
        if self._data is None:
            # load the data
            self._data = self.czi.asarray(max_workers=self._max_workers)
        return self._data

    @property
    def dims(self) -> str:
        """
        Returns
        -------
        The native shape of the image.
        """
        return self.czi.axes

    @property
    def metadata(self) -> xml.etree.ElementTree:
        """
        Lazy load the metadata from the CZI file
        Returns
        -------
        The xml Element Tree of the metadata
        """
        if self._metadata is None:
            # load the metadata
            self._metadata = self.czi.metadata
        return self._metadata

    def __init__(self, file: types.FileLike, max_workers: Optional[int] = None):
        """
        :param file_path(str): The path for the file that is to be opened.
        """
        super().__init__(file)
        try:
            self.czi = czifile.CziFile(self._bytes)
        except Exception:
            log.error("czifile could not parse this input")
            raise

        self.has_time_dimension = "T" in self.czi.axes
        self._max_workers = max_workers

    def close(self):
        self.czi.close()
        super().close()

    def size_z(self):
        return self.czi.shape[3] if self.has_time_dimension else self.czi.shape[2]

    def size_c(self):
        return self.czi.shape[2] if self.has_time_dimension else self.czi.shape[1]

    def size_t(self):
        return self.czi.shape[1] if self.has_time_dimension else 1

    def size_x(self):
        return self.czi.shape[5] if self.has_time_dimension else self.czi.shape[4]

    def size_y(self):
        return self.czi.shape[4] if self.has_time_dimension else self.czi.shape[3]

    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        the data type of the ndarray being returned (uint16, uint8, etc)
        """
        return self.czi.dtype

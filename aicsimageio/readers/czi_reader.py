#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
from typing import Optional, Tuple
from xml.etree import ElementTree

import dask.array as da
import numpy as np
from aicspylibczi import CziFile

from .. import types
from ..buffer_reader import BufferReader
from ..exceptions import UnsupportedFileFormatError
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class CziReader(Reader):
    """
    CziReader wraps aicspylibczi to provide the same reading capabilities but abstracts the specifics of using the
    backend library to create a unified interface. This enables higher level functions to duck type the File Readers.

    Parameters
    ----------
    data: types.FileLike
        A string or path to the CZI file to be read.
    """
    ZEISS_2BYTE = b'ZI'             # First two characters of a czi file according to Zeiss docs
    ZEISS_10BYTE = b'ZISRAWFILE'    # First 10 characters of a well formatted czi file.

    def __init__(self, data: types.FileLike, **kwargs):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        with BufferReader(buffer) as buffer_reader:
            if buffer_reader.endianness != CziReader.ZEISS_2BYTE:
                return False
            header = buffer_reader.endianness + buffer_reader.read_bytes(8)
            return header == CziReader.ZEISS_10BYTE

    @property
    def data(self) -> da.core.Array:
        """
        Returns
        -------
        Constructed dask array where each chunk is a delayed read from the CZI file.
        Places dimensions in the native order (i.e. "TZCYX")
        """
        if self._data is None:
            # Construct dask array
            self._data = self.czi.asarray(max_workers=self._max_workers)
        return self._data

    @property
    def dims(self) -> str:
        return self.czi.axes

    @property
    def metadata(self) -> ElementTree:
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

    def get_channel_names(self, scene: int = 0):
        chelem = self.metadata.findall("./Metadata/Information/Image/Dimensions/Channels/Channel")
        return [ch.get("Name") for ch in chelem]

    # TODO refactor this utility function into a metadata wrapper class
    def _getmetadataxmltext(self, findpath, default=None):
        ref = self.metadata.find(findpath)
        if ref is None:
            return default
        return ref.text

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        px = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='X']/Value", "1.0"))
        py = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Y']/Value", "1.0"))
        pz = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Z']/Value", "1.0"))
        return (px, py, pz)

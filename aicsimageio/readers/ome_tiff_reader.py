#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import re
from typing import Tuple

from tifffile import TiffFile

from .. import types
from ..vendor import omexml
from .tiff_reader import TiffReader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class OmeTiffReader(TiffReader):
    """
    Opening and processing the contents of an OME Tiff file
    """

    def __init__(self, data: types.FileLike, **kwargs):
        super().__init__(data, **kwargs)

        # Lazy load is ome
        self._is_ome = None

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        is_tif = TiffReader._is_this_type(buffer)
        if is_tif:
            buf = TiffReader.get_image_description(buffer)
            if buf is None:
                return False
            if buf[0:5] != b"<?xml":
                return False
            match = re.search(
                b'<(\\w*)(:?)OME [^>]*xmlns\\2\\1="http://www.openmicroscopy.org/Schemas/[Oo][Mm][Ee]/',
                buf
            )
            if match is None:
                return False
            return True
        return False

    def _lazy_init_metadata(self) -> omexml.OMEXML:
        with TiffFile(self._file) as tiff:
            if self._metadata is None and tiff.is_ome:
                description = tiff.pages[0].description.strip()
                if not (description.startswith("<?xml version=") and description.endswith("</OME>")):
                    raise ValueError(f"Description does not conform to OME specification: {description[:100]}")
                self._metadata = omexml.OMEXML(description)
        return self._metadata

    def is_ome(self):
        return self.is_this_type(self._file)

    @property
    def metadata(self) -> omexml.OMEXML:
        if self._metadata is None:
            return self._lazy_init_metadata()

        return self._metadata

    def size_s(self) -> int:
        return self.metadata.image_count

    def size_t(self) -> int:
        return self.metadata.image().Pixels.SizeT

    def size_c(self) -> int:
        return self.metadata.image().Pixels.SizeC

    def size_z(self) -> int:
        return self.metadata.image().Pixels.SizeZ

    def size_y(self) -> int:
        return self.metadata.image().Pixels.SizeY

    def size_x(self) -> int:
        return self.metadata.image().Pixels.SizeX

    def get_channel_names(self, scene: int = 0):
        return self.metadata.image(scene).Pixels.get_channel_names()

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        p = self.metadata.image(scene).Pixels
        return (p.get_PhysicalSizeX(), p.get_PhysicalSizeY(), p.get_PhysicalSizeZ())

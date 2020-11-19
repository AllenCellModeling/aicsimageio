#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError

from .. import exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .tiff_reader import TiffReader

###############################################################################

class OmeTiffReader(TiffReader):

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    return tiff.pages[0].is_ome

        except (TiffFileError, TypeError):
            return False

    def __init__(self, image: types.PathLike):
        """
        Wraps the tifffile and ome-types APIs to provide the same aicsimageio Reader
        API but for volumetric OME Tiff images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension = self.path.split(".")[-1]

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path):
            raise exceptions.UnsupportedFileFormatError(self.extension)

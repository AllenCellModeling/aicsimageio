#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
from dask import delayed
from tifffile import TiffFile

from .. import types
from ..buffer_reader import BufferReader
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class TiffReader(Reader):
    """
    This class is used to open and process the contents of a generic tiff file.

    This will create create a delayed dask array where each chunk is a ZYX plane.
    """

    def __init__(self, data: types.FileLike, S: int = 0, **kwargs):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

        # Store parameters needed for dask read
        self.specific_s_index = S

    @staticmethod
    def _is_this_type(buffer: io.BufferedIOBase) -> bool:
        with BufferReader(buffer) as buffer_reader:
            # Per the TIFF-6 spec (https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf),
            # 'II' is little-endian (Intel format) and 'MM' is big-endian (Motorola format)
            if buffer_reader.endianness not in [buffer_reader.INTEL_ENDIAN, buffer_reader.MOTOROLA_ENDIAN]:
                return False
            magic = buffer_reader.read_uint16()

            # Per TIFF-6, magic is 42.
            if magic == 42:
                ifd_offset = buffer_reader.read_uint32()
                if ifd_offset == 0:
                    return False

            # Per BigTIFF (https://www.awaresystems.be/imaging/tiff/bigtiff.html), magic is 43.
            if magic == 43:
                # Alex magic here...
                if buffer_reader.read_uint16() != 8:
                    return False
                if buffer_reader.read_uint16() != 0:
                    return False
                ifd_offset = buffer_reader.read_uint64()
                if ifd_offset == 0:
                    return False
            return True

    @staticmethod
    def _imread(
        img: Path,
        scene: int,
        page: int
    ) -> np.ndarray:
        # Load Tiff
        with TiffFile(img) as tiff:
            # Get proper scene
            scene = tiff.series[scene]

            # Get proper page
            page = scene.pages[page]

            # Return numpy
            return page.asarray()

    @property
    def dask_data(self) -> da.core.Array:
        if self._dask_data is None:
            # Load Tiff
            with TiffFile(self._file) as tiff:
                # Check each scene has the same shape
                # If scene shape checking fails, use the specified scene and update operating shape
                scenes = tiff.series
                operating_shape = scenes[0].shape
                for scene in scenes:
                    if scene.shape != operating_shape:
                        operating_shape = scenes[self.specific_s_index].shape
                        scenes = [scenes[self.specific_s_index]]

                # Get sample yx plane
                sample = scenes[0].pages[0].asarray()

                # Combine length of scenes and operating shape
                # Replace YX dims with empty dimensions
                operating_shape = (len(scenes), *operating_shape)
                operating_shape = operating_shape[:-2] + (1, 1)

                # Make ndarray for lazy arrays to fill
                lazy_arrays = np.ndarray(operating_shape, dtype=object)
                for all_page_index, np_tuple in enumerate(np.ndenumerate(lazy_arrays)):
                    # Unpack np_tuple
                    np_index, _ = np_tuple

                    # Scene index is the first index in np_index
                    scene_index = np_index[0]

                    # This page index is current enumeration divided by scene index + 1
                    # For example if the image has 10 Z slices and 5 scenes, there would be 50 total pages
                    this_page_index = all_page_index // (scene_index + 1)

                    # Fill the numpy array with the delayed arrays
                    lazy_arrays[np_index] = da.from_delayed(
                        delayed(TiffReader._imread)(self._file, scene_index, this_page_index),
                        shape=sample.shape,
                        dtype=sample.dtype
                    )

                # Convert the numpy array of lazy readers into a dask array
                self._dask_data = da.block(lazy_arrays.tolist())

        return self._dask_data

    @property
    def dims(self) -> str:
        if self._dims is None:
            # Get a single scenes dimensions in order
            with TiffFile(self._file) as tiff:
                single_scene_dims = tiff.series[0].pages.axes

            self._dims = f"S{single_scene_dims}"

        return self._dims

    @staticmethod
    def get_image_description(buffer: io.BufferedIOBase) -> Optional[bytearray]:
        """Retrieve the image description as one large string."""
        description_length = 0
        description_offset = 0

        with BufferReader(buffer) as buffer_reader:
            # Per the TIFF-6 spec (https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf),
            # 'II' is little-endian (Intel format) and 'MM' is big-endian (Motorola format)
            if buffer_reader.endianness not in [buffer_reader.INTEL_ENDIAN, buffer_reader.MOTOROLA_ENDIAN]:
                return None
            magic = buffer_reader.read_uint16()

            # Per TIFF-6, magic is 42.
            if magic == 42:
                found = False
                while not found:
                    ifd_offset = buffer_reader.read_uint32()
                    if ifd_offset == 0:
                        return None
                    buffer_reader.buffer.seek(ifd_offset, 0)
                    entries = buffer_reader.read_uint16()
                    for n in range(0, entries):
                        tag = buffer_reader.read_uint16()
                        type = buffer_reader.read_uint16()
                        count = buffer_reader.read_uint32()
                        offset = buffer_reader.read_uint32()
                        if tag == 270:
                            description_length = count - 1  # drop the NUL from the end
                            description_offset = offset
                            found = True
                            break

            # Per BigTIFF (https://www.awaresystems.be/imaging/tiff/bigtiff.html), magic is 43.
            if magic == 43:
                # Alex magic here...
                if buffer_reader.read_uint16() != 8:
                    return None
                if buffer_reader.read_uint16() != 0:
                    return None
                found = False
                while not found:
                    ifd_offset = buffer_reader.read_uint64()
                    if ifd_offset == 0:
                        return None
                    buffer_reader.buffer.seek(ifd_offset, 0)
                    entries = buffer_reader.read_uint64()
                    for n in range(0, entries):
                        tag = buffer_reader.read_uint16()
                        type = buffer_reader.read_uint16()  # noqa: F841
                        count = buffer_reader.read_uint64()
                        offset = buffer_reader.read_uint64()
                        if tag == 270:
                            description_length = count - 1  # drop the NUL from the end
                            description_offset = offset
                            found = True
                            break

            if description_offset == 0:
                # Nothing was found
                return bytearray("")
            else:
                buffer_reader.buffer.seek(description_offset, 0)
                return bytearray(buffer_reader.buffer.read(description_length))

    @property
    def metadata(self) -> str:
        if self._metadata is None:
            with open(self._file, "rb") as rb:
                description = self.get_image_description(rb)
            if description is None:
                self._metadata = ""
            else:
                self._metadata = description.decode()
        return self._metadata

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

from .. import exceptions, types
from ..buffer_reader import BufferReader
from ..constants import Dimensions
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class TiffReader(Reader):
    """
    TiffReader wraps tifffile to provide the same reading capabilities but abstracts the specifics of using the
    backend library to create a unified interface. This enables higher level functions to duck type the File Readers.

    Parameters
    ----------
    data: types.FileLike
        A string or path to the TIFF file to be read.
    S: int
        If the image has different dimensions on any scene from another, the dask array construction will fail.
        In that case, use this parameter to specify a specific scene to construct a dask array for.
        Default: 0 (select the first scene)
    """

    def __init__(self, data: types.FileLike, S: int = 0, **kwargs):
        # Run super init to check filepath provided
        super().__init__(data, **kwargs)

        # Store parameters needed for dask read
        self.specific_s_index = S

        # Lazy load and hold on to dtype
        self._dtype = None

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
        """
        Read a TIFF image file as a delayed dask array where each chunk of the constructed array is a delayed YX plane.

        Returns
        -------
        img: dask.array.core.Array
            The constructed delayed YX plane dask array.
        """
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
                        log.info(
                            f"File contains variable dimensions per scene, "
                            f"selected scene: {self.specific_s_index} for data retrieval."
                        )
                        break

                # Get sample yx plane
                sample = scenes[0].pages[0].asarray()

                # Combine length of scenes and operating shape
                # Replace YX dims with empty dimensions
                operating_shape = (len(scenes), *operating_shape)
                operating_shape = operating_shape[:-2] + (1, 1)

                # Make ndarray for lazy arrays to fill
                lazy_arrays = np.ndarray(operating_shape, dtype=object)
                for all_page_index, (np_index, _) in enumerate(np.ndenumerate(lazy_arrays)):
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
                data = da.block(lazy_arrays.tolist())

                # Only return the scene dimension if multiple scenes are present
                if len(scenes) == 1:
                    data = data[0, :]

                # Set _dask_data
                self._dask_data = data

        return self._dask_data

    def load_slice(self, slice_index: int = 0) -> np.ndarray:
        with TiffFile(self._file) as tiff:
            return tiff.asarray(key=slice_index)

    def dtype(self):
        if self._dtype is None:
            with TiffFile(self._file) as tiff:
                self._dtype = tiff.pages[0].dtype

        return self._dtype

    @property
    def dims(self) -> str:
        if self._dims is None:
            # Get a single scenes dimensions in order
            with TiffFile(self._file) as tiff:
                single_scene_dims = tiff.series[0].pages.axes

                # We can sometimes trust the dimension info in the image
                if all([d in Dimensions.DefaultOrder for d in single_scene_dims]):
                    # Add scene dimension only if there are multiple scenes
                    if len(tiff.series) == 1:
                        self._dims = single_scene_dims
                    else:
                        self._dims = f"{Dimensions.Scene}{single_scene_dims}"
                # Sometimes the dimension info is wrong in certain dimensions, so guess that dimension
                else:
                    guess = self.guess_dim_order(tiff.series[0].pages.shape)
                    best_guess = []
                    for dim_from_meta in single_scene_dims:
                        if dim_from_meta in Dimensions.DefaultOrder:
                            best_guess.append(dim_from_meta)
                        else:
                            appended_dim = False
                            for guessed_dim in guess:
                                if guessed_dim not in best_guess:
                                    best_guess.append(guessed_dim)
                                    appended_dim = True
                                    log.info(
                                        f"Unsure how to handle dimension: {dim_from_meta}. "
                                        f"Replaced with guess: {guessed_dim}"
                                    )
                                    break

                            # All of our guess dims were already in the dim list, append the dim read from meta
                            if not appended_dim:
                                best_guess.append(dim_from_meta)

                    best_guess = "".join(best_guess)

                    # Add scene dimension only if there are multiple scenes
                    if len(tiff.series) == 1:
                        self._dims = best_guess
                    else:
                        self._dims = f"{Dimensions.Scene}{best_guess}"

        return self._dims

    @dims.setter
    def dims(self, dims: str):
        # Check amount of provided dims against data shape
        if len(dims) != len(self.dask_data.shape):
            raise exceptions.InvalidDimensionOrderingError(
                f"Provided too many dimensions for the associated file. "
                f"Received {len(dims)} dimensions [dims: {dims}] "
                f"for image with {len(self.data.shape)} dimensions [shape: {self.data.shape}]."
            )

        # Set the dims
        self._dims = dims

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
from aicsimageio import AICSImage, types
from aicsimageio.readers.reader import Reader
from distributed.protocol import deserialize, serialize
from fsspec.implementations.local import LocalFileSystem
from psutil import Process
from xarray.testing import assert_equal

###############################################################################


def check_local_file_not_open(image_container: Union[AICSImage, Reader]):
    if isinstance(image_container, AICSImage):
        image_container = image_container.reader

    # Check that there are no open file pointers
    if isinstance(image_container.fs, LocalFileSystem):
        proc = Process()
        assert str(image_container.path) not in [f.path for f in proc.open_files()]


def check_can_serialize_image_container(image_container: Union[AICSImage, Reader]):
    # Dump and reconstruct
    reconstructed = deserialize(*serialize(image_container))

    # Assert primary attrs are equal
    if image_container._xarray_data is None:
        assert reconstructed._xarray_data is None
    else:
        assert_equal(image_container._xarray_data, reconstructed._xarray_data)

    if image_container._xarray_dask_data is None:
        assert reconstructed._xarray_dask_data is None
    else:
        assert_equal(image_container._xarray_dask_data, reconstructed._xarray_dask_data)


def run_image_read_checks(
    ImageContainer: ClassVar[Union[AICSImage, Reader]],
    uri: types.PathLike,
    set_scene: str,
    expected_scenes: Tuple[str],
    expected_current_scene: str,
    expected_shape: Tuple[int],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[float],
) -> Union[AICSImage, Reader]:
    """
    A general suite of tests to run against image containers (Reader and AICSImage).
    """
    # Read file
    image_container = ImageContainer(uri)

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    # Set scene
    image_container.set_scene(set_scene)

    # Check scene info
    assert image_container.scenes == expected_scenes
    assert image_container.current_scene == expected_current_scene

    # Check basics
    assert image_container.shape == expected_shape
    assert image_container.dtype == expected_dtype
    assert image_container.dims.order == expected_dims_order
    assert image_container.dims.shape == expected_shape
    assert image_container.metadata
    assert image_container.channel_names == expected_channel_names
    assert image_container.physical_pixel_sizes == expected_physical_pixel_sizes

    # Read only a chunk, then read a chunk from the in-memory, compare
    np.testing.assert_array_equal(
        image_container.get_image_dask_data("YX").compute(),
        image_container.get_image_data("YX"),
    )

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == expected_shape
    assert image_container.data.dtype == expected_dtype

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    return image_container


def run_multi_scene_image_read_checks(
    ImageContainer: ClassVar[Union[AICSImage, Reader]],
    uri: types.PathLike,
    first_scene_id: str,
    first_scene_shape: Tuple[int],
    first_scene_dtype: np.dtype,
    second_scene_id: str,
    second_scene_shape: Tuple[int],
    second_scene_dtype: np.dtype,
) -> Union[AICSImage, Reader]:
    """
    A suite of tests to ensure that data is reset when switching scenes.
    """
    # Read file
    image_container = ImageContainer(uri)

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    # Set scene
    image_container.set_scene(first_scene_id)

    # Check basics
    assert image_container.shape == first_scene_shape
    assert image_container.dtype == first_scene_dtype

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == first_scene_shape
    assert image_container.data.dtype == first_scene_dtype

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    # Change scene
    image_container.set_scene(second_scene_id)

    # Check data was reset
    assert image_container._xarray_dask_data is None
    assert image_container._xarray_data is None
    assert image_container._dims is None
    assert image_container._metadata is None

    # Check basics
    assert image_container.shape == second_scene_shape
    assert image_container.dtype == second_scene_dtype

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == first_scene_shape
    assert image_container.data.dtype == first_scene_dtype

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    return image_container

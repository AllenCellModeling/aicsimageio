#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
from distributed.protocol import deserialize, serialize
from fsspec.implementations.local import LocalFileSystem
from psutil import Process
from xarray.testing import assert_equal

from aicsimageio import AICSImage, types
from aicsimageio.readers.reader import Reader

###############################################################################


def check_local_file_not_open(image_container: Union[AICSImage, Reader]) -> None:
    if isinstance(image_container, AICSImage):
        image_container = image_container.reader

    # Check that there are no open file pointers
    if isinstance(image_container._fs, LocalFileSystem):
        proc = Process()
        assert str(image_container._path) not in [f.path for f in proc.open_files()]


def check_can_serialize_image_container(
    image_container: Union[AICSImage, Reader]
) -> None:
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


def run_image_container_checks(
    image_container: Union[AICSImage, Reader],
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_current_scene: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> Union[AICSImage, Reader]:
    """
    A general suite of tests to run against image containers (Reader and AICSImage).
    """

    # Check serdes
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
    assert image_container.channel_names == expected_channel_names
    assert image_container.physical_pixel_sizes == expected_physical_pixel_sizes
    assert isinstance(image_container.metadata, expected_metadata_type)

    # Read different chunks
    zyx_chunk_from_delayed = image_container.get_image_dask_data("ZYX").compute()
    cyx_chunk_from_delayed = image_container.get_image_dask_data("CYX").compute()

    # Check image still not fully in memory
    assert image_container._xarray_data is None

    # Read in mem then pull chunks
    zyx_chunk_from_mem = image_container.get_image_data("ZYX")
    cyz_chunk_from_mem = image_container.get_image_data("CYX")

    # Compare chunk reads
    np.testing.assert_array_equal(
        zyx_chunk_from_delayed,
        zyx_chunk_from_mem,
    )
    np.testing.assert_array_equal(
        cyx_chunk_from_delayed,
        cyz_chunk_from_mem,
    )

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == expected_shape
    assert image_container.data.dtype == expected_dtype

    # Check serdes
    check_can_serialize_image_container(image_container)

    return image_container


def run_image_container_mosaic_checks(
    tiles_image_container: Reader,
    stitched_image_container: Reader,
    tiles_set_scene: str,
    stitched_set_scene: str,
) -> None:
    """
    A general suite of tests to run against readers that can stitch mosaic tiles.

    This tests uses in-memory numpy to compare. Test mosaics should be small enough to
    fit into memory.
    """
    # Set scenes
    tiles_image_container.set_scene(tiles_set_scene)
    stitched_image_container.set_scene(stitched_set_scene)

    # Get data subset
    from_tiles_stitched_data = tiles_image_container.mosaic_data
    already_stitched_data = stitched_image_container.data

    # Compare
    np.testing.assert_array_equal(from_tiles_stitched_data, already_stitched_data)


def run_image_file_checks(
    ImageContainer: Type[Union[AICSImage, Reader]],
    image: types.PathLike,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_current_scene: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> Union[AICSImage, Reader]:
    # Init container
    image_container = ImageContainer(image)

    # Check for file pointers
    check_local_file_not_open(image_container)

    # Run array and metadata check operations
    run_image_container_checks(
        image_container=image_container,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=expected_current_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )

    # Check for file pointers
    check_local_file_not_open(image_container)

    return image_container


def run_multi_scene_image_read_checks(
    ImageContainer: Type[Union[AICSImage, Reader]],
    image: types.PathLike,
    first_scene_id: str,
    first_scene_shape: Tuple[int, ...],
    first_scene_dtype: np.dtype,
    second_scene_id: str,
    second_scene_shape: Tuple[int, ...],
    second_scene_dtype: np.dtype,
) -> Union[AICSImage, Reader]:
    """
    A suite of tests to ensure that data is reset when switching scenes.
    """
    # Read file
    image_container = ImageContainer(image)

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

    # Check basics
    assert image_container.shape == second_scene_shape
    assert image_container.dtype == second_scene_dtype

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == first_scene_shape
    assert image_container.data.dtype == first_scene_dtype

    check_local_file_not_open(image_container)
    check_can_serialize_image_container(image_container)

    return image_container

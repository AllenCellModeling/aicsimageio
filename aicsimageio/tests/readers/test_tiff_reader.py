#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from distributed import Client, LocalCluster

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers import TiffReader
from aicsimageio.readers.reader import Reader

from ..conftest import LOCAL, REMOTE, get_resource_full_path, host
from ..image_container_test_utils import (
    run_image_file_checks,
    run_multi_scene_image_read_checks,
)


@host
@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
        ),
        (
            "s_1_t_1_c_1_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (10, 1736, 1776),
            np.uint16,
            "CYX",
            [f"Channel:0:{i}" for i in range(10)],
        ),
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 325, 475),
            np.uint16,
            "TCYX",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:1:0", "Channel:1:1", "Channel:1:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:2:0", "Channel:2:1", "Channel:2:2"],
        ),
        (
            "s_1_t_1_c_1_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (7548, 7548, 3),
            np.uint16,
            "YXS",  # S stands for samples dimension
            None,
        ),
        (
            # Doesn't affect this test but this is actually an OME-TIFF file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (2, 32, 32, 3),
            np.uint8,
            "CYXS",  # S stands for samples dimension
            ["Channel:0:0", "Channel:0:1"],
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_2_z_1.lif",
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:1",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:3",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_tiff_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=TiffReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(None, None, None),
        expected_metadata_type=str,
    )


@host
@pytest.mark.parametrize(
    "filename, "
    "first_scene_id, "
    "first_scene_shape, "
    "second_scene_id, "
    "second_scene_shape",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            (5, 3, 325, 475),
            "Image:1",
            (5, 3, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (5, 3, 325, 475),
            "Image:2",
            (5, 3, 325, 475),
        ),
    ],
)
def test_multi_scene_tiff_reader(
    filename: str,
    host: str,
    first_scene_id: str,
    first_scene_shape: Tuple[int, ...],
    second_scene_id: str,
    second_scene_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_multi_scene_image_read_checks(
        ImageContainer=TiffReader,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


@pytest.mark.parametrize(
    "dims_from_meta, guessed_dims, expected",
    [
        ("QZYX", "CZYX", "CZYX"),
        ("ZQYX", "CZYX", "ZCYX"),
        ("ZYXC", "CZYX", "ZYXC"),
        ("TQQYX", "TCZYX", "TCZYX"),
        ("QTQYX", "TCZYX", "CTZYX"),
        # testing that nothing happens when Q isn't present
        ("LTCYX", "DIMOK", "LTCYX"),
    ],
)
def test_merge_dim_guesses(
    dims_from_meta: str, guessed_dims: str, expected: str
) -> None:
    assert TiffReader._merge_dim_guesses(dims_from_meta, guessed_dims) == expected


def test_micromanager_ome_tiff_binary_file() -> None:
    # Construct full filepath
    uri = get_resource_full_path(
        "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos001_000.ome.tif",
        LOCAL,
    )

    # Even though the file name says it is an OME TIFF, this is
    # a binary TIFF file where the actual metadata for all scenes
    # lives in a different image file.
    # (image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif)
    # Because of this, we will read "non-main" micromanager files as just
    # normal TIFFs

    # Run image read checks on the first scene
    # (this files binary data)
    run_image_file_checks(
        ImageContainer=TiffReader,
        image=uri,
        set_scene="Image:0",
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=(50, 5, 3, 256, 256),
        expected_dtype=np.dtype(np.uint16),
        # Note this dimension order is correct but is different from OmeTiffReader
        # because we swap the dimensions into "standard" order
        expected_dims_order="TZCYX",
        expected_channel_names=["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        expected_physical_pixel_sizes=(None, None, None),
        expected_metadata_type=str,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_metadata_type",
    [
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (None, None, None),
            str,
        ),
        (
            "s_1_t_1_c_1_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 7548, 7548, 3),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (None, None, None),
            str,
        ),
    ],
)
def test_aicsimage(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_image_file_checks(
        ImageContainer=AICSImage,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "set_dims, "
    "set_channel_names, "
    "expected_dims, "
    "expected_channel_names, "
    "expected_shape",
    [
        # First check to show nothing changes
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            None,
            None,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (10, 3, 1, 325, 475),
        ),
        # Check just dims to see default channel name creation
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            "ZCYX",
            None,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (1, 3, 10, 325, 475),
        ),
        # Check setting both as simple definitions
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            "ZCYX",
            ["A", "B", "C"],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B", "C"],
            (1, 3, 10, 325, 475),
        ),
        # Check setting channels as a list of lists definitions
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            "ZCYX",
            [["A", "B", "C"]],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B", "C"],
            (1, 3, 10, 325, 475),
        ),
        # Check setting dims as list of dims
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ["ZCYX"],
            ["A", "B", "C"],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B", "C"],
            (1, 3, 10, 325, 475),
        ),
        # Check setting dims as list of None (scene has unknown dims)
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            [None],
            ["A", "B", "C"],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B", "C"],
            (10, 3, 1, 325, 475),
        ),
        # Check providing too many dims
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            "ABCDEFG",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check providing too many channels
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            None,
            ["A", "B", "C", "D"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check providing channels but no channel dim
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            "TZYX",
            ["A", "B", "C"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check number of scenes dims list matches n scenes
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ["ABC", "DEF", "GHI"],
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check number of scenes channels list matches n scenes
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            None,
            [["A", "B", "C"], ["D", "E", "F"]],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
    ],
)
def test_set_coords(
    filename: str,
    set_scene: str,
    set_dims: Optional[Union[str, List[str]]],
    set_channel_names: Optional[Union[List[str], List[List[str]]]],
    expected_dims: str,
    expected_channel_names: List[str],
    expected_shape: Tuple[int, ...],
) -> None:
    # As a reminder, AICSImage always has certain dimensions
    # If you provide a dimension that isn't one of those,
    # it will only be available on the reader, not the AICSImage object.

    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri, dim_order=set_dims, channel_names=set_channel_names)

    # Set scene
    img.set_scene(set_scene)

    # Compare AICSImage results
    assert img.dims.order == expected_dims
    assert img.channel_names == expected_channel_names
    assert img.shape == expected_shape


@pytest.mark.parametrize(
    "filename, expected_reader",
    [
        (
            "s_1_t_1_c_1_z_1.tiff",
            TiffReader,
        ),
    ],
)
def test_selected_tiff_reader(
    filename: str,
    expected_reader: Type[Reader],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri)

    # Assert basics
    assert isinstance(img.reader, expected_reader)


@pytest.mark.parametrize(
    "filename, set_reader, extra_kwargs, expected_dims, expected_shape",
    [
        (
            "actk.ome.tiff",
            TiffReader,
            {},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (1, 6, 65, 233, 345),
        ),
        # Test good reader but also allows extra kwargs
        (
            "actk.ome.tiff",
            TiffReader,
            {"dim_order": "CTYX"},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (65, 6, 1, 233, 345),
        ),
    ],
)
def test_set_reader(
    filename: str,
    set_reader: Type[Reader],
    extra_kwargs: Dict[str, Any],
    expected_dims: str,
    expected_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri, reader=set_reader, **extra_kwargs)

    # Assert basics
    assert img.dims.order == expected_dims
    assert img.shape == expected_shape


@pytest.mark.parametrize(
    "filename, set_scene, get_dims, get_specific_dims, expected_shape",
    [
        (
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            "CYXS",
            {},
            (2, 32, 32, 3),
        ),
    ],
)
@pytest.mark.parametrize("chunk_dims", ["YX", "ZYX"])
@pytest.mark.parametrize("processes", [True, False])
def test_parallel_read(
    filename: str,
    set_scene: str,
    chunk_dims: str,
    processes: bool,
    get_dims: str,
    get_specific_dims: Dict[str, Union[int, slice, range, Tuple[int, ...], List[int]]],
    expected_shape: Tuple[int, ...],
) -> None:
    """
    This test ensures that our produced dask array can be read in parallel.
    """
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init image
    img = AICSImage(uri, chunk_dims=chunk_dims)
    img.set_scene(set_scene)

    # Init cluster
    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    # Select data
    out = img.get_image_dask_data(get_dims, **get_specific_dims).compute()
    assert out.shape == expected_shape

    # Shutdown and then safety measure timeout
    cluster.close()
    client.close()
    time.sleep(5)


@pytest.mark.parametrize(
    "filename, "
    "host, "
    "first_scene, "
    "expected_first_chunk_shape, "
    "second_scene, "
    "expected_second_chunk_shape",
    [
        (
            "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
            LOCAL,
            0,
            (50, 5, 256, 256),
            1,
            (50, 5, 256, 256),
        ),
        (
            "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
            LOCAL,
            1,
            (50, 5, 256, 256),
            0,
            (50, 5, 256, 256),
        ),
        pytest.param(
            "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
            REMOTE,
            1,  # Start with second scene to trigger error faster
            (50, 5, 256, 256),
            0,
            (50, 5, 256, 256),
            marks=pytest.mark.raises(exceptions=IndexError),
        ),
    ],
)
@pytest.mark.parametrize("processes", [True, False])
def test_parallel_multifile_tiff_read(
    filename: str,
    host: str,
    first_scene: int,
    expected_first_chunk_shape: Tuple[int, ...],
    second_scene: int,
    expected_second_chunk_shape: Tuple[int, ...],
    processes: bool,
) -> None:
    """
    This test ensures that we can serialize and read 'multi-file multi-scene' formats.
    See: https://github.com/AllenCellModeling/aicsimageio/issues/196

    We specifically test with a Distributed cluster to ensure that we serialize and
    read properly from each file.
    """
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Init image
    img = AICSImage(uri)

    # Init cluster
    cluster = LocalCluster(processes=processes)
    client = Client(cluster)

    # Select data
    img.set_scene(first_scene)
    first_out = img.get_image_dask_data("TZYX").compute()
    assert first_out.shape == expected_first_chunk_shape

    # Update scene and select data
    img.set_scene(second_scene)
    second_out = img.get_image_dask_data("TZYX").compute()
    assert second_out.shape == expected_second_chunk_shape

    # Shutdown and then safety measure timeout
    cluster.close()
    client.close()
    time.sleep(5)


@pytest.mark.parametrize(
    "filename, select_scenes",
    [
        ("s_1_t_10_c_3_z_1.tiff", None),
    ],
)
def test_roundtrip_save_all_scenes(
    filename: str, select_scenes: Optional[List[str]]
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Read initial
    original = AICSImage(uri)

    # Save to temp and compare
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / f"converted-{filename}.ome.tiff"
        original.save(save_path, select_scenes=select_scenes)

        # Re-read
        result = AICSImage(save_path)

        # Compare all scenes
        # They may not have the same scene ids as some readers use scene names as the
        # id, see LifReader for example
        if select_scenes is None:
            select_original_scenes = list(original.scenes)
        else:
            select_original_scenes = select_scenes

        assert len(select_original_scenes) == len(result.scenes)
        for original_scene_id, result_scene_id in zip(
            select_original_scenes, result.scenes
        ):
            # Compare
            original.set_scene(original_scene_id)
            result.set_scene(result_scene_id)

            np.testing.assert_array_equal(original.data, result.data)
            assert original.dims.order == result.dims.order
            assert original.channel_names == result.channel_names


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        ("s_1_t_10_c_3_z_1.tiff", (10, 3, 1, 325, 475)),
    ],
)
def test_no_scene_prop_access(
    filename: str,
    expected_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct image and check no scene call with property access
    img = AICSImage(uri)
    assert img.shape == expected_shape

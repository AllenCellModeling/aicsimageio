#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from ome_types import OME

from aicsimageio import AICSImage, dimensions, exceptions, readers, types

from .conftest import LOCAL, get_resource_full_path
from .image_container_test_utils import (
    run_image_container_checks,
    run_image_file_checks,
    run_multi_scene_image_read_checks,
)

###############################################################################

# We only run checks on a subset of local files
# The base reader unit tests show that we can handle both local or remote
# If we handled them here, tests would just take longer than they already do

###############################################################################


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
        #######################################################################
        # DefaultReader
        (
            "example.png",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 800, 537, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        (
            "example.gif",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 268, 268, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        (
            "example_valid_frame_count.mp4",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 272, 272, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        #######################################################################
        # TiffReader
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (1.0, 1.0, 1.0),
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
            (1.0, 1.0, 1.0),
            str,
        ),
        #######################################################################
        # OmeTiffReader
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 10, 1, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [f"C:{i}" for i in range(10)],  # This is the actual metadata
            (1.0, 1.0, 1.0),
            OME,
        ),
        (
            # This is actually an OME-TIFF file
            # Shows we don't just work off of extensions
            # But the content of the file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (1, 2, 1, 32, 32, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0", "Channel:0:1"],
            (1.0, 1.0, 1.0),
            OME,
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
            OME,
        ),
        (
            "actk.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 6, 65, 233, 345),
            np.float64,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "nucleus_segmentation",
                "membrane_segmentation",
                "dna",
                "membrane",
                "structure",
                "brightfield",
            ],
            (0.29, 0.29, 0.29),
            OME,
        ),
        #######################################################################
        # LifReader
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            ("b2_001_Crop001_Resize001",),
            (4, 2, 1, 614, 614),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"],
            (1.0, 2.9485556406398508, 2.9485556406398508),
            ET.Element,
        ),
        (
            "tiled.lif",
            "TileScan_002",
            ("TileScan_002",),
            (1, 4, 1, 5622, 7666),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray", "Red", "Green", "Cyan"],
            (1.0, 4.984719055966396, 4.984719055966396),
            ET.Element,
        ),
        #######################################################################
        # Errors
        pytest.param(
            "example.txt",
            None,
            None,
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
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
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
    "first_scene_id, "
    "first_scene_shape, "
    "second_scene_id, "
    "second_scene_shape",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            (1, 3, 5, 325, 475),
            "Image:1",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (1, 3, 5, 325, 475),
            "Image:2",
            (1, 3, 5, 325, 475),
        ),
    ],
)
def test_multi_scene_aicsimage(
    filename: str,
    first_scene_id: str,
    first_scene_shape: Tuple[int, ...],
    second_scene_id: str,
    second_scene_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_multi_scene_image_read_checks(
        ImageContainer=AICSImage,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


@pytest.mark.parametrize(
    "image, "
    "dim_order, "
    "channel_names, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dims, "
    "expected_channel_names",
    [
        # Check no metadata provided 2D
        # these are really simple just None checks
        (
            np.random.rand(1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Check no metadata provided 4D
        # these check that channel names are created for all
        # and specifically for xr that channel names are overwritten
        (
            np.random.rand(1, 1, 1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Test many scene, same dim_order, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1"],
        ),
        # Test many scene, different dim_order, different channel_names, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        # Test filled in xarray(s)
        # no metadata should be overwritten
        (
            xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(
                np.random.rand(1, 1, 1), dims=list("CYX"), coords={"C": ["A"]}
            ),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        # Test non-standard dimensions
        (
            np.random.rand(2, 2, 2),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(2, 2, 2)),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Test that we can support many dimensions if dims is provided
        (
            np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0",),
            (1, 3, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((1, 2, 3, 4, 5, 6, 7, 8)),
            ],
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 3, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((2, 3, 4, 5, 6, 7, 8, 9)),
            ],
            "ABCDEFGH",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 4, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1", "Channel:1:2", "Channel:1:3"],
        ),
        # Test that without dims and with more than five dims, it raises an error
        # Our guess dim order only support up to five dims
        pytest.param(
            np.random.rand(1, 2, 3, 4, 5, 6),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            "hello world",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_aicsimage_from_array(
    image: Union[types.MetaArrayLike, List[types.MetaArrayLike]],
    dim_order: Optional[str],
    channel_names: Optional[List[str]],
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dims: str,
    expected_channel_names: List[str],
) -> None:
    # Init
    image_container = AICSImage(image, dim_order=dim_order, channel_names=channel_names)

    run_image_container_checks(
        image_container=image_container,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.float64),
        expected_dims_order=expected_dims,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
        # we allow both None and Dict because the user can pass an already initialized
        # xarray DataArray which has metadata as a dict
        expected_metadata_type=(
            type(None),
            dict,
        ),
    )


@pytest.mark.parametrize(
    "filename, select_scenes",
    [
        ("s_1_t_10_c_3_z_1.tiff", None),
        ("s_3_t_1_c_3_z_5.ome.tiff", None),
        ("s_3_t_1_c_3_z_5.ome.tiff", ["Image:2", "Image:1"]),
        ("s_1_t_4_c_2_z_1.lif", None),
        ("tiled.lif", None),
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
    "filename, "
    "set_scene, "
    "set_dims, "
    "set_channel_names, "
    "expected_dims, "
    "expected_channel_names, "
    "expected_shape",
    [
        # DefaultReader
        # First check to show nothing changes
        (
            "example.gif",
            "Image:0",
            None,
            None,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (72, 1, 1, 268, 268, 4),
        ),
        # Check just dims to see default channel name creation
        (
            "example.gif",
            "Image:0",
            "ZYXC",
            None,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3"],
            (1, 4, 72, 268, 268),
        ),
        # Check setting both as simple definitions
        (
            "example.gif",
            "Image:0",
            "ZYXC",
            ["Red", "Green", "Blue", "Alpha"],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Red", "Green", "Blue", "Alpha"],
            (1, 4, 72, 268, 268),
        ),
        # Check providing too many dims
        pytest.param(
            "example.gif",
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
            "example.gif",
            "Image:0",
            "ZYXC",
            ["A", "B", "C"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check providing channels but no channel dim
        pytest.param(
            "example.gif",
            "Image:0",
            None,
            ["A", "B", "C"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        ######################################
        # TiffReader
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
    "filename, set_reader, extra_kwargs, expected_dims, expected_shape",
    [
        (
            "actk.ome.tiff",
            readers.TiffReader,
            {},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (1, 6, 65, 233, 345),
        ),
        # See shape to see why you should use TiffReader :)
        (
            "actk.ome.tiff",
            readers.default_reader.DefaultReader,
            {},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (390, 1, 1, 233, 345),
        ),
        # Test good reader but also allows extra kwargs
        (
            "actk.ome.tiff",
            readers.TiffReader,
            {"dim_order": "CTYX"},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (65, 6, 1, 233, 345),
        ),
        # Test incompatible reader
        pytest.param(
            "actk.ome.tiff",
            readers.lif_reader.LifReader,
            {},
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_set_reader(
    filename: str,
    set_reader: types.ReaderType,
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

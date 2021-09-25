#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from distributed import Client, LocalCluster
from ome_types import OME

from aicsimageio import AICSImage, dimensions, exceptions, readers, types

from .conftest import LOCAL, REMOTE, get_resource_full_path
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
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
            (None, 0.33914910277324634, 0.33914910277324634),
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
            (None, 0.20061311154598827, 0.20061311154598827),
            ET.Element,
        ),
        #######################################################################
        # CziReader
        (
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
            ET.Element,
        ),
        (
            "RGB-8bit.czi",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 624, 924, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (None, 1.0833333333333333, 1.0833333333333333),
            ET.Element,
        ),
        #######################################################################
        # BioformatsReader
        (
            "ND2_jonas_header_test2.nd2",
            "Image:0",
            ("Image:0",),
            (4, 1, 5, 520, 696),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
            dict,
        ),
        (
            "DV_siRNAi-HeLa_IN_02.r3d_D3D.dv",
            "DV_siRNAi-HeLa_IN_02.r3d_D3D.dv",
            ("DV_siRNAi-HeLa_IN_02.r3d_D3D.dv",),
            (1, 4, 40, 512, 512),
            np.int16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3"],
            (0.20000000298023224, 0.06502940505743027, 0.06502940505743027),
            OME,
        ),
        (
            "Olympus-OIR_etienne_amy_slice_z_stack_0001.oir",
            "Olympus-OIR_etienne_amy_slice_z_stack_0001.oir",
            ("Olympus-OIR_etienne_amy_slice_z_stack_0001.oir",),
            (32, 1, 1, 512, 512),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["CH3"],
            (1.0, 1.242961138804478, 1.242961138804478),
            OME,
        ),
        (
            "Imaris-IMS_davemason_Convallaria_3C_1T_confocal.ims",
            "Imaris-IMS_davemason_Convallaria_3C_1T_confocal.ims Resolution Level 1",
            ("Imaris-IMS_davemason_Convallaria_3C_1T_confocal.ims Resolution Level 1",),
            (1, 3, 1, 1024, 1024),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
            (0.001, 1.2059374999999999, 1.2059570312500014),
            OME,
        ),
        (
            "DICOM_samples_MR-MONO2-8-16x-heart.dcm",
            "Series 0",
            ("Series 0",),
            (1, 1, 16, 256, 256),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
            (None, None, None),
            OME,
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
    "filename, expected_shape",
    [
        ("example.png", (1, 1, 1, 800, 537, 4)),
        ("s_1_t_10_c_3_z_1.tiff", (10, 3, 1, 325, 475)),
        ("s_1_t_1_c_10_z_1.ome.tiff", (1, 10, 1, 1736, 1776)),
        ("s_1_t_4_c_2_z_1.lif", (4, 2, 1, 614, 614)),
        ("RGB-8bit.czi", (1, 1, 1, 624, 924, 3)),
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
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            (1, 3, 5, 325, 475),
            "P3",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P3",
            (1, 3, 5, 325, 475),
            "P1",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            1,
            (1, 3, 5, 325, 475),
            2,
            (1, 3, 5, 325, 475),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.czi",
            ["this is not a scene id"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=TypeError),
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
        expected_physical_pixel_sizes=(None, None, None),
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
        ("s_1_t_1_c_1_z_1.czi", None),
        ("s_3_t_1_c_3_z_5.czi", ["P2", "P1"]),
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
    set_reader: Type[readers.reader.Reader],
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
    "filename, "
    "reconstruct_mosaic, "
    "set_scene, "
    "expected_shape, "
    "expected_mosaic_tile_dims, "
    "specific_tile_index",
    [
        (
            "tiled.lif",
            True,
            "TileScan_002",
            (1, 4, 1, 5622, 7666),
            (512, 512),
            0,
        ),
        (
            "tiled.lif",
            False,
            "TileScan_002",
            (165, 1, 4, 1, 512, 512),
            (512, 512),
            0,
        ),
        (
            "OverViewScan.czi",
            True,
            "TR1",
            (1, 1, 1, 3212, 7398),
            (440, 544),
            0,
        ),
        (
            "OverViewScan.czi",
            False,
            "TR1",
            (120, 1, 1, 1, 440, 544),
            (440, 544),
            0,
        ),
        pytest.param(
            "tiled.lif",
            False,
            "TileScan_002",
            (165, 1, 4, 1, 512, 512),
            (512, 512),
            999,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "actk.ome.tiff",
            True,
            "Image:0",
            (1, 6, 65, 233, 345),
            None,
            0,
            # AttributeError raises not because of error in rollback
            # but because cannot access Y or X from
            # None return from `mosaic_tile_dims` because
            # image is not a mosaic tiled image
            marks=pytest.mark.raises(exception=AttributeError),
        ),
    ],
)
def test_mosaic_passthrough(
    filename: str,
    reconstruct_mosaic: bool,
    set_scene: str,
    expected_shape: Tuple[int, ...],
    expected_mosaic_tile_dims: Tuple[int, int],
    specific_tile_index: int,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri, reconstruct_mosaic=reconstruct_mosaic)
    img.set_scene(set_scene)

    # Assert basics
    assert img.shape == expected_shape
    assert img.mosaic_tile_dims.Y == expected_mosaic_tile_dims[0]  # type: ignore
    assert img.mosaic_tile_dims.X == expected_mosaic_tile_dims[1]  # type: ignore

    # Ensure that regardless of stitched or not, we can get tile position
    img.get_mosaic_tile_position(specific_tile_index)


@pytest.mark.parametrize(
    "filename, set_scene, get_dims, get_specific_dims, expected_shape",
    [
        # Check normal volumetric data
        (
            "3d-cell-viewer.ome.tiff",
            "Image:0",
            "CZYX",
            {"C": [0, 1, 2, 3]},
            (4, 74, 1024, 1024),
        ),
        (
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            "CYXS",
            {},
            (2, 32, 32, 3),
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            "TYX",
            {},
            (4, 614, 614),
        ),
        # Check mosaic chunk handling
        (
            "tiled.lif",
            "TileScan_002",
            "CYX",
            {"Y": slice(0, 2000), "X": slice(0, 2000)},
            (4, 2000, 2000),
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

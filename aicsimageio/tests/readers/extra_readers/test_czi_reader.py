#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest
from ome_types import OME

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers import ArrayLikeReader
from aicsimageio.readers.czi_reader import CziReader

from ...conftest import LOCAL, REMOTE, get_resource_full_path
from ...image_container_test_utils import (
    run_image_container_mosaic_checks,
    run_image_file_checks,
    run_multi_scene_image_read_checks,
)


@pytest.mark.parametrize(
    ["filename", "num_subblocks", "acquistion_time"],
    [
        ("s_1_t_1_c_1_z_1.czi", 1, "2019-06-27T18:33:41.1154211Z"),
        ("s_3_t_1_c_3_z_5.czi", 45, "2019-06-27T18:39:26.6459707Z"),
        (
            "variable_scene_shape_first_scene_pyramid.czi",
            27,
            "2019-05-09T09:49:17.9414649Z",
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_subblocks(filename: str, num_subblocks: int, acquistion_time: str) -> None:
    reader = CziReader(
        get_resource_full_path(filename, LOCAL),
        include_subblock_metadata=True,
    )

    subblocks = reader.metadata.findall("./Subblocks/Subblock")

    assert len(subblocks) == num_subblocks
    # Ensure one of the elements in the first subblock has expected data
    assert subblocks[0].find(".//AcquisitionTime").text == acquistion_time


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "S=2_4x2_T=2=Z=3_CH=2.czi",
            "TR1",
            ("TR1", "TR2"),
            (1, 2, 2, 3, 8, 256, 256),
            np.uint16,
            "HTCZMYX",
            ["DAPI", "EGFP"],
            (1.0, 0.4, 0.4),
        ),
        (
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            ("Image:0",),
            (1, 325, 475),
            np.uint16,
            "CYX",
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P3",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P1",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "RGB-8bit.czi",
            "Image:0",
            ("Image:0",),
            (1, 624, 924, 3),
            np.uint8,
            "TYXS",
            None,
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        pytest.param(
            "variable_scene_shape_first_scene_pyramid.czi",
            "A1",
            ("A1", "A2"),
            (3, 9, 2208, 2752),
            np.uint16,
            "CMYX",
            [
                "EGFP",
                "mCher",
                "PGC",
            ],
            (None, 9.082107048835329, 9.082107048835329),
            marks=pytest.mark.xfail(reason="Missing scenes"),
        ),
        pytest.param(
            "example.txt",
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_czi_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_image_file_checks(
        ImageContainer=CziReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=ET.Element,
    )


@pytest.mark.xfail(reason="Do no support remote CZI reading yet")
def test_czi_reader_remote_xfail() -> None:
    # Construct full filepath
    uri = get_resource_full_path("s_1_t_1_c_1_z_1.czi", REMOTE)
    CziReader(uri)


@pytest.mark.parametrize(
    "tiles_filename, stitched_filename, tiles_set_scene, stitched_set_scene",
    [
        (
            "OverViewScan.czi",
            "OverView.npy",
            "TR1",
            "Image:0",
        )
    ],
)
def test_czi_reader_mosaic_stitching(
    tiles_filename: str,
    stitched_filename: str,
    tiles_set_scene: str,
    stitched_set_scene: str,
) -> None:
    # Construct full filepath
    tiles_uri = get_resource_full_path(tiles_filename, LOCAL)
    stitched_uri = get_resource_full_path(stitched_filename, LOCAL)

    # Construct reader
    tiles_reader = CziReader(tiles_uri)
    stitched_np = np.load(stitched_uri)
    stitched_reader = ArrayLikeReader(stitched_np)

    # Run checks
    run_image_container_mosaic_checks(
        tiles_image_container=tiles_reader,
        stitched_image_container=stitched_reader,
        tiles_set_scene=tiles_set_scene,
        stitched_set_scene=stitched_set_scene,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_tile_dims, "
    "select_tile_index, "
    "expected_tile_top_left",
    [
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            0,
            (0, 0),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            50,
            (1188, 4406),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            3,
            (0, 1469),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            119,
            (2772, 0),
        ),
        pytest.param(
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            999,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            None,
            None,
            None,
            # The value returned from all of the calls is None
            # Which when trying to operate on will raise an AttributeError
            # Because None doesn't have a Y or X attribute for example
            marks=pytest.mark.raises(exception=AttributeError),
        ),
    ],
)
def test_czi_reader_mosaic_tile_inspection(
    filename: str,
    set_scene: str,
    expected_tile_dims: Tuple[int, int],
    select_tile_index: int,
    expected_tile_top_left: Tuple[int, int],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = CziReader(uri)
    reader.set_scene(set_scene)

    # Check basics
    assert reader.mosaic_tile_dims.Y == expected_tile_dims[0]  # type: ignore
    assert reader.mosaic_tile_dims.X == expected_tile_dims[1]  # type: ignore

    # Pull tile info for compare
    tile_y_pos, tile_x_pos = reader.get_mosaic_tile_position(select_tile_index)
    assert tile_y_pos == expected_tile_top_left[0]
    assert tile_x_pos == expected_tile_top_left[1]

    # Pull actual pixel data to compare
    tile_from_m_index = reader.get_image_dask_data(
        reader.dims.order.replace(dimensions.DimensionNames.MosaicTile, ""),
        M=select_tile_index,
    ).compute()

    # Position ops construction to pull using array slicing
    position_ops = []
    for dim in reader.dims.order:
        if dim not in [
            dimensions.DimensionNames.MosaicTile,
            dimensions.DimensionNames.SpatialY,
            dimensions.DimensionNames.SpatialX,
        ]:
            position_ops.append(slice(None))
        if dim is dimensions.DimensionNames.SpatialY:
            position_ops.append(
                slice(
                    tile_y_pos,
                    tile_y_pos + reader.mosaic_tile_dims.Y,  # type: ignore
                )
            )
        if dim is dimensions.DimensionNames.SpatialX:
            position_ops.append(
                slice(
                    tile_x_pos,
                    tile_x_pos + reader.mosaic_tile_dims.X,  # type: ignore
                )
            )

    tile_from_position = reader.mosaic_dask_data[tuple(position_ops)].compute()

    # Assert all close
    # CZI tiles have about 20% overlap it looks
    # Relative tolerance of 300 is enough to pass
    np.testing.assert_allclose(tile_from_m_index, tile_from_position, rtol=300)


@pytest.mark.parametrize(
    "filename, "
    "expected_tile_y_coords, "
    "expected_tile_x_coords, "
    "expected_mosaic_y_coords, "
    "expected_mosaic_x_coords",
    [
        (
            "OverViewScan.czi",
            np.arange(0, 2012.719549253996, 4.5743626119409),
            np.arange(0, 2488.45326089585, 4.5743626119409),
            np.arange(0, 14692.852709554172, 4.5743626119409),
            np.arange(0, 33836.560240526844, 4.5743626119409),
        ),
    ],
)
def test_czi_reader_mosaic_coords(
    filename: str,
    expected_tile_y_coords: np.ndarray,
    expected_tile_x_coords: np.ndarray,
    expected_mosaic_y_coords: np.ndarray,
    expected_mosaic_x_coords: np.ndarray,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = CziReader(uri)

    # Check tile y and x min max
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_tile_y_coords,
    )
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_tile_x_coords,
    )

    # Check mosaic y and x min max
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_mosaic_y_coords,
    )
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_mosaic_x_coords,
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
    "filename, expected_shape",
    [
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
    "filename, select_scenes",
    [
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
    "reconstruct_mosaic, "
    "set_scene, "
    "expected_shape, "
    "expected_mosaic_tile_dims, "
    "specific_tile_index",
    [
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
    "filename",
    [
        "s_1_t_1_c_1_z_1.czi",
        "RGB-8bit.czi",
    ],
)
def test_ome_metadata(filename: str) -> None:
    # Get full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init image
    img = AICSImage(uri)

    # Test the transform
    assert isinstance(img.ome_metadata, OME)

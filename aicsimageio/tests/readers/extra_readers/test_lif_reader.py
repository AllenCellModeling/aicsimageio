#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from distributed import Client, LocalCluster
from readlif.reader import LifFile

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers.lif_reader import LifReader

from ...conftest import LOCAL, get_resource_full_path, host
from ...image_container_test_utils import (
    run_image_container_mosaic_checks,
    run_image_file_checks,
)


@host
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
            "s_1_t_1_c_2_z_1.lif",
            "PEI_laminin_35k",
            ("PEI_laminin_35k",),
            (1, 2, 1, 2048, 2048),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"],
            (None, 0.32499999999999996, 0.32499999999999996),
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            ("b2_001_Crop001_Resize001",),
            (4, 2, 1, 614, 614),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"],
            (None, 0.33914910277324634, 0.33914910277324634),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            ("TileScan_002",),
            (165, 1, 4, 1, 512, 512),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray", "Red", "Green", "Cyan"],
            (None, 0.20061311154598827, 0.20061311154598827),
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
            "3d-cell-viewer.ome.tiff",
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
def test_lif_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=LifReader,
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


@pytest.mark.parametrize("filename", ["s_1_t_1_c_2_z_1.lif", "s_1_t_4_c_2_z_1.lif"])
@pytest.mark.parametrize("chunk_dims", ["ZYX", "TYX", "CYX"])
@pytest.mark.parametrize("get_dims", ["ZYX", "TYX"])
def test_sanity_check_correct_indexing(
    filename: str,
    chunk_dims: str,
    get_dims: str,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = LifReader(uri, chunk_dims=chunk_dims)
    lif_img = LifFile(uri).get_image(0)

    # Pull a chunk from LifReader
    chunk_from_lif_reader = reader.get_image_dask_data(get_dims).compute()

    # Pull what should be the same chunk from LifImage
    planes = []
    reshape_values = []
    for dim in get_dims:
        dim_size = getattr(lif_img.info["dims"], dim.lower())
        reshape_values.append(dim_size)

        if dim not in ["Y", "X"]:
            for i in range(dim_size):
                planes.append(np.asarray(lif_img.get_frame(**{dim.lower(): i})))

    # Stack and reshape
    chunk_from_read_lif = np.stack(planes).reshape(tuple(reshape_values))

    # Compare
    np.testing.assert_array_equal(chunk_from_lif_reader, chunk_from_read_lif)


@pytest.mark.parametrize(
    "tiles_filename, " "stitched_filename, " "tiles_set_scene, " "stitched_set_scene, ",
    [
        (
            "tiled.lif",
            "merged-tiles.lif",
            "TileScan_002",
            "TileScan_002_Merging",
        ),
        # s_1_t_4_c_2_z_1.lif has no mosaic tiles
        pytest.param(
            "s_1_t_4_c_2_z_1.lif",
            "merged-tiles.lif",
            "b2_001_Crop001_Resize001",
            "TileScan_002_Merging",
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
def test_lif_reader_mosaic_stitching(
    tiles_filename: str,
    stitched_filename: str,
    tiles_set_scene: str,
    stitched_set_scene: str,
) -> None:
    # Construct full filepath
    tiles_uri = get_resource_full_path(tiles_filename, LOCAL)
    stitched_uri = get_resource_full_path(stitched_filename, LOCAL)

    # Construct reader
    tiles_reader = LifReader(tiles_uri)
    stitched_reader = LifReader(stitched_uri)

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
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            0,
            (5110, 7154),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            50,
            (3577, 4599),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            3,
            (5110, 5621),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            164,
            (0, 0),
        ),
        pytest.param(
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            999,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "merged-tiles.lif",
            "TileScan_002_Merging",
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
def test_lif_reader_mosaic_tile_inspection(
    filename: str,
    set_scene: str,
    expected_tile_dims: Tuple[int, int],
    select_tile_index: int,
    expected_tile_top_left: Tuple[int, int],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = LifReader(uri)
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
    tile_from_position = reader.mosaic_xarray_dask_data[
        :,
        :,
        :,
        tile_y_pos : (tile_y_pos + reader.mosaic_tile_dims.Y),  # type: ignore
        tile_x_pos : (tile_x_pos + reader.mosaic_tile_dims.X),  # type: ignore
    ].compute()

    # Remove the first Y and X pixels
    # The stitched tiles overlap each other by 1px each so this is just
    # ignoring what would be overlap / cleaned up
    tile_from_m_index = tile_from_m_index[:, :, :, 1:, 1:]
    tile_from_position = tile_from_position[:, :, :, 1:, 1:]

    # Assert equal
    np.testing.assert_array_equal(tile_from_m_index, tile_from_position)


@pytest.mark.parametrize(
    "filename, "
    "expected_tile_y_coords, "
    "expected_tile_x_coords, "
    "expected_mosaic_y_coords, "
    "expected_mosaic_x_coords",
    [
        (
            "tiled.lif",
            np.arange(0, 102.71391311154599, 0.20061311154598827),
            np.arange(0, 102.71391311154599, 0.20061311154598827),
            np.arange(0, 1127.846913111546, 0.20061311154598827),
            np.arange(0, 1537.900113111546, 0.20061311154598827),
        ),
    ],
)
def test_lif_reader_mosaic_coords(
    filename: str,
    expected_tile_y_coords: np.ndarray,
    expected_tile_x_coords: np.ndarray,
    expected_mosaic_y_coords: np.ndarray,
    expected_mosaic_x_coords: np.ndarray,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = LifReader(uri)

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
    "filename, select_scenes",
    [
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
        pytest.param(
            "tiled.lif",
            False,
            "TileScan_002",
            (165, 1, 4, 1, 512, 512),
            (512, 512),
            999,
            marks=pytest.mark.raises(exception=IndexError),
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
    "filename, expected_shape",
    [
        ("s_1_t_4_c_2_z_1.lif", (4, 2, 1, 614, 614)),
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

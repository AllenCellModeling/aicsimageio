#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
from urllib.error import HTTPError

import numpy as np
import pytest
from ome_types import OME

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import OmeTiffReader

from ..conftest import LOCAL, get_resource_full_path, host
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
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 10, 1, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [f"C:{i}" for i in range(10)],  # This is the actual metadata
            (None, None, None),
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
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
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
            "s_1_t_1_c_2_z_1.lif",
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
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_ome_tiff_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=OmeTiffReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


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
            "pre-variance-cfe.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 9, 65, 600, 900),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "Bright_2",
                "EGFP",
                "CMDRP",
                "H3342",
                "SEG_STRUCT",
                "SEG_Memb",
                "SEG_DNA",
                "CON_Memb",
                "CON_DNA",
            ],
            (0.29, 0.10833333333333334, 0.10833333333333334),
        ),
        (
            "variance-cfe.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 9, 65, 600, 900),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "CMDRP",
                "EGFP",
                "H3342",
                "Bright_2",
                "SEG_STRUCT",
                "SEG_Memb",
                "SEG_DNA",
                "CON_Memb",
                "CON_DNA",
            ],
            (0.29, 0.10833333333333332, 0.10833333333333332),
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
        ),
    ],
)
def test_ome_tiff_reader_large_files(
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
        ImageContainer=OmeTiffReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
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
def test_multi_scene_ome_tiff_reader(
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
        ImageContainer=OmeTiffReader,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


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
        # TODO:
        # Select a different level besides level 0
        # TiffReader / OmeTiffReader defaults to reading level 0
        (
            "variable_scene_shape_first_scene_pyramid.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 3, 1, 6184, 7712),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "mCher", "PGC"],
            (None, 0.9082107048835328, 0.9082107048835328),
        ),
        (
            "variable_scene_shape_first_scene_pyramid.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 1, 1, 2030, 422),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0"],
            (None, 0.9082107048835328, 0.9082107048835328),
        ),
    ],
)
def test_multi_resolution_ome_tiff_reader(
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
        ImageContainer=OmeTiffReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


@host
@pytest.mark.parametrize(
    "filename",
    [
        # Pipline 4 is valid, :tada:
        "pipeline-4.ome.tiff",
        # Some of our test files are valid, :tada:
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        # A lot of our files aren't valid, :upside-down-smiley:
        # These files have invalid schema / layout
        pytest.param(
            "3d-cell-viewer.ome.tiff",
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "pre-variance-cfe.ome.tiff",
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "variance-cfe.ome.tiff",
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "actk.ome.tiff",
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        # This file has a namespace that doesn't exist
        pytest.param(
            "s_1_t_1_c_10_z_1.ome.tiff", marks=pytest.mark.raises(exception=HTTPError)
        ),
    ],
)
def test_known_errors_without_cleaning(filename: str, host: str) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    OmeTiffReader(uri, clean_metadata=False)


def test_micromanager_ome_tiff_main_file() -> None:
    # Construct full filepath
    uri = get_resource_full_path(
        "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
        LOCAL,
    )

    # MicroManager will split up multi-scene image sets into multiple files
    # tifffile will then read in all of the scenes at once when it detects
    # the file is a micromanager file set
    # resulting in this single file truly only containing the binary for a
    # single scene but containing the metadata for all files in the set
    # and, while this file only contains the binary for itself, tifffile will
    # read the image data for the linked files

    # Run image read checks on the first scene
    # (this files binary data)
    run_image_file_checks(
        ImageContainer=OmeTiffReader,
        image=uri,
        set_scene="Image:0",
        expected_scenes=("Image:0", "Image:1"),
        expected_current_scene="Image:0",
        expected_shape=(50, 3, 5, 256, 256),
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER,
        expected_channel_names=["Cy5", "DAPI", "FITC"],
        expected_physical_pixel_sizes=(1.75, 2.0, 2.0),
        expected_metadata_type=OME,
    )

    # TODO:
    # The user shouldn't do this because it can raise a "Seek on closed file" error
    # Long term solution is something like:
    # https://github.com/AllenCellModeling/aicsimageio/issues/196
    # or more generally "support many file OME-TIFFs"
    #
    # Run image read checks on the second scene
    # (a different files binary data)
    # (image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos001_000.ome.tif)
    # run_image_read_checks(
    #     ImageContainer=OmeTiffReader,
    #     uri=uri,
    #     set_scene="Image:1",
    #     expected_scenes=("Image:0", "Image:1"),
    #     expected_current_scene="Image:1",
    #     expected_shape=(50, 3, 5, 256, 256),
    #     expected_dtype=np.uint16,
    #     expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER,
    #     expected_channel_names=["Cy5", "DAPI", "FITC"],
    #     expected_physical_pixel_sizes=(1.75, 2.0, 2.0),
    # )

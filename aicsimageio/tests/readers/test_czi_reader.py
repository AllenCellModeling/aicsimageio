#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import ArrayLikeReader
from aicsimageio.readers.czi_reader import CziReader

from ..conftest import LOCAL, get_resource_full_path
from ..image_container_test_utils import (
    run_image_container_mosaic_checks,
    run_image_file_checks,
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
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            ("Image:0",),
            (1, 325, 475),
            np.uint16,
            "CYX",
            ["Channel:0--Bright--Brightfield"],
            (1.0, 1.0833333333333333e-06, 1.0833333333333333e-06),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "Channel:0--EGFP--Fluorescence",
                "Channel:1--TaRFP--Fluorescence",
                "Channel:2--Bright--Brightfield",
            ],
            (1e-06, 1.0833333333333333e-06, 1.0833333333333333e-06),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P3",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "Channel:0--EGFP--Fluorescence",
                "Channel:1--TaRFP--Fluorescence",
                "Channel:2--Bright--Brightfield",
            ],
            (1e-06, 1.0833333333333333e-06, 1.0833333333333333e-06),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P1",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "Channel:0--EGFP--Fluorescence",
                "Channel:1--TaRFP--Fluorescence",
                "Channel:2--Bright--Brightfield",
            ],
            (1e-06, 1.0833333333333333e-06, 1.0833333333333333e-06),
        ),
        (
            "variable_scene_shape_first_scene_pyramid.czi",
            "A1",
            ("A1",),
            (9, 3, 2208, 2752),
            np.uint16,
            "MCYX",
            [
                "Channel:0--EGFP--Fluorescence",
                "Channel:1--mCher--Fluorescence",
                "Channel:2--PGC--Phase",
            ],
            (1.0, 9.082107048835329e-07, 9.082107048835329e-07),
        ),
        (
            "RGB-8bit.czi",
            "Image:0",
            ("Image:0",),
            (1, 624, 924, 3),
            np.uint8,
            "TYXS",
            None,
            (1.0, 1.0833333333333333e-06, 1.0833333333333333e-06),
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


# @pytest.mark.parametrize(
#     "tiles_filename, " "stitched_filename, " "tiles_set_scene, " "stitched_set_scene, ",
#     [
#         (
#             "OverViewScan.czi",
#             "OverView.npy",
#             "TR1",
#             "Image:0",
#         )
#     ],
# )
# def test_czi_reader_mosaic_stitching(
#     tiles_filename: str,
#     stitched_filename: str,
#     tiles_set_scene: str,
#     stitched_set_scene: str,
# ) -> None:
#     # Construct full filepath
#     tiles_uri = get_resource_full_path(tiles_filename, LOCAL)
#     stitched_uri = get_resource_full_path(stitched_filename, LOCAL)
#
#     # Construct reader
#     tiles_reader = CziReader(tiles_uri)
#     stitched_np = np.load(stitched_uri)
#     stitched_reader = ArrayLikeReader(image=stitched_np)
#
#     # Run checks
#     run_image_container_mosaic_checks(
#         tiles_image_container=tiles_reader,
#         stitched_image_container=stitched_reader,
#         tiles_set_scene=tiles_set_scene,
#         stitched_set_scene=stitched_set_scene,
#     )


# @pytest.mark.parametrize(
#     "tiles_filename, "
#     "stitched_filename, "
#     "tiles_set_scene, "
#     "stitched_set_scene, ",
#     [
#         (
#             "Multiscene_CZI_3Scenes.czi",
#             "Multiscene_CZI_3Scenes.png",
#             "TR1",
#             "Image:0"
#         ),
#     ],
# )
# def test_czi_reader_mosaic_stitching(
#     tiles_filename: str,
#     stitched_filename: str,
#     tiles_set_scene: str,
#     stitched_set_scene: str,
# ) -> None:
#     # Construct full filepath
#     tiles_uri = get_resource_full_path(tiles_filename, LOCAL)
#     stitched_uri = get_resource_full_path(stitched_filename, LOCAL)
#
#     # Construct reader
#     tiles_reader = CziReader(tiles_uri)
#     stitched_reader = DefaultReader(image=stitched_uri)
#
#     # Run checks
#     run_image_container_mosaic_checks(
#         tiles_image_container=tiles_reader,
#         stitched_image_container=stitched_reader,
#         tiles_set_scene=tiles_set_scene,
#         stitched_set_scene=stitched_set_scene,
#     )

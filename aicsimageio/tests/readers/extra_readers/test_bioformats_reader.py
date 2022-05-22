#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Tuple, Union

import numpy as np
import pytest
from ome_types import OME

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers.bioformats_reader import BioFile, BioformatsReader
from aicsimageio.tests.image_container_test_utils import (
    run_image_file_checks,
    run_multi_scene_image_read_checks,
)

from ...conftest import LOCAL, get_resource_full_path, host


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
            "s_1_t_1_c_1_z_1.czi #1",
            ("s_1_t_1_c_1_z_1.czi #1",),
            (1, 1, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image0",
            ("Image0",),
            (1, 2, 1, 32, 32, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0", "Channel:0:1"],
            (None, None, None),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "s_3_t_10_c_3_z_5.czi #1",
            (
                "s_3_t_10_c_3_z_5.czi #1",
                "s_3_t_10_c_3_z_5.czi #2",
                "s_3_t_10_c_3_z_5.czi #3",
            ),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "s_3_t_10_c_3_z_5.czi #2",
            (
                "s_3_t_10_c_3_z_5.czi #1",
                "s_3_t_10_c_3_z_5.czi #2",
                "s_3_t_10_c_3_z_5.czi #3",
            ),
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
        (
            "s_3_t_1_c_3_z_5.czi",
            "s_3_t_1_c_3_z_5.czi #1",
            (
                "s_3_t_1_c_3_z_5.czi #1",
                "s_3_t_1_c_3_z_5.czi #2",
                "s_3_t_1_c_3_z_5.czi #3",
            ),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "RGB-8bit.czi",
            "RGB-8bit.czi #1",
            ("RGB-8bit.czi #1",),
            (1, 1, 1, 624, 924, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["C1"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_1_t_1_c_2_z_1.lif",
            "PEI_laminin_35k",
            ("PEI_laminin_35k",),
            (1, 2, 1, 2048, 2048),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1"],  # not as nice as lif reader
            (None, 0.325, 0.325),
        ),
        (
            "ND2_aryeh_but3_cont200-1.nd2",
            "ND2_aryeh_but3_cont200-1.nd2 (series 1)",
            (
                "ND2_aryeh_but3_cont200-1.nd2 (series 1)",
                "ND2_aryeh_but3_cont200-1.nd2 (series 2)",
                "ND2_aryeh_but3_cont200-1.nd2 (series 3)",
                "ND2_aryeh_but3_cont200-1.nd2 (series 4)",
                "ND2_aryeh_but3_cont200-1.nd2 (series 5)",
            ),
            (1, 2, 1, 1040, 1392),
            np.dtype(">u2"),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["20phase", "20xDiO"],
            (None, None, None),
        ),
        (
            "ND2_jonas_header_test2.nd2",
            "ND2_jonas_header_test2.nd2 (series 1)",
            ("ND2_jonas_header_test2.nd2 (series 1)",),
            (4, 1, 5, 520, 696),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["PSM_GFP"],
            (0.5, 0.12863494437945, 0.12863494437945),
        ),
        (
            "ND2_maxime_BF007.nd2",
            "ND2_maxime_BF007.nd2 (series 1)",
            ("ND2_maxime_BF007.nd2 (series 1)",),
            (1, 1, 1, 156, 164),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["405/488/561/633nm"],
            (None, 0.158389678930686, 0.158389678930686),
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
        ),
        (
            "DV_siRNAi-HeLa_IN_02.r3d",
            "DV_siRNAi-HeLa_IN_02.r3d",
            ("DV_siRNAi-HeLa_IN_02.r3d",),
            (1, 4, 40, 512, 512),
            np.dtype(">i2"),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3"],
            (0.20000000298023224, 0.06502940505743027, 0.06502940505743027),
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
        ),
        (
            "KLB_samples_img.klb",
            "KLB_samples_img.klb",
            ("KLB_samples_img.klb",),
            (1, 1, 29, 151, 101),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
            (1.0, 1.0, 1.0),
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
    ],
)
def test_bioformats_reader(
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
        ImageContainer=BioformatsReader,
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
            "3500002402_100X_20181009_1-Scene-10-aligned_cropped-P9-G04.czi #1",
            ("3500002402_100X_20181009_1-Scene-10-aligned_cropped-P9-G04.czi #1",),
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
        pytest.param(
            "variance-cfe.ome.tiff",
            "3500003034_100X_20190520_1-Scene-18-P108-G03.czi #1",
            ("3500003034_100X_20190520_1-Scene-18-P108-G03.czi #1",),
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
            "IMAGE0",
            ("IMAGE0",),
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
def test_bioformats_reader_large_files(
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
        ImageContainer=BioformatsReader,
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
            "s_3_t_10_c_3_z_5.czi #1",
            (1, 3, 5, 325, 475),
            "s_3_t_10_c_3_z_5.czi #2",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "s_3_t_10_c_3_z_5.czi #2",
            (1, 3, 5, 325, 475),
            "s_3_t_10_c_3_z_5.czi #3",
            (1, 3, 5, 325, 475),
        ),
    ],
)
def test_multi_scene_bioformats_reader(
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
        ImageContainer=BioformatsReader,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


def test_biofile_scene_change() -> None:
    """Make sure that ResourceBackedDaskArray doesn't close an opened file."""
    uri = get_resource_full_path("ND2_dims_p4z5t3c2y32x32.nd2", LOCAL)
    f = BioFile(uri)
    assert isinstance(f.to_dask().compute(), np.ndarray)
    f.set_series(1)
    assert isinstance(f.to_dask().compute(), np.ndarray)
    f.close()


@pytest.mark.parametrize("filename, ", [("CMU-1-Small-Region.svs")])
def test_bioformats_dask_tiling_shapes(filename: str) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    bf_tiled = BioformatsReader(uri, dask_tiles=True)
    bf_fullplane = BioformatsReader(uri, dask_tiles=False)
    bf_tiled_set = BioformatsReader(uri, dask_tiles=True, tile_size=(1024, 1024))

    np.testing.assert_array_equal(
        bf_tiled.dask_data.shape, bf_fullplane.dask_data.shape
    )
    np.testing.assert_array_equal(
        bf_tiled_set.dask_data.shape, bf_fullplane.dask_data.shape
    )
    np.testing.assert_array_equal(
        bf_tiled.dask_data.shape, bf_fullplane.dask_data.shape
    )
    np.testing.assert_array_equal(bf_tiled.dask_data.chunksize, (1, 1, 1, 240, 240, 3))
    np.testing.assert_array_equal(
        bf_tiled_set.dask_data.chunksize, (1, 1, 1, 1024, 1024, 3)
    )


@pytest.mark.parametrize(
    "filename, ",
    [("s_1_t_1_c_10_z_1.ome.tiff"), ("CMU-1-Small-Region.svs")],
)
def test_bioformats_dask_tiling_read(filename: str) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    bf_tiled = BioformatsReader(uri, dask_tiles=True)
    bf_tiled_set = BioformatsReader(uri, dask_tiles=True, tile_size=(128, 128))
    bf_fullplane = BioformatsReader(uri, dask_tiles=False)

    arr_tiled = bf_tiled.dask_data.compute()
    arr_fullplane = bf_fullplane.dask_data.compute()
    arr_tiled_set = bf_tiled_set.dask_data.compute()

    np.testing.assert_array_equal(arr_tiled, arr_fullplane)
    np.testing.assert_array_equal(arr_tiled_set, arr_fullplane)


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

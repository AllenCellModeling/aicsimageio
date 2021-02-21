#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest
from readlif.reader import LifFile

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import LifReader

from ..conftest import LOCAL, get_resource_full_path, host
from ..image_container_test_utils import run_image_file_checks


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
            (1, 1, 2, 1, 2048, 2048),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"],
            (1.0, 3.076923076923077, 3.076923076923077),
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            ("b2_001_Crop001_Resize001",),
            (1, 4, 2, 1, 614, 614),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"],
            (1.0, 2.9485556406398508, 2.9485556406398508),
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
    expected_physical_pixel_sizes: Tuple[float, float, float],
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
@pytest.mark.parametrize("chunk_by_dims", ["ZYX", "TYX", "CYX"])
@pytest.mark.parametrize("get_dims", ["ZYX", "TYX"])
def test_sanity_check_correct_indexing(
    filename: str,
    chunk_by_dims: str,
    get_dims: str,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct reader
    reader = LifReader(uri, chunk_by_dims=chunk_by_dims)
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

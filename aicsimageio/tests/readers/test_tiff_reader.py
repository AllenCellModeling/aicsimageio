#!/usr/bin/env python
import numpy as np
import pytest

from aicsimageio.readers import TiffReader


@pytest.mark.parametrize("name, expected", [
    ("s_1_t_1_c_1_z_1.tiff", True),
    ("s_1_t_10_c_3_z_1.tiff", True),
    ("example.png", False),
    ("s_1_t_1_c_1_z_1.ome.tiff", True)
])
def test_type_check(resources_dir, name, expected):
    tiff_image = resources_dir / name
    assert TiffReader.is_this_type(tiff_image) == expected


@pytest.mark.parametrize("name, shape, dims, metadata, dtype", [
    ("s_1_t_1_c_1_z_1.tiff", (325, 475), "YX", "ImageJ", np.uint16),
    ("s_1_t_10_c_3_z_1.tiff", (10, 3, 325, 475), "CZYX", "ImageJ", np.uint16),
    ("s_1_t_1_c_1_z_1.ome.tiff", (325, 475), "YX", "OME-XML", np.uint16)
])
def test_load(resources_dir, name, shape, dims, metadata, dtype):
    with TiffReader(resources_dir / name) as reader:
        data, actual_dims, actual_metadata = reader.load()
        assert data.shape == shape
        assert actual_dims == dims
        if metadata is None:
            assert actual_metadata == ""
        else:
            assert metadata in actual_metadata
        assert dtype == reader.dtype()

from aicsimageio.readers.czi_reader import CziReader
from io import BytesIO
from pathlib import Path
import pytest

import os
import unittest

@pytest.mark.parametrize("file", [
    "/Users/jamies/20180907_M01_001.czi",
    pytest.param(BytesIO(b"abcdef"), marks=pytest.mark.raises(exception=CziReader.FileNotCompatibleWithCziFileLibrary)),
    pytest.param("non_existent_file.random", marks=pytest.mark.raises(exception=FileNotFoundError)),
    pytest.param(Path("/nonexistent/file/file.random"), marks=pytest.mark.raises(exception=FileNotFoundError))
    ]
)
def test_czi_reader(file):
    czi = CziReader(file)
    czi.close()


@pytest.mark.parametrize("test_input,expected", [
    (BytesIO(b'ZInotaczinope'), False),
    (BytesIO(b'ZISRAWFILE'), True),
    (BytesIO(b'ZI'), False),
    (BytesIO(b''), False)
    ]
)
def test_is_this_type(test_input, expected):
    res = CziReader._is_this_type(test_input)
    assert res == expected


@pytest.mark.parametrize("test_input,expected", [
    ("test_5_dimension.czi", "BCYX0"),
    ("T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi", "BTCZYX0")
])
def test_dims(image_dir, test_input, expected):
    czi = CziReader(image_dir / test_input)
    assert czi.dims == expected


@pytest.mark.parametrize("test_input,expected", [
    ("test_5_dimension.czi", 'uint16'),
    ("T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi", 'uint8')
])
def test_dtype(image_dir, test_input, expected):
    czi = CziReader(image_dir / test_input)
    assert czi.dtype() == expected


def test_shape(image_dir):
    czi = CziReader(image_dir / "T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi")
    data = czi.data
    data_shape = data.shape
    assert data_shape[0] == 1
    assert data_shape[1] == czi.size_t()
    assert data_shape[2] == czi.size_c()
    assert data_shape[3] == czi.size_z()
    assert data_shape[4] == czi.size_y()
    assert data_shape[5] == czi.size_x()
    assert czi._is_multiscene() is False


@pytest.mark.parametrize("test_input,expected", [
    ('T', 1), ('C', 2), ('Z', 3), ('Y', 4), ('X', 5), ('S', -1), ('B', 0), ('M', -1), ('V', -1),
    pytest.param("TZ", False, marks=pytest.mark.raises(exception=TypeError))
])
def test_lookup_index(image_dir, test_input, expected):
    czi = CziReader(image_dir / 'T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi')
    assert czi._lookup_dimension_index(test_input) == expected


def test_missing_dimension(image_dir):
    czi = CziReader(image_dir / 'T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi')
    assert czi._size_of_dimension('V') == 0

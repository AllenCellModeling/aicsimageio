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
    CziReader(file)


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


def test_dims():
    czi = CziReader("/Users/jamies/20180907_M01_001.czi")
    assert czi.dims == "BSTCZYX0"


def test_shape():
    czi = CziReader("img/test_5_dimensions.czi")
    data = czi.data
    data_shape = data.shape
    assert data_shape[0] == 1
    assert data_shape[1] == 1
    assert data_shape[2] == czi.size_t()
    assert data_shape[3] == czi.size_c()
    assert data_shape[4] == czi.size_z()
    assert data_shape[5] == czi.size_y()
    assert data_shape[6] == czi.size_x()
    assert czi._is_multiscene() is False

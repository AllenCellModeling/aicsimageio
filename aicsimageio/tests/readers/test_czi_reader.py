from io import BytesIO
from pathlib import Path

import pytest

from aicsimageio.exceptions import UnsupportedFileFormatError
from aicsimageio.readers.czi_reader import CziReader

# Example files
TWO_DIM_CZI = "s_1_t_1_c_1_z_1.czi"
SIX_DIM_CZI = "s_3_t_1_c_3_z_5.czi"


# TODO It would be good to test that given a multiscene defined exception is raised


@pytest.mark.parametrize("file", [
    TWO_DIM_CZI,
    SIX_DIM_CZI,
    pytest.param(BytesIO(b"abcdef"), marks=pytest.mark.raises(exception=UnsupportedFileFormatError)),
    pytest.param("non_existent_file.random", marks=pytest.mark.raises(exception=FileNotFoundError)),
    pytest.param(Path("/nonexistent/file/file.random"), marks=pytest.mark.raises(exception=FileNotFoundError))
    ]
)
def test_czi_reader(resources_dir, file):
    fobj = file
    if isinstance(file, str):
        fobj = resources_dir / file
    czi = CziReader(fobj)
    czi.close()


@pytest.mark.parametrize("test_input, expected", [
    (BytesIO(b"ZInotaczinope"), False),
    (BytesIO(b"ZISRAWFILE"), True),
    (BytesIO(b"ZI"), False),
    (BytesIO(b""), False)
    ]
)
def test_is_this_type(test_input, expected):
    res = CziReader._is_this_type(test_input)
    assert res == expected


@pytest.mark.parametrize("test_input, expected", [
    (TWO_DIM_CZI, "BCYX0"),
    (SIX_DIM_CZI, "BSCZYX0")
])
def test_dims(resources_dir, test_input, expected):
    czi = CziReader(resources_dir / test_input)
    assert czi.dims == expected


@pytest.mark.parametrize("test_input, expected", [
    (TWO_DIM_CZI, "uint16"),
    (SIX_DIM_CZI, "uint16")
])
def test_dtype(resources_dir, test_input, expected):
    czi = CziReader(resources_dir / test_input)
    assert czi.dtype() == expected


def test_shape(resources_dir):
    czi = CziReader(resources_dir / SIX_DIM_CZI)
    data = czi.data
    data_shape = data.shape
    # BSCZYX0
    assert data_shape[0] == 1
    assert data_shape[1] == czi.size_s()
    assert data_shape[2] == czi.size_c()
    assert data_shape[3] == czi.size_z()
    assert data_shape[4] == czi.size_y()
    assert data_shape[5] == czi.size_x()
    assert czi._is_multiscene() is False


@pytest.mark.parametrize("test_input, expected", [
    ("T", -1), ("C", 2), ("Z", 3), ("Y", 4), ("X", 5), ("S", 1), ("B", 0), ("M", -1), ("V", -1),
    pytest.param("TZ", False, marks=pytest.mark.raises(exception=TypeError))
])
def test_dimension_index(resources_dir, test_input, expected):
    czi = CziReader(resources_dir / SIX_DIM_CZI)
    # BSCZYX0
    assert czi._lookup_dimension_index(test_input) == expected


def test_missing_dimension(resources_dir):
    czi = CziReader(resources_dir / SIX_DIM_CZI)
    assert czi._size_of_dimension("V") == 1
    assert czi._lookup_dimension_index("V") == -1


# NOTE:
# These are all going to have the same channel setup id because they were all created from the same base file
# As in, look at the most complex czi, and all of the other czis are just various slices of that one.
@pytest.mark.parametrize("test_input, expected", [
    (TWO_DIM_CZI, "636972569326165806"),
    (SIX_DIM_CZI, "636972569326165806")
])
def test_metadata(resources_dir, test_input, expected):
    czi = CziReader(resources_dir / test_input)
    checked = False
    for it in czi.metadata.iter("Channel"):
        x = it.attrib.get("ChannelSetupId")
        if x:
            assert x == expected
            checked = True
            break
    assert checked

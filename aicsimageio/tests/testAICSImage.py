from xml.etree import cElementTree as etree

import numpy as np

import pytest
from aicsimageio import AICSImage, exceptions, readers
from aicsimageio.vendor import omexml


@pytest.mark.parametrize('filename, expected_reader', [
    ('img40_1.png', readers.DefaultReader),
    ('img40_1_dna.tif', readers.TiffReader),
    ('img40_1.ome.tif', readers.OmeTiffReader),
    ('T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi', readers.CziReader),
    pytest.param('not/a/file.czi', None, marks=pytest.mark.raises(exception=FileNotFoundError)),
    pytest.param(
        'not_an_image.txt', None, marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError)
    )
])
def test_typing(filename, expected_reader, image_dir):
    actual_reader = AICSImage.determine_reader(image_dir / filename)
    assert actual_reader == expected_reader


@pytest.mark.parametrize('arr', [
    np.zeros((2, 2, 2)),
    np.ones((2, 2, 2)),
    np.random.rand(2, 2, 2)
])
def test_support_for_ndarray(arr):
    actual_reader = AICSImage.determine_reader(arr)
    assert actual_reader == readers.NdArrayReader


@pytest.mark.parametrize("data,expected", [
    (np.zeros((5, 4, 3)), (1, 1, 1, 5, 4, 3)),
    (np.zeros((1, 2, 3, 4, 5, 6)), (1, 2, 3, 4, 5, 6)),
    (np.random.rand(10), (1, 1, 1, 1, 1, 10))
])
def test_aicsimage_default_dims(data, expected):
    img = AICSImage(data=data)
    assert img.data.shape == expected


@pytest.mark.parametrize("data_shape,dims,expected", [
    ((5, 4, 3), "STC", (5, 4, 3, 1, 1, 1)),
    ((1, 2, 3, 4, 5, 6), "XYZCTS", (6, 5, 4, 3, 2, 1)),
    ((5, 4, 3), "SCY", (5, 1, 4, 1, 3, 1)),
    ((1, 2, 3, 4, 5, 6), "STCZYX", (1, 2, 3, 4, 5, 6))
])
def test_aicsimage_force_dims(data_shape, dims, expected):
    img = AICSImage(data=np.zeros(data_shape))
    img._reader.dims = dims
    assert img.data.shape == expected
    assert data_shape == img.get_image_data(out_orientation=dims).shape


@pytest.mark.parametrize("filepath", [
        "img40_1.ome.tif",
        pytest.param("fakeimage.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
        pytest.param("a/bogus/file.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
])
def test_file_exceptions(image_dir, filepath):
    f = image_dir / filepath
    AICSImage(f)


def test_file_passed_was_directory(image_dir):
    with pytest.raises(IsADirectoryError):
        AICSImage(image_dir)


def test_file_passed_was_byte_string(image_dir):
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        AICSImage(b"not-a-valid-image-byte-array")


@pytest.mark.parametrize('filename, expected_metadata_type', [
    ('img40_1.png', (str, type(None))),
    ('img40_1_dna.tif', (str, type(None))),
    ('img40_1.ome.tif', (str, omexml.OMEXML)),
    ('T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi', (str, etree.Element))
])
def test_metadata(filename, expected_metadata_type, image_dir):
    img = AICSImage(image_dir / filename)
    print(f"class: {img.metadata.__class__.__name__}")
    assert isinstance(img.metadata, expected_metadata_type)


@pytest.mark.parametrize('filename, expected_reader', [
    ('img40_1.png', readers.DefaultReader),
    ('img40_1_dna.tif', readers.TiffReader),
    ('img40_1.ome.tif', readers.OmeTiffReader),
    ('T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi', readers.CziReader),
    pytest.param('not/a/file.czi', None, marks=pytest.mark.raises(exception=FileNotFoundError)),
])
def test_reader(filename, expected_reader, image_dir):
    img = AICSImage(image_dir / filename)
    assert isinstance(img.reader, expected_reader)

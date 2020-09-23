#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from unittest import mock
from xml.etree.ElementTree import Element

import dask.array as da
import numpy as np
import pytest
from lxml.etree import _Element
from psutil import Process

from aicsimageio import AICSImage, exceptions, imread, readers
from aicsimageio.vendor import omexml

# Example files
BMP_FILE = "example.bmp"
JPG_FILE = "example.jpg"
PNG_FILE = "example.png"
GIF_FILE = "example.gif"
TIF_FILE = "s_1_t_1_c_1_z_1.tiff"
CZI_FILE = "s_1_t_1_c_1_z_1.czi"
LIF_FILE = "s_1_t_1_c_2_z_1.lif"
OME_FILE = "s_1_t_1_c_1_z_1.ome.tiff"
MED_TIF_FILE = "s_1_t_10_c_3_z_1.tiff"
BIG_OME_FILE = "s_3_t_1_c_3_z_5.ome.tiff"
BIG_CZI_FILE = "s_3_t_1_c_3_z_5.czi"
BIG_LIF_FILE = "s_1_t_4_c_2_z_1.lif"
TXT_FILE = "example.txt"


@pytest.mark.parametrize(
    "filename, expected_reader",
    [
        (BMP_FILE, readers.DefaultReader),
        (JPG_FILE, readers.DefaultReader),
        (PNG_FILE, readers.DefaultReader),
        (GIF_FILE, readers.DefaultReader),
        (TIF_FILE, readers.TiffReader),
        (OME_FILE, readers.OmeTiffReader),
        (CZI_FILE, readers.CziReader),
        (LIF_FILE, readers.LifReader),
        pytest.param(
            TXT_FILE,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "not/a/file.czi",
            None,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
    ],
)
def test_typing(filename, expected_reader, resources_dir):
    # Get filepath
    f = resources_dir / filename

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    actual_reader = AICSImage.determine_reader(f)
    assert actual_reader == expected_reader

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


def test_file_passed_was_directory(resources_dir):
    # Get filepath
    f = resources_dir

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    with pytest.raises(IsADirectoryError):
        AICSImage(resources_dir)

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((2, 2, 2)),
        np.ones((2, 2, 2)),
        np.random.rand(2, 2, 2),
        da.zeros((2, 2, 2)),
        da.ones((2, 2, 2)),
        da.random.random((2, 2, 2)),
    ],
)
def test_support_for_ndarray(arr):
    # Check basics
    actual_reader = AICSImage.determine_reader(arr)
    assert actual_reader == readers.ArrayLikeReader


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.zeros((5, 4, 3)), (1, 1, 1, 5, 4, 3)),
        (np.zeros((1, 2, 3, 4, 5, 6)), (1, 2, 3, 4, 5, 6)),
        (np.random.rand(10), (1, 1, 1, 1, 1, 10)),
        (da.zeros((5, 4, 3)), (1, 1, 1, 5, 4, 3)),
        (da.zeros((1, 2, 3, 4, 5, 6)), (1, 2, 3, 4, 5, 6)),
        (da.random.random((10)), (1, 1, 1, 1, 1, 10)),
    ],
)
def test_default_shape_expansion(data, expected):
    # Check basics
    img = AICSImage(data=data)
    assert img.dask_data.shape == expected
    assert img.shape == expected


@pytest.mark.parametrize(
    "data, dims, expected_shape",
    [
        (np.zeros((5, 4, 3)), "SYX", (5, 1, 1, 1, 4, 3)),
        (np.zeros((1, 2, 3, 4, 5)), "STCYX", (1, 2, 3, 1, 4, 5)),
        (np.zeros((10, 20)), "XY", (1, 1, 1, 1, 20, 10)),
        pytest.param(
            np.zeros((2, 2, 2)),
            "ABI",
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        (da.zeros((5, 4, 3)), "SYX", (5, 1, 1, 1, 4, 3)),
        (da.zeros((1, 2, 3, 4, 5)), "STCYX", (1, 2, 3, 1, 4, 5)),
        (da.zeros((10, 20)), "XY", (1, 1, 1, 1, 20, 10)),
        pytest.param(
            da.zeros((2, 2, 2)),
            "ABI",
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
def test_known_dims(data, dims, expected_shape):
    img = AICSImage(data, known_dims=dims)
    assert img.data.shape == expected_shape
    assert img.size_x == expected_shape[5]
    assert img.size_y == expected_shape[4]
    assert img.size_z == expected_shape[3]
    assert img.size_c == expected_shape[2]
    assert img.size_t == expected_shape[1]
    assert img.size_s == expected_shape[0]
    assert img.size(dims) == data.shape


@pytest.mark.parametrize(
    "data_shape, dims, expected",
    [
        ((5, 4, 3), "STC", (5, 4, 3, 1, 1, 1)),
        ((1, 2, 3, 4, 5, 6), "XYZCTS", (6, 5, 4, 3, 2, 1)),
        ((5, 4, 3), "SCY", (5, 1, 4, 1, 3, 1)),
        ((1, 2, 3, 4, 5, 6), "STCZYX", (1, 2, 3, 4, 5, 6)),
    ],
)
def test_force_dims(data_shape, dims, expected):
    img = AICSImage(data=da.zeros(data_shape))
    img._reader._dims = dims
    assert img.data.shape == expected
    assert data_shape == img.get_image_data(out_orientation=dims).shape
    assert img.size_x == expected[5]
    assert img.size_y == expected[4]
    assert img.size_z == expected[3]
    assert img.size_c == expected[2]
    assert img.size_t == expected[1]
    assert img.size_s == expected[0]
    assert img.size(dims) == data_shape


@pytest.mark.parametrize(
    "filename, expected_metadata_type",
    [
        (PNG_FILE, dict),
        (TIF_FILE, str),
        (OME_FILE, omexml.OMEXML),
        (CZI_FILE, _Element),
        (LIF_FILE, Element),
    ],
)
def test_metadata(resources_dir, filename, expected_metadata_type):
    # Get filepath
    f = resources_dir / filename

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    img = AICSImage(f)
    assert isinstance(img.metadata, expected_metadata_type)

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        (PNG_FILE, (1, 1, 4, 1, 800, 537)),
        (TIF_FILE, (1, 1, 1, 1, 325, 475)),
        (OME_FILE, (1, 1, 1, 1, 325, 475)),
        (CZI_FILE, (1, 1, 1, 1, 325, 475)),
        (LIF_FILE, (1, 1, 2, 1, 2048, 2048)),
        (MED_TIF_FILE, (1, 10, 3, 1, 325, 475)),
        (BIG_OME_FILE, (3, 1, 3, 5, 325, 475)),
        (BIG_CZI_FILE, (3, 1, 3, 5, 325, 475)),
        (BIG_LIF_FILE, (1, 4, 2, 1, 614, 614)),
    ],
)
def test_imread(resources_dir, filename, expected_shape):
    # Get filepath
    f = resources_dir / filename

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    img = imread(f)
    assert img.shape == expected_shape

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, expected_channel_names",
    [
        (PNG_FILE, ["Red", "Green", "Blue", "Alpha"]),
        (GIF_FILE, ["Red", "Green", "Blue", "Alpha"]),
        (JPG_FILE, ["Red", "Green", "Blue"]),
        (TIF_FILE, ["0"]),
        (MED_TIF_FILE, ["0", "1", "2"]),
        (CZI_FILE, ["Bright"]),
        (LIF_FILE, ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"]),
        (OME_FILE, ["Bright"]),
        (BIG_CZI_FILE, ["EGFP", "TaRFP", "Bright"]),
        (BIG_LIF_FILE, ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"]),
    ],
)
def test_channel_names(resources_dir, filename, expected_channel_names):
    # Get filepath
    f = resources_dir / filename

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    img = AICSImage(f)
    assert img.get_channel_names() == expected_channel_names
    assert len(img.get_channel_names()) == img.size_c

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, expected_sizes",
    [
        (PNG_FILE, (1.0, 1.0, 1.0)),
        (TIF_FILE, (1.0, 1.0, 1.0)),
        (CZI_FILE, (1.0833333333333333e-06, 1.0833333333333333e-06, 1.0)),
        (OME_FILE, (1.0833333333333333, 1.0833333333333333, 1.0)),
        (LIF_FILE, (3.25e-07, 3.25e-07, 1.0)),
        (BIG_LIF_FILE, (3.3914910277324634e-07, 3.3914910277324634e-07, 1.0)),
    ],
)
def test_physical_pixel_size(resources_dir, filename, expected_sizes):
    # Get filepath
    f = resources_dir / filename

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    img = AICSImage(f)
    assert img.get_physical_pixel_size() == expected_sizes

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "data, rgb, expected_data, expected_visible, expected_ndim, expected_axis_labels",
    [
        (
            # C Z Y X
            np.ones((3, 2, 2, 2)),
            False,
            da.ones((3, 2, 2, 2)),
            True,
            3,
            "ZYX",
        ),
        (
            # C Z Y X
            da.ones((3, 2, 2, 2)),
            False,
            da.ones((3, 2, 2, 2)),
            True,
            3,
            "ZYX",
        ),
        (
            # C Z Y X
            np.ones((3, 2, 2, 2)),
            True,
            da.ones((2, 2, 2, 3)),
            True,
            3,
            "ZYX",
        ),
        (
            # C Z Y X
            da.ones((3, 2, 2, 2)),
            True,
            da.ones((2, 2, 2, 3)),
            True,
            3,
            "ZYX",
        ),
        (
            # S T C Z Y X
            np.ones((1, 1, 3, 2, 2, 2)),
            False,
            da.ones((3, 2, 2, 2)),
            True,
            3,
            "ZYX",
        ),
        (
            # S T C Z Y X
            da.ones((1, 1, 3, 2, 2, 2)),
            False,
            da.ones((3, 2, 2, 2)),
            True,
            3,
            "ZYX",
        ),
        (
            # S T C Z Y X
            np.ones((1, 1, 3, 2, 2, 2)),
            True,
            da.ones((2, 2, 2, 3)),
            True,
            3,
            "ZYX",
        ),
        (
            # S T C Z Y X
            da.ones((1, 1, 3, 2, 2, 2)),
            True,
            da.ones((2, 2, 2, 3)),
            True,
            3,
            "ZYX",
        ),
        (
            # S T C Z Y X
            np.ones((3, 20, 5, 2, 2, 2)),
            False,
            da.ones((3, 20, 5, 2, 2, 2)),
            False,
            3,
            "STZYX",
        ),
        (
            # S T C Z Y X
            da.ones((3, 20, 5, 2, 2, 2)),
            False,
            da.ones((3, 20, 5, 2, 2, 2)),
            False,
            3,
            "STZYX",
        ),
    ],
)
def test_view_napari(
    data, rgb, expected_data, expected_visible, expected_ndim, expected_axis_labels
):
    # Init image
    img = AICSImage(data)

    # Mock napari view
    with mock.patch("napari.gui_qt"):
        with mock.patch("napari.view_image") as mocked_napari:
            img.view_napari(rgb)

            # Check array equal
            args = mocked_napari.call_args[0]
            assert args[0].shape == expected_data.shape

            # Check extra call kwargs
            call_kwargs = mocked_napari.call_args[1]
            assert not call_kwargs["is_pyramid"]
            assert call_kwargs["ndisplay"] == expected_ndim
            assert call_kwargs["axis_labels"] == expected_axis_labels
            if not rgb:
                assert call_kwargs["visible"] == expected_visible


@pytest.mark.parametrize(
    "filename, expected_shape, expected_metadata_type",
    [
        (PNG_FILE, (1, 1, 4, 1, 800, 537), dict),
        (TIF_FILE, (1, 1, 1, 1, 325, 475), str),
        (OME_FILE, (1, 1, 1, 1, 325, 475), omexml.OMEXML),
        (CZI_FILE, (1, 1, 1, 1, 325, 475), _Element),
        (
            LIF_FILE,
            (1, 1, 2, 1, 2048, 2048),
            Element,
        ),
        (MED_TIF_FILE, (1, 10, 3, 1, 325, 475), str),
        (BIG_OME_FILE, (3, 1, 3, 5, 325, 475), omexml.OMEXML),
        (BIG_CZI_FILE, (3, 1, 3, 5, 325, 475), _Element),
    ],
)
def test_aicsimage_serialize(
    resources_dir,
    tmpdir,
    filename,
    expected_shape,
    expected_metadata_type,
):
    """
    Test that the entire AICSImage object can be serialized - a requirement to
    distribute on dask clusters.

    https://distributed.dask.org/en/latest/serialization.html
    """
    # Get file
    f = resources_dir / filename

    # Read file
    img = AICSImage(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.shape == expected_shape
    assert isinstance(img.metadata, expected_metadata_type)

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Serialize object
    serialized = pickle.dumps(img)

    # Reload
    img = pickle.loads(serialized)

    # Check array
    assert isinstance(img.data, np.ndarray)
    assert img.shape == expected_shape
    assert img.data.shape == expected_shape
    assert isinstance(img.metadata, expected_metadata_type)

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, expected_dims, expected_shape",
    [
        ("s_1_t_1_c_3_z_1_RGB.tiff", "STCZYX", (1, 1, 3, 1, 7548, 7548)),
        pytest.param(
            "s_1_t_1_c_6_z_1_RGB.tiff",
            "STCZYX",
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_rgb_images(
    resources_dir,
    filename,
    expected_dims,
    expected_shape,
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = AICSImage(f)

    assert img.dims == expected_dims
    assert img.shape == expected_shape

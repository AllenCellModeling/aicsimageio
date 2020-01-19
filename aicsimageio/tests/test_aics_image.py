#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest
from dask.diagnostics import Profiler
from lxml.etree import _Element
from psutil import Process

from aicsimageio import AICSImage, exceptions, imread, imread_dask, readers
from aicsimageio.vendor import omexml

# Example files
BMP_FILE = "example.bmp"
JPG_FILE = "example.jpg"
PNG_FILE = "example.png"
GIF_FILE = "example.gif"
TIF_FILE = "s_1_t_1_c_1_z_1.tiff"
CZI_FILE = "s_1_t_1_c_1_z_1.czi"
OME_FILE = "s_1_t_1_c_1_z_1.ome.tiff"
MED_TIF_FILE = "s_1_t_10_c_3_z_1.tiff"
BIG_OME_FILE = "s_3_t_1_c_3_z_5.ome.tiff"
BIG_CZI_FILE = "s_3_t_1_c_3_z_5.czi"
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
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        actual_reader = AICSImage.determine_reader(resources_dir / filename)
        assert actual_reader == expected_reader
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


def test_file_passed_was_directory(resources_dir):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        with pytest.raises(IsADirectoryError):
            AICSImage(resources_dir)
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize("arr", [
    np.zeros((2, 2, 2)),
    np.ones((2, 2, 2)),
    np.random.rand(2, 2, 2),
    da.zeros((2, 2, 2)),
    da.ones((2, 2, 2)),
    da.random.random((2, 2, 2))
])
def test_support_for_ndarray(arr):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        actual_reader = AICSImage.determine_reader(arr)
        assert actual_reader == readers.ArrayLikeReader
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


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
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = AICSImage(data=data)
        assert img.dask_data.shape == expected
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


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
            marks=pytest.mark.raises(exception=TypeError),
        ),
        (da.zeros((5, 4, 3)), "SYX", (5, 1, 1, 1, 4, 3)),
        (da.zeros((1, 2, 3, 4, 5)), "STCYX", (1, 2, 3, 1, 4, 5)),
        (da.zeros((10, 20)), "XY", (1, 1, 1, 1, 20, 10)),
        pytest.param(
            da.zeros((2, 2, 2)),
            "ABI",
            None,
            marks=pytest.mark.raises(exception=TypeError),
        ),
    ],
)
def test_known_dims(data, dims, expected_shape):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = AICSImage(data, known_dims=dims)
        assert img.data.shape == expected_shape
        assert img.size_x == expected_shape[5]
        assert img.size_y == expected_shape[4]
        assert img.size_z == expected_shape[3]
        assert img.size_c == expected_shape[2]
        assert img.size_t == expected_shape[1]
        assert img.size_s == expected_shape[0]
        assert img.size(dims) == data.shape

        # Due to reshape and transpose there will be 2 tasks in the graph
        assert len(prof.results) == 2

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


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
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
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

        # Two operations are happening
        # First, img.data is called and so two tasks of reshape and transpose are ran
        # Then get_image_data is ran and two more reshape and transpose are ran
        assert len(prof.results) == 4

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize(
    "filename, expected_metadata_type",
    [
        (PNG_FILE, dict),
        (TIF_FILE, str),
        (OME_FILE, omexml.OMEXML),
        (CZI_FILE, _Element),
    ],
)
def test_metadata(resources_dir, filename, expected_metadata_type):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = AICSImage(resources_dir / filename)
        assert isinstance(img.metadata, expected_metadata_type)

        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        (PNG_FILE, (1, 1, 4, 1, 800, 537)),
        (TIF_FILE, (1, 1, 1, 1, 325, 475)),
        (OME_FILE, (1, 1, 1, 1, 325, 475)),
        (CZI_FILE, (1, 1, 1, 1, 325, 475)),
    ],
)
def test_imread(resources_dir, filename, expected_shape):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = imread(resources_dir / filename)
        assert img.shape == expected_shape

        # Reshape and transpose are required so there should be two tasks in the graph
        assert len(prof.results) == 2

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize("filename, expected_shape, expected_task_count", [
    # Because we are directly requesting the data, the transpose and reshape calls get reduced
    (MED_TIF_FILE, (1, 10, 3, 1, 325, 475), 60),
    (BIG_OME_FILE, (3, 1, 3, 5, 325, 475), 90),
    (BIG_CZI_FILE, (3, 1, 3, 5, 325, 475), 18)
])
def test_large_imread(resources_dir, filename, expected_shape, expected_task_count):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = imread(resources_dir / filename)
        assert img.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize("filename, expected_shape, expected_task_count", [
    # Because we are directly returning the dask array nothing has been computed
    (MED_TIF_FILE, (1, 10, 3, 1, 325, 475), 0),
    (BIG_OME_FILE, (3, 1, 3, 5, 325, 475), 0),
    (BIG_CZI_FILE, (3, 1, 3, 5, 325, 475), 0)
])
def test_large_imread_dask(resources_dir, filename, expected_shape, expected_task_count):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = imread_dask(resources_dir / filename)
        assert img.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize(
    "filename, expected_channel_names",
    [
        (PNG_FILE, ["0", "1", "2", "3"]),
        (TIF_FILE, ["0"]),
        (CZI_FILE, ["Bright"]),
        (OME_FILE, ["Bright"]),
    ],
)
def test_channel_names(resources_dir, filename, expected_channel_names):
    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        img = AICSImage(resources_dir / filename)
        assert img.get_channel_names() == expected_channel_names

        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0

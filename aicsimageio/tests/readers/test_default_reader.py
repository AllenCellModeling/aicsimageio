#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from dask.diagnostics import Profiler
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader


@pytest.mark.parametrize("filename, expected_shape, expected_dims, expected_chunksize, expected_task_count", [
    ("example.bmp", (480, 640, 4), "YXC", (480, 640, 4), 0),
    ("example.png", (800, 537, 4), "YXC", (800, 537, 4), 0),
    ("example.jpg", (452, 400, 3), "YXC", (452, 400, 3), 0),
    # Task count for multiple image formats should be 2 * number of images in file
    ("example.gif", (72, 268, 268, 4), "TYXC", (1, 268, 268, 4), 144),
    pytest.param(
        "example.txt",
        None,
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError)
    )
])
def test_default_reader(
    resources_dir,
    filename,
    expected_shape,
    expected_dims,
    expected_chunksize,
    expected_task_count
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = DefaultReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    with Profiler() as prof:
        assert img.dims == expected_dims
        assert img.metadata
        assert img.dask_data.shape == expected_shape
        assert img.dask_data.chunksize == expected_chunksize
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize("expected_starting_dims, set_dims, expected_ending_dims", [
    ("YXC", "ZXC", "ZXC"),
    ("YXC", "YXZ", "YXZ"),
    ("YXC", "ABC", "ABC"),
    pytest.param("YXC", "ABCDE", None, marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError))
])
def test_dims_setting(resources_dir, expected_starting_dims, set_dims, expected_ending_dims):
    # Get file
    f = resources_dir / "example.png"

    # Read file
    img = DefaultReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    with Profiler() as prof:
        assert img.dims == expected_starting_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check no tasks happen during dims setting
    with Profiler() as prof:
        img.dims = set_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check no tasks happen during dims getting
    with Profiler() as prof:
        assert img.dims == expected_ending_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]

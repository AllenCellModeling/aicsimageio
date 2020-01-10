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
    assert len(proc.open_files()) == 0

    # Check basics
    assert img.dask_data.shape == expected_shape
    assert img.dims == expected_dims
    assert img.dask_data.chunksize == expected_chunksize

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after retrieval
    assert len(proc.open_files()) == 0

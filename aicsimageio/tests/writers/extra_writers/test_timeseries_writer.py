#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader
from aicsimageio.writers.timeseries_writer import TimeseriesWriter

from ...conftest import LOCAL, array_constructor, get_resource_write_full_path


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape, read_dim_order",
    [
        ((30, 100, 100), None, (30, 100, 100), "TYX"),
        # Note that files get saved out with RGBA, instead of just RGB
        ((30, 100, 100, 3), None, (30, 100, 100, 4), "TYXS"),
        ((100, 30, 100), "XTY", (30, 100, 100), "TYX"),
        # Note that files get saved out with RGBA, instead of just RGB
        ((3, 100, 30, 100), "SYTX", (30, 100, 100, 4), "TYXS"),
        pytest.param(
            (1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1),
            "ABCD",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.gif"])
def test_timeseries_writer(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: str,
    read_shape: Tuple[int, ...],
    read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    TimeseriesWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    reader = DefaultReader(save_uri)

    # Check basics
    assert reader.shape == read_shape
    assert reader.dims.order == read_dim_order

    # Can't do "easy" testing because compression + shape mismatches on RGB data


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape, read_dim_order",
    [
        # We use 112 instead of 100 because FFMPEG block size warnings are annoying
        ((30, 112, 112), None, (30, 112, 112, 3), "TYXS"),
        # Note that files get saved out with RGBA, instead of just RGB
        ((30, 112, 112, 3), None, (30, 112, 112, 3), "TYXS"),
        ((112, 30, 112), "XTY", (30, 112, 112, 3), "TYXS"),
        # Note that files get saved out with RGBA, instead of just RGB
        ((3, 112, 30, 112), "SYTX", (30, 112, 112, 3), "TYXS"),
        pytest.param(
            (1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1),
            "ABCD",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["f.mp4"])
def test_timeseries_writer_ffmpeg(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: str,
    read_shape: Tuple[int, ...],
    read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Catch invalid save
    # if host == REMOTE:
    #     with pytest.raises(IOError):
    #         TimeseriesWriter.save(arr, save_uri, write_dim_order)

    #     return

    # Normal save
    TimeseriesWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    reader = DefaultReader(save_uri)

    # Check basics
    assert reader.shape == read_shape
    assert reader.dims.order == read_dim_order

    # Can't do "easy" testing because compression + shape mismatches on RGB data

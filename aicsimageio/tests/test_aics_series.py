#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from aicsimageio import AICSSeries, constants, exceptions

###############################################################################

DATA_DIR = Path(__file__).parent / "resources"

###############################################################################


@pytest.mark.parametrize("images, series_dim, expected_images", [
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"]
    ),
    (
        DATA_DIR,
        "T",
        [
            DATA_DIR / "example.bmp",
            DATA_DIR / "example.gif",
            DATA_DIR / "example.jpg",
            DATA_DIR / "example.png",
            DATA_DIR / "example.txt",
            DATA_DIR / "s_1_t_10_c_3_z_1.tiff",
            DATA_DIR / "s_1_t_1_c_10_z_1.ome.tiff",
            DATA_DIR / "s_1_t_1_c_1_z_1.czi",
            DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff",
            DATA_DIR / "s_1_t_1_c_1_z_1.tiff",
            DATA_DIR / "s_3_t_1_c_3_z_5.czi",
            DATA_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
        ]
    ),
    pytest.param(
        DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff",
        None,
        None,
        marks=pytest.mark.raises(exception=NotADirectoryError)
    ),
    pytest.param(1, None, None, marks=pytest.mark.raises(exception=TypeError)),
    pytest.param([DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"], None, None, marks=pytest.mark.raises(exception=ValueError)),
    pytest.param(DATA_DIR, 1, None, marks=pytest.mark.raises(exception=ValueError)),
    pytest.param(DATA_DIR, "B", None, marks=pytest.mark.raises(exception=ValueError)),
    pytest.param(DATA_DIR, "ABC", None, marks=pytest.mark.raises(exception=ValueError))
])
def test_aics_series_init(images, series_dim, expected_images):
    series = AICSSeries(images, series_dim)
    for i, img_path in enumerate(series.images):
        assert img_path == expected_images[i]


@pytest.mark.parametrize("data_shape, operating_index, prior_shape", [
    ((2, 1, 4, 60, 480, 480), 1, None),
    ((2, 2, 2, 1, 100, 100), 3, None),
    pytest.param((2, 2, 2), 0, None, marks=pytest.mark.raises(exceptions=exceptions.InvalidDimensionOrderingError)),
    ((2, 1, 4, 60, 480, 480), 1, (2, 1, 4, 60, 480, 480)),
    pytest.param(
        (2, 1, 2, 1, 100, 100),
        1,
        (1, 1, 3),
        marks=pytest.mark.raises(exceptions=exceptions.InconsitentDataShapeException)
    )
])
def test_valid_data_shape(data_shape, operating_index, prior_shape):
    AICSSeries._ensure_valid_data_shape(data_shape, operating_index, prior_shape)


@pytest.mark.parametrize("images, series_dim, dims, expected_size", [
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        constants.DEFAULT_DIMENSION_ORDER,
        (1, 2, 1, 1, 325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        "CZYX",
        (1, 1, 325, 475)
    ),
    pytest.param(
        [DATA_DIR / "s_3_t_1_c_3_z_5.ome.tiff", DATA_DIR / "s_3_t_1_c_3_z_5.ome.tiff"],
        "Z",
        None,
        None,
        marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError)
    )
])
def test_size(images, series_dim, dims, expected_size):
    series = AICSSeries(images, series_dim)
    assert series.size(dims) == expected_size


@pytest.mark.parametrize("images, series_dim, expected_sizes", [
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (1, 2, 1, 1, 325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "S",
        (2, 1, 1, 1, 325, 475)
    )
])
def test_single_dim_size_properties(images, series_dim, expected_sizes):
    series = AICSSeries(images, series_dim)
    assert series.size_s == expected_sizes[0]
    assert series.size_t == expected_sizes[1]
    assert series.size_c == expected_sizes[2]
    assert series.size_z == expected_sizes[3]
    assert series.size_y == expected_sizes[4]
    assert series.size_x == expected_sizes[5]


@pytest.mark.parametrize("images, series_dim, selections, expected_shape", [
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, 0, 0, 0, slice(None, None, None), slice(None, None, None)),
        (325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, slice(None, None, None), 0, 0, slice(None, None, None), slice(None, None, None)),
        (2, 325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0,),
        (2, 1, 1, 325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, 1, 0, ),
        (1, 325, 475)
    ),
    (
        [DATA_DIR / "s_1_t_1_c_10_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_10_z_1.ome.tiff"],
        "T",
        (0, 1, ),
        (10, 1, 1736, 1776,)
    ),
    # Fails because too many operations provided
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, 1, 0, 0, 0, 0, 0, 0),
        None,
        marks=pytest.mark.raises(exception=IndexError)
    ),
    # Fails because "hello" isn't a valid slice operation
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, 1, 0, "hello", ),
        None,
        marks=pytest.mark.raises(exception=TypeError)
    ),
    # Fails because requested index is out of range (for list of images)
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "T",
        (0, 10, ),
        None,
        marks=pytest.mark.raises(exception=IndexError)
    ),
    # Fails because requested index is out of range (for actual data array)
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff"],
        "C",
        (0, 10, ),
        None,
        marks=pytest.mark.raises(exception=IndexError)
    ),
    # Fails because different overall shape changes
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_10_z_1.ome.tiff"],
        "T",
        (0, ),
        None,
        marks=pytest.mark.raises(exception=exceptions.InconsitentDataShapeException)
    ),
    # Fails because size of c changes
    pytest.param(
        [DATA_DIR / "s_1_t_1_c_1_z_1.ome.tiff", DATA_DIR / "s_1_t_1_c_10_z_1.ome.tiff"],
        "C",
        (0, ),
        None,
        marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError)
    )
])
def test_getitem(images, series_dim, selections, expected_shape):
    series = AICSSeries(images, series_dim)
    assert series[selections].shape == expected_shape

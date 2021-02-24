#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ome_types import to_xml
from ome_types.model import OME
import pytest
from typing import Callable, Optional, Tuple
import urllib

from aicsimageio import exceptions
from aicsimageio.readers import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter

from ..conftest import array_constructor, get_resource_write_full_path, LOCAL


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, expected_read_shape, expected_read_dim_order",
    [
        ((5, 16, 16), None, (1, 1, 5, 16, 16), "TCZYX"),
        ((5, 16, 16), "ZYX", (1, 1, 5, 16, 16), "TCZYX"),
        # OmeTiffReader is curently reordering dims to TCZYX always
        ((5, 16, 16), "CYX", (1, 5, 1, 16, 16), "TCZYX"),
        ((10, 5, 16, 16), "ZCYX", (1, 5, 10, 16, 16), "TCZYX"),
        ((5, 10, 16, 16), "CZYX", (1, 5, 10, 16, 16), "TCZYX"),
        ((16, 16), "YX", (1, 1, 1, 16, 16), "TCZYX"),
        pytest.param(
            (2, 3, 3),
            "AYX",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            (2, 3, 3),
            "YXZ",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            (2, 5, 16, 16),
            "CYX",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((1, 2, 3, 4, 5), None, (1, 2, 3, 4, 5), "TCZYX"),
        ((2, 3, 4, 5, 6), "TCZYX", (2, 3, 4, 5, 6), "TCZYX"),
        ((2, 3, 4, 5, 6), None, (2, 3, 4, 5, 6), "TCZYX"),
        ((1, 2, 3, 4, 5, 3), None, (1, 2, 3, 4, 5, 3), "TCZYXS"),
        # error 6D data doesn't work unless last dim is 3 or 4
        pytest.param(
            (1, 2, 3, 4, 5, 6),
            None,
            (1, 2, 3, 4, 5, 6),
            "TCZYXS",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        ((5, 16, 16, 3), "ZYXS", (1, 1, 5, 16, 16, 3), "TCZYXS"),
        ((5, 16, 16, 4), "CYXS", (1, 5, 1, 16, 16, 4), "TCZYXS"),
        ((3, 5, 16, 16, 4), "ZCYXS", (1, 5, 3, 16, 16, 4), "TCZYXS"),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_no_meta(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: Optional[str],
    expected_read_shape: Tuple[int, ...],
    expected_read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    OmeTiffWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert len(reader.scenes) == 1
    assert reader.shape == expected_read_shape
    assert reader.dims.order == expected_read_dim_order


@array_constructor
@pytest.mark.parametrize(
    "shape_to_create, ome_xml, expected_dim_order",
    [
        # ok dims
        (
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.uint8)])),
            "TCZYX",
        ),
        (
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.uint8)]),
            "TCZYX",
        ),
        # with RGB data:
        (
            (2, 2, 3, 4, 5, 3),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)])),
            "TCZYXS",
        ),
        (
            (2, 2, 3, 4, 5, 3),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)]),
            "TCZYXS",
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float)])),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float)]),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float)])),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float)]),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OME()),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            OME(),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # even more blatantly bad ome
        pytest.param(
            (1, 2, 3, 4, 5),
            "bad ome string",
            "TCZYX",
            # raised from within ome-types
            marks=pytest.mark.raises(exception=urllib.error.URLError),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_with_meta(
    array_constructor,
    shape_to_create,
    ome_xml,
    expected_dim_order,
    filename,
) -> None:
    # Create array
    arr = array_constructor(shape_to_create, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    OmeTiffWriter.save(arr, save_uri, dimension_order=None, ome_xml=ome_xml)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert len(reader.scenes) == 1
    assert reader.shape == shape_to_create
    assert reader.dims.order == expected_dim_order


@pytest.mark.parametrize(
    "array_data, write_dim_order, read_shapes, read_dim_order",
    [
        ([np.random.rand(5, 16, 16)], None, [(1, 1, 5, 16, 16)], ["TCZYX"]),
        (
            [np.random.rand(5, 16, 16), np.random.rand(4, 12, 12)],
            None,
            [(1, 1, 5, 16, 16), (1, 1, 4, 12, 12)],
            ["TCZYX", "TCZYX"],
        ),
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12, 3)],
            None,
            [(1, 5, 16, 16, 3), (1, 4, 12, 12, 3)],
            ["TCZYX", "TCZYX"],
        ),
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12, 3)],
            ["ZYXS", "CYXS"],
            [(1, 1, 5, 16, 16, 3), (1, 4, 1, 12, 12, 3)],
            ["TCZYXS", "TCZYXS"],
        ),
        (
            [np.random.rand(5, 16, 16), np.random.rand(4, 12, 12, 3)],
            ["ZYX", "CYXS"],
            [(1, 1, 5, 16, 16), (1, 4, 1, 12, 12, 3)],
            ["TCZYX", "TCZYXS"],
        ),
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12)],
            ["ZYXS", "CYX"],
            [(1, 1, 5, 16, 16, 3), (1, 4, 1, 12, 12)],
            ["TCZYXS", "TCZYX"],
        ),
        # two scenes but only one dimension order
        pytest.param(
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12)],
            ["ZYXS"],
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.ConflictingArgumentsError),
        ),
        # bad dims
        pytest.param(
            [np.random.rand(2, 3, 3)],
            "AYX",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_multiscene(
    array_data,
    write_dim_order,
    read_shapes,
    read_dim_order,
    filename,
) -> None:
    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    OmeTiffWriter.save(array_data, save_uri, write_dim_order)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert len(reader.scenes) == len(read_shapes)
    for i in range(len(reader.scenes)):
        reader.set_scene(reader.scenes[i])
        assert reader.shape == read_shapes[i]
        assert reader.dims.order == read_dim_order[i]

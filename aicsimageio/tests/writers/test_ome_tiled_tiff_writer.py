#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
from typing import Callable, Tuple, Union

import numpy as np
import pytest
from ome_types import OME, to_xml

from aicsimageio.readers import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers.bfio_writers import OmeTiledTiffWriter

from ..conftest import LOCAL, array_constructor, get_resource_write_full_path


@array_constructor
@pytest.mark.parametrize(
    "write_shape, expected_read_shape, expected_read_dim_order",
    [
        ((5, 16, 16), (1, 1, 5, 16, 16), "TCZYX"),
        ((5, 16, 16), (1, 1, 5, 16, 16), "TCZYX"),
        # OmeTiffReader is curently reordering dims to TCZYX always
        ((5, 1, 16, 16), (1, 5, 1, 16, 16), "TCZYX"),
        ((5, 10, 16, 16), (1, 5, 10, 16, 16), "TCZYX"),
        ((5, 10, 16, 16), (1, 5, 10, 16, 16), "TCZYX"),
        ((16, 16), (1, 1, 1, 16, 16), "TCZYX"),
        ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), "TCZYX"),
        ((2, 3, 4, 5, 6), (2, 3, 4, 5, 6), "TCZYX"),
        ((2, 3, 4, 5, 6), (2, 3, 4, 5, 6), "TCZYX"),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiled_tiff_writer_no_meta(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    expected_read_shape: Tuple[int, ...],
    expected_read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    OmeTiledTiffWriter.save(arr, save_uri)

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
        pytest.param(
            (2, 2, 3, 4, 5, 3),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)])),
            "TCZYXS",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            (2, 2, 3, 4, 5, 3),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)]),
            "TCZYXS",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float32)])),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float32)]),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float32)])),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float32)]),
            "TCZYX",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OME()),
            "TCZYX",
            marks=pytest.mark.raises(exception=IndexError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            OME(),
            "TCZYX",
            marks=pytest.mark.raises(exception=IndexError),
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
    array_constructor: Callable,
    shape_to_create: Tuple[int, ...],
    ome_xml: Union[str, OME, None],
    expected_dim_order: Tuple[int, ...],
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(shape_to_create, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Normal save
    OmeTiledTiffWriter.save(arr, save_uri, dimension_order=None, ome_xml=ome_xml)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert len(reader.scenes) == 1
    assert reader.shape == shape_to_create
    assert reader.dims.order == expected_dim_order

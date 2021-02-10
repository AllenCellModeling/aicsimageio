#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ome_types import to_xml
from ome_types.model import OME
import pytest
import urllib

from aicsimageio import exceptions
from aicsimageio.readers import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter

from ..conftest import array_constructor, get_resource_write_full_path, host


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape, read_dim_order",
    [
        ((5, 16, 16), None, (1, 1, 5, 16, 16), "TCZYX"),
        ((5, 16, 16), "ZYX", (1, 1, 5, 16, 16), "TCZYX"),
        # OmeTiffReader is curently reordering dims to TCZYX always
        ((5, 16, 16), "CYX", (1, 5, 1, 16, 16), "TCZYX"),
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
        ((1, 2, 3, 4, 5, 6), None, (2, 3, 4, 5, 6), "TCZYX"),
        ((5, 16, 16, 3), "ZYXS", (1, 1, 5, 16, 16, 3), "TCZYXS"),
        ((5, 16, 16, 4), "CYXS", (1, 5, 1, 16, 16, 4), "TCZYXS"),
    ],
)
@host
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_no_meta(
    array_constructor,
    host,
    write_shape,
    write_dim_order,
    read_shape,
    read_dim_order,
    filename,
):
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, host)

    # Normal save
    OmeTiffWriter.save(arr, save_uri, write_dim_order)
    OmeTiffWriter.save(arr, "C:\\Users\\dmt\\test.ome.tiff", write_dim_order)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert reader.shape == read_shape
    assert reader.dims.order == read_dim_order


@array_constructor
@pytest.mark.parametrize(
    "ome_xml",
    [
        # ok dims
        (to_xml(OmeTiffWriter.build_ome((1, 2, 3, 4, 5), np.dtype(np.uint8)))),
        (OmeTiffWriter.build_ome((1, 2, 3, 4, 5), np.dtype(np.uint8))),
        # wrong dtype
        pytest.param(
            to_xml(OmeTiffWriter.build_ome((1, 2, 3, 4, 5), np.dtype(np.float))),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            OmeTiffWriter.build_ome((1, 2, 3, 4, 5), np.dtype(np.float)),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # wrong dims
        pytest.param(
            to_xml(OmeTiffWriter.build_ome((2, 2, 3, 4, 5), np.dtype(np.float))),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            OmeTiffWriter.build_ome((2, 2, 3, 4, 5), np.dtype(np.float)),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            to_xml(OME()),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            OME(),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # even more blatantly bad ome
        pytest.param(
            "bad ome string",
            # raised from within ome-types
            marks=pytest.mark.raises(exception=urllib.error.URLError),
        ),
    ],
)
@host
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_with_meta(
    array_constructor,
    host,
    ome_xml,
    filename,
):
    # Create array
    arr = array_constructor((1, 2, 3, 4, 5), dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, host)

    # Normal save
    OmeTiffWriter.save(arr, save_uri, dimension_order=None, ome_xml=ome_xml)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert reader.shape == (1, 2, 3, 4, 5)
    assert reader.dims.order == "TCZYX"

    # Can't do "easy" testing because compression + shape mismatches on RGB data

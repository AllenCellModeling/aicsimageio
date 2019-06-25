#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from aicsimageio import exceptions
from aicsimageio.readers.ndarray_reader import NdArrayReader


@pytest.mark.parametrize("arr, expected_dims", [
    (np.ones((1, 1)), "YX"),
    (np.ones((1, 1, 1)), "ZYX"),
    (np.ones((1, 1, 1, 1)), "CZYX")
])
def test_ndarray_reader_get_default_dims(arr, expected_dims):
    # Open
    with NdArrayReader(arr) as r:
        assert r.dims == expected_dims
        assert r.metadata is None
        assert NdArrayReader.is_this_type(arr)


@pytest.mark.parametrize("expected", [
    "XYC",
    "STC",
    pytest.param("HELLOWORLD", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError)),
    pytest.param("NO", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError))
])
def test_default_reader_set_dims(expected):
    with NdArrayReader(np.ones((1, 1, 1))) as r:
        r.dims = expected
        assert r.dims == expected

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from aicsimageio.plugins.napari import utils

from ...test_aics_image import (BIG_CZI_FILE, BIG_OME_FILE, CZI_FILE, GIF_FILE,
                                MED_TIF_FILE, OME_FILE, PNG_FILE, TIF_FILE)


@pytest.mark.parametrize("filename, compute, expected_dtype, expected_shape, expected_channel_axis", [
    (PNG_FILE, True, np.ndarray, (1, 1, 1), None),
    (GIF_FILE, True, np.ndarray, (1, 1, 1), None),
    (CZI_FILE, True, np.ndarray, (1, 325, 475), None),
    (CZI_FILE, False, da.core.Array, (1, 325, 475), None),
    (OME_FILE, True, np.ndarray, (1, 325, 475), None),
    (OME_FILE, False, da.core.Array, (1, 325, 475), None),
    (TIF_FILE, True, np.ndarray, (1, 325, 475), None),
    (TIF_FILE, False, da.core.Array, (1, 325, 475), None),
    ([CZI_FILE, CZI_FILE], True, np.ndarray, (2, 325, 475), None),
    ([CZI_FILE, CZI_FILE], False, da.core.Array, (2, 325, 475), None),
    (MED_TIF_FILE, False, da.core.Array, (1, 10, 3, 325, 475), None),
    (BIG_CZI_FILE, False, da.core.Array, (1, 3, 3, 5, 325, 475), None),
    (BIG_OME_FILE, False, da.core.Array, (1, 3, 5, 3, 325, 475), None),
])
def test_reader(resources_dir, filename, compute, expected_dtype, expected_shape, expected_channel_axis):
    # Append filename(s) to resources dir
    if isinstance(filename, str):
        path = str(resources_dir / filename)
    else:
        path = [str(resources_dir / _path) for _path in filename]

    # Get reader
    reader = utils.get_reader(path, compute, processes=False)

    # Check callable
    assert callable(reader)

    # Get data
    layer_data = reader(path)

    # We only return one layer
    data, _ = layer_data[0]

    # Check layer data
    assert isinstance(data, expected_dtype)
    assert data.shape == expected_shape

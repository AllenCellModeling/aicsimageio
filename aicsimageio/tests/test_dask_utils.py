#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pytest
from distributed import get_client

from .test_aics_image import CZI_FILE, OME_FILE, PNG_FILE, TIF_FILE


def test_aicsimage_context_manager(resources_dir):
    from aicsimageio import AICSImage

    # Ensure that no dask cluster or client is available before
    with pytest.raises(ValueError):
        get_client()

    # Load the image in a context manager that spawn and closes a cluster and client
    # Processes = False informs dask to use threads instead of processes
    # We must use threads here to make sure we can properly run codecov
    with AICSImage(resources_dir / "s_3_t_1_c_3_z_5.czi", dask_kwargs={"processes": False}) as image:
        assert get_client() is not None
        assert image.data.shape == (3, 1, 3, 5, 325, 475)

    # Ensure that no dask cluster or client is available after
    with pytest.raises(ValueError):
        get_client()


def test_reader_context_manager(resources_dir):
    from aicsimageio.readers import CziReader

    # Ensure that no dask cluster or client is available before
    with pytest.raises(ValueError):
        get_client()

    # Load the image in a context manager that spawn and closes a cluster and client
    # Processes = False informs dask to use threads instead of processes
    # We must use threads here to make sure we can properly run codecov
    with CziReader(resources_dir / "s_3_t_1_c_3_z_5.czi", dask_kwargs={"processes": False}) as reader:
        assert get_client() is not None
        assert reader.data.shape == (1, 3, 3, 5, 325, 475)

    # Ensure that no dask cluster or client is available after
    with pytest.raises(ValueError):
        get_client()


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        (PNG_FILE, (1, 1, 4, 1, 800, 537)),
        (TIF_FILE, (1, 1, 1, 1, 325, 475)),
        (OME_FILE, (1, 1, 1, 1, 325, 475)),
        (CZI_FILE, (1, 1, 1, 1, 325, 475)),
    ],
)
def test_aicsimageio_no_networking(resources_dir, filename, expected_shape):
    # This should test and make sure that distributed isn't imported when aicsimageio is
    # Importing distributed on a machine (or container) that doesn't have any networking capabilities
    # results in socket errors, _during the import_
    # See: https://github.com/AllenCellModeling/aicsimageio/issues/82
    if "distributed" in sys.modules:
        del sys.modules["distributed"]

    # Re import
    import aicsimageio  # noqa: F401

    # Some basic operation to ensure that distributed is not imported anywhere down the line
    img = aicsimageio.AICSImage(resources_dir / filename)
    assert img.data.shape == expected_shape

    # Assert not imported
    assert "distributed" not in sys.modules

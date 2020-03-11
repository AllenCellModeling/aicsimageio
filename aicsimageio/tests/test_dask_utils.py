#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from unittest import mock

import pytest

from .test_aics_image import CZI_FILE, OME_FILE, PNG_FILE, TIF_FILE


@pytest.mark.parametrize("address, nworkers", [
    (None, None),
    (None, 2),
    ("tcp://some-spawned-cluster", None),
])
def test_aicsimage_context_manager(resources_dir, address, nworkers):
    from aicsimageio import AICSImage

    # Patch the spawn function
    with mock.patch("aicsimageio.dask_utils.spawn_cluster_and_client") as mocked_spawner:
        return "a", "b"

        # Patch the shutdown function
        with mock.patch("aicsimageio.dask_utils.shutdown_cluster_and_client") as mocked_shutdown:
            return "a", "b"

            # Load the image in a context manager that spawn and closes a cluster and client
            with AICSImage(resources_dir / "s_3_t_1_c_3_z_5.czi", address=address, nworkers=nworkers):
                assert mocked_spawner.call_args[0][0] == address
                assert mocked_spawner.call_arg[1]["nworkers"] == nworkers

            # Check that the cluster and client were scheduled to shutdown
            assert mocked_shutdown.called


@pytest.mark.parametrize("address, nworkers", [
    (None, None),
    (None, 2),
    ("tcp://some-spawned-cluster", None),
])
def test_reader_context_manager(resources_dir, address, nworkers):
    from aicsimageio.readers import CziReader

    # Patch the spawn function
    with mock.patch("aicsimageio.dask_utils.spawn_cluster_and_client") as mocked_spawner:
        return "a", "b"

        # Patch the shutdown function
        with mock.patch("aicsimageio.dask_utils.shutdown_cluster_and_client") as mocked_shutdown:
            return "a", "b"

            # Load the image in a context manager that spawn and closes a cluster and client
            with CziReader(resources_dir / "s_3_t_1_c_3_z_5.czi", address=address, nworkers=nworkers):
                assert mocked_spawner.call_args[0][0] == address
                assert mocked_spawner.call_arg[1]["nworkers"] == nworkers

            # Check that the cluster and client were scheduled to shutdown
            assert mocked_shutdown.called


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

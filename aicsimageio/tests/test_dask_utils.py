#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import mock

import pytest

from aicsimageio import AICSImage
from aicsimageio.readers import CziReader


@pytest.mark.parametrize("address, nworkers", [
    (None, None),
    (None, 2),
    ("tcp://some-spawned-cluster", None),
])
def test_aicsimage_context_manager(resources_dir, address, nworkers):
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

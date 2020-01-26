#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import pytest

from aicsimageio import AICSImage
from aicsimageio.readers import CziReader


@pytest.mark.parametrize("address, nworkers", [
    (None, None),
    (None, 4),
    pytest.param("tcp://this.will.fail", None, marks=pytest.mark.raises(exception=ValueError))
])
def test_aicsimage_context_manager(resources_dir, address, nworkers):
    # Load the image in a context manager that spawn and closes a cluster and client
    with AICSImage(resources_dir / "s_3_t_1_c_3_z_5.czi", address=address, nworkers=nworkers) as img:
        # We can assert this because by the time we get to this check the cases where we provided a address will have
        # already failed
        assert "tcp://127.0.0.1" in img.client.scheduler.address

        # Check that kwargs got passed down
        if nworkers is not None:
            assert len(img.cluster.workers) == nworkers
        else:
            assert len(img.cluster.workers) >= 1

    # Check that the cluster and client were shutdown after exiting context manager
    assert img.cluster.status == "closed"
    assert img.client.status == "closed"

    # Give some time to the system to fully shutdown the cluster and workers prior to next test
    time.sleep(3)


@pytest.mark.parametrize("address, nworkers", [
    (None, None),
    (None, 4),
    pytest.param("tcp://this.will.fail", None, marks=pytest.mark.raises(exception=ValueError))
])
def test_reader_context_manager(resources_dir, address, nworkers):
    # Load the image in a context manager that spawn and closes a cluster and client
    with CziReader(resources_dir / "s_3_t_1_c_3_z_5.czi", address=address, nworkers=nworkers) as img:
        # We can assert this because by the time we get to this check the cases where we provided a address will have
        # already failed
        assert "tcp://127.0.0.1" in img.client.scheduler.address

        # Check that kwargs got passed down
        if nworkers is not None:
            assert len(img.cluster.workers) == nworkers
        else:
            assert len(img.cluster.workers) >= 1

    # Check that the cluster and client were shutdown after exiting context manager
    assert img.cluster.status == "closed"
    assert img.client.status == "closed"

    # Give some time to the system to fully shutdown the cluster and workers prior to next test
    time.sleep(10)

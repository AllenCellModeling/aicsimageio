#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from contextlib import contextmanager
from typing import Optional, Tuple

from distributed import Client, LocalCluster

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def spawn_cluster_and_client(
    address: Optional[str] = None,
    **kwargs
) -> Tuple[Optional[LocalCluster], Optional[Client]]:
    """
    If provided an address, create a Dask Client connection.
    If not provided an address, create a LocalCluster and Client connection.
    If not provided an address, other Dask kwargs are accepted and passed down to the LocalCluster object.

    Notes
    -----
    When using this function, the processing machine or container must have networking capabilities enabled to
    function properly.
    """
    cluster = None
    if address is not None:
        client = Client(address)
        log.info(f"Connected to Remote Dask Cluster: {client}")
    else:
        cluster = LocalCluster(**kwargs)
        client = Client(cluster)
        log.info(f"Connected to Local Dask Cluster: {client}")

    return cluster, client


def shutdown_cluster_and_client(
    cluster: Optional[LocalCluster],
    client: Optional[Client]
) -> Tuple[Optional[LocalCluster], Optional[Client]]:
    """
    Shutdown a cluster and client.

    Notes
    -----
    When using this function, the processing machine or container must have networking capabilities enabled to
    function properly.
    """
    if cluster is not None:
        cluster.close()
    if client is not None:
        client.shutdown()
        client.close()

    return cluster, client


@contextmanager
def cluster_and_client(address: Optional[str] = None, **kwargs):
    """
    If provided an address, create a Dask Client connection.
    If not provided an address, create a LocalCluster and Client connection.
    If not provided an address, other Dask kwargs are accepted and passed down to the LocalCluster object.

    These objects will only live for the duration of this context manager.

    Examples
    --------
    >>> with cluster_and_client() as (cluster, client):
    ...     img1 = AICSImage("1.tiff")
    ...     img2 = AICSImage("2.czi")
    ...     other processing

    Notes
    -----
    When using this context manager, the processing machine or container must have networking capabilities enabled to
    function properly.
    """
    try:
        cluster, client = spawn_cluster_and_client(address=address, **kwargs)
        yield cluster, client
    finally:
        shutdown_cluster_and_client(cluster=cluster, client=client)

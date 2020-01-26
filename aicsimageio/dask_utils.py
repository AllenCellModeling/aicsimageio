#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Optional, Tuple

from distributed import Client, LocalCluster

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@staticmethod
def spawn_cluster(address: Optional[str], **kwargs) -> Tuple[Optional[LocalCluster], Optional[Client]]:
    """
    If provided an address, create a Dask Client connection.
    If not provided an address, create a LocalCluster and Client connection.
    If not provided an address, other Dask kwargs are accepted and passed down to the LocalCluster object.
    """
    if address is not None:
        client = Client(address)
        log.info(f"Connected to Remote Dask Cluster: {client}")
    else:
        cluster = LocalCluster(**kwargs)
        client = Client(cluster)
        log.info(f"Connected to Local Dask Cluster: {client}")

    return cluster, client


@staticmethod
def shutdown_cluster_and_client(
    cluster: Optional[LocalCluster],
    client: Optional[Client]
) -> Tuple[Optional[LocalCluster], Optional[Client]]:
    """
    Shutdown a cluster and client.
    """
    if cluster is not None:
        cluster.close()
    if client is not None:
        client.close()

    return cluster, client

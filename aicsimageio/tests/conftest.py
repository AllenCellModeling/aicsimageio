#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

###############################################################################


LOCAL = "LOCAL"
REMOTE = "REMOTE"

LOCAL_RESOURCES_DIR = Path(__file__).parent / "resources"
REMOTE_RESOURCES_DIR = "s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources"  # noqa: E501


def get_resource_full_path(filename, host):
    if host is LOCAL:
        return LOCAL_RESOURCES_DIR / filename
    elif host is REMOTE:
        return f"{REMOTE_RESOURCES_DIR}/{filename}"

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest


@pytest.fixture
def local_resources_dir() -> str:
    return Path(__file__).parent / "resources"


@pytest.fixture
def remote_resources_dir() -> str:
    return "s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources"  # noqa: E501

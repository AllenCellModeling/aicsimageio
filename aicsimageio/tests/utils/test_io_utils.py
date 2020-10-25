#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.utils.io_utils import pathlike_to_fs

from ..conftest import LOCAL, REMOTE, get_resource_full_path


@pytest.mark.parametrize(
    "filename, host, enforce_exists",
    [
        ("example.txt", LOCAL, False),
        ("example.txt", REMOTE, False),
        ("does-not-exist.good", LOCAL, False),
        ("does-not-exist.good", REMOTE, False),
        pytest.param(
            "does-not-exist.bad",
            LOCAL,
            True,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
        pytest.param(
            "does-not-exist.bad",
            REMOTE,
            True,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
    ],
)
def test_pathlike_to_fs(filename, host, enforce_exists):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    pathlike_to_fs(uri, enforce_exists)

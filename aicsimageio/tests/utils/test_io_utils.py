#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.utils.io_utils import pathlike_to_fs

from ..conftest import LOCAL, REMOTE, get_resource_full_path


@pytest.mark.parametrize("host", [LOCAL, REMOTE])
@pytest.mark.parametrize(
    "filename, enforce_exists",
    [
        ("example.txt", False),
        ("example.txt", False),
        ("does-not-exist.good", False),
        ("does-not-exist.good", False),
        pytest.param(
            "does-not-exist.bad",
            True,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
        pytest.param(
            "does-not-exist.bad",
            True,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
    ],
)
def test_pathlike_to_fs(filename: str, host: str, enforce_exists: bool) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    pathlike_to_fs(uri, enforce_exists)

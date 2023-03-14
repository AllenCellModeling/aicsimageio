#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio import AICSImage, exceptions

from .conftest import LOCAL, get_resource_full_path


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            "example.txt",
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "does-not-exist-klafjjksdafkjl.bad",
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
    ],
)
def test_aicsimage(
    filename: str,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)
    AICSImage(uri)

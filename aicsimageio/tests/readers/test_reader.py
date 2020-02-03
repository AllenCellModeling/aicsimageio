#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.readers.reader import Reader


@pytest.mark.parametrize("filename", [
    ("example.png"),
    pytest.param(None, marks=pytest.mark.raises(exception=IsADirectoryError)),
    pytest.param(1, marks=pytest.mark.raises(exception=TypeError)),
    pytest.param("non_existent_file.random", marks=pytest.mark.raises(exception=FileNotFoundError)),
])
def test_resolve_image_path(resources_dir, filename):
    # Get file
    if isinstance(filename, str):
        f = resources_dir / filename
    elif filename is None:
        f = resources_dir
    else:
        f = filename

    # Test path resolution
    Reader._resolve_image_path(f)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.readers import OmeTiffReader
from urllib.error import HTTPError
from xmlschema.validators import (
    XMLSchemaChildrenValidationError,
    XMLSchemaValidationError,
)

from ..conftest import get_resource_full_path


@pytest.mark.parametrize(
    "filename",
    [
        "3500000719_100X_20170317_F08_P06.ome.tiff",
        "new-cfe-file.ome.tiff",
        "old-cfe-file.ome.tiff",
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_1_t_1_c_10_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        "actk.ome.tiff",
    ],
)
def test_ome_validates_after_cleaning(filename):
    filepath = get_resource_full_path(filename, "LOCAL")
    OmeTiffReader(filepath)


@pytest.mark.parametrize(
    "filename",
    [
        # Pipline 4 is valid, :tada:
        "3500000719_100X_20170317_F08_P06.ome.tiff",
        # Some of our test files are valid, :tada:
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        # A lot of our files aren't valid, :upside-down-smiley:
        # These files have invalid schema / layout
        pytest.param(
            "new-cfe-file.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaChildrenValidationError),
        ),
        pytest.param(
            "old-cfe-file.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaChildrenValidationError),
        ),
        pytest.param(
            "actk.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaValidationError),
        ),
        # This file has a namespace that doesn't exist
        pytest.param(
            "s_1_t_1_c_10_z_1.ome.tiff", marks=pytest.mark.raises(exception=HTTPError)
        ),
    ],
)
def test_known_errors_without_cleaning(filename):
    filepath = get_resource_full_path(filename, "LOCAL")
    OmeTiffReader(filepath, clean_metadata=False)

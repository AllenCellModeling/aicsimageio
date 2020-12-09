#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio import exceptions
from aicsimageio.readers import OmeTiffReader

from ..conftest import get_resource_full_path

@pytest.mark.parametrize("filename", [
    "3500000719_100X_20170317_F08_P06.ome.tiff",
    "new-cfe-file.ome.tiff",
    "old-cfe-file.ome.tiff",
    "s_1_t_1_c_1_z_1.ome.tiff",
    "s_1_t_1_c_10_z_1.ome.tiff",
    "s_3_t_1_c_3_z_5.ome.tiff",
    "actk.ome.tiff",
])
def test_ome_validates(filename):
    filepath = get_resource_full_path(filename, "LOCAL")

    r = OmeTiffReader(filepath)

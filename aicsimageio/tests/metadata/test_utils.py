#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from ome_types import OME

from aicsimageio import AICSImage

from ..conftest import LOCAL, get_resource_full_path

###############################################################################


@pytest.mark.parametrize(
    "filename",
    [
        # DefaultReader
        pytest.param(
            "example.png",
            marks=pytest.mark.raises(exception=NotImplementedError),
        ),
        # TiffReader
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            marks=pytest.mark.raises(exception=NotImplementedError),
        ),
        # OmeTiffReader
        "actk.ome.tiff",
        # LifReader
        pytest.param(
            "tiled.lif",
            marks=pytest.mark.raises(exception=NotImplementedError),
        ),
        # CziReader
        "s_1_t_1_c_1_z_1.czi",
        "RGB-8bit.czi",
    ],
)
def test_ome_metadata(filename: str) -> None:
    # Get full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init image
    img = AICSImage(uri)

    # Test the transform
    assert isinstance(img.ome_metadata, OME)

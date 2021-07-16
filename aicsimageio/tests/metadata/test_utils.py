#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from ome_types import OME

from aicsimageio import AICSImage

from ..conftest import get_resource_full_path, host

###############################################################################


@host
@pytest.mark.parametrize(
    "filename",
    [
        # DefaultReader
        pytest.param(
            "example.png",
            marks=pytest.mark.raises(execeptions=NotImplementedError),
        ),
        # TiffReader
        pytest.param(
            "s_1_t_10_c_3_z_1.tiff",
            marks=pytest.mark.raises(execeptions=NotImplementedError),
        ),
        # OmeTiffReader
        ("actk.ome.tiff"),
        # LifReader
        pytest.param(
            "tiled.lif",
            marks=pytest.mark.raises(execeptions=NotImplementedError),
        ),
        # CziReader
        pytest.param(
            "s_1_t_1_c_1_z_1.czi",
            marks=pytest.mark.raises(execeptions=NotImplementedError),
        ),
        pytest.param(
            "RGB-8bit.czi",
            marks=pytest.mark.raises(execeptions=NotImplementedError),
        ),
    ],
)
def test_ome_metadata(filename: str, host: str) -> None:
    # Get full filepath
    uri = get_resource_full_path(filename, host)

    # Init image
    img = AICSImage(uri)

    # Test the transform
    assert isinstance(img.ome_metadata, OME)

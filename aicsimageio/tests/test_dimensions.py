#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.dimensions import Dimensions


def test_dimensions_getitem() -> None:
    dims = Dimensions("TCZYX", (1, 4, 75, 624, 924))
    assert dims["T"] == (1,)
    assert dims["T", "C"] == (1, 4)

    # out of order indexing
    assert dims["C", "T", "Y"] == (4, 1, 624)

    with pytest.raises(IndexError):
        dims["blarg"]
    with pytest.raises(IndexError):
        dims["blarg", "nope"]
    with pytest.raises(TypeError):
        dims[0]  # type: ignore

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Collection, Tuple

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
        # Ironic we have to type ignore this because uhhh
        # we are testing a TypeError
        dims[0]  # type: ignore

    assert dims.T == 1
    assert dims.order == "TCZYX"


@pytest.mark.parametrize(
    "dims, shape",
    [
        (["Z", "Y", "X"], (70, 980, 980)),
        pytest.param(
            "ZYXS",
            (70, 980, 980),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            "YX",
            (70, 980, 980),
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_dimensions_mismatched_dims_len_and_shape_size(
    dims: Collection[str],
    shape: Tuple[int, ...],
) -> None:
    # Just check success
    assert Dimensions(dims, shape)


@pytest.mark.parametrize(
    "dims, shape",
    [
        (["Z", "Y", "X"], (70, 980, 980)),
        pytest.param(
            ["C", "ZY", "X"],
            (70, 980, 980),
            marks=pytest.mark.raises(exception=ValueError),
        ),
        pytest.param(
            ["YX"],
            (70, 980, 980),
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_dimensions_bad_iterable_of_characters(
    dims: Collection[str],
    shape: Tuple[int, ...],
) -> None:
    # Just check success
    assert Dimensions(dims, shape)

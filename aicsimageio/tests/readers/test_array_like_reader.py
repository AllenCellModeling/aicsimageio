#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from aicsimageio import AICSImage, dimensions, exceptions, types
from aicsimageio.readers import ArrayLikeReader

from ..image_container_test_utils import run_image_container_checks


@pytest.mark.parametrize(
    "image, "
    "dim_order, "
    "channel_names, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dims, "
    "expected_channel_names",
    [
        # Check no metadata provided 2D
        # these are really simple just None checks
        (np.random.rand(1, 1), None, None, "Image:0", ("Image:0",), (1, 1), "YX", None),
        (
            da.random.random((1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1),
            "YX",
            None,
        ),
        (
            xr.DataArray(np.random.rand(1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1),
            "YX",
            None,
        ),
        (
            xr.DataArray(da.random.random((1, 1))),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1),
            "YX",
            None,
        ),
        # Check no metadata provided 3D
        # these are really simply just None checks
        (
            np.random.rand(1, 1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            da.random.random((1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            xr.DataArray(da.random.random((1, 1, 1))),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        # Check no metadata provided 4D
        # these check that channel names are created for all
        # and specifically for xr that channel names are overwritten
        (
            np.random.rand(1, 1, 1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1),
            "CZYX",
            ["Channel:0:0"],
        ),
        (
            da.random.random((1, 1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1),
            "CZYX",
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1),
            "CZYX",
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(da.random.random((1, 1, 1, 1))),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1),
            "CZYX",
            ["Channel:0:0"],
        ),
        # Check no channel names provided 3D but dim_order forces channel dim
        # these check that channel names are created for all
        # and specifically for xr that channel names are overwritten
        (
            np.random.rand(1, 1, 1),
            "CYX",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            da.random.random((1, 1, 1)),
            "CYX",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1)),
            "CYX",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(da.random.random((1, 1, 1))),
            "CYX",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        # Test many scene, same dim_order, first scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            "CYX",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            "CYX",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            "CYX",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            "CYX",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "CYX",
            ["Channel:0:0"],
        ),
        # Test many scene, same dim_order, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        # Test many scene, same channel_names, first scene
        (
            [np.random.rand(2, 1, 1, 1), np.random.rand(2, 2, 2, 2)],
            None,
            ["A", "B"],
            "Image:0",
            ("Image:0", "Image:1"),
            (2, 1, 1, 1),
            "CZYX",
            ["A", "B"],
        ),
        (
            [da.random.random((2, 1, 1, 1)), da.random.random((2, 2, 2, 2))],
            None,
            ["A", "B"],
            "Image:0",
            ("Image:0", "Image:1"),
            (2, 1, 1, 1),
            "CZYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(np.random.rand(2, 1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2, 2)),
            ],
            None,
            ["A", "B"],
            "Image:0",
            ("Image:0", "Image:1"),
            (2, 1, 1, 1),
            "CZYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(da.random.random((2, 1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2, 2))),
            ],
            None,
            ["A", "B"],
            "Image:0",
            ("Image:0", "Image:1"),
            (2, 1, 1, 1),
            "CZYX",
            ["A", "B"],
        ),
        # Test many scene, same channel_names, second scene
        (
            [np.random.rand(2, 1, 1, 1), np.random.rand(2, 2, 2, 2)],
            None,
            ["A", "B"],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2, 2),
            "CZYX",
            ["A", "B"],
        ),
        (
            [da.random.random((2, 1, 1, 1)), da.random.random((2, 2, 2, 2))],
            None,
            ["A", "B"],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2, 2),
            "CZYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(np.random.rand(2, 1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2, 2)),
            ],
            None,
            ["A", "B"],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2, 2),
            "CZYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(da.random.random((2, 1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2, 2))),
            ],
            None,
            ["A", "B"],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2, 2),
            "CZYX",
            ["A", "B"],
        ),
        # Test many scene, different dim_order, first scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            [None, "CYX"],
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            [None, "CYX"],
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        # Test many scene, different dim_order, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            [None, "CYX"],
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            [None, "CYX"],
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["Channel:1:0", "Channel:1:1"],
        ),
        # Test many scene, different dim_order, different channel_names, first scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "ZYX",
            None,
        ),
        # Test many scene, different dim_order, different channel_names, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["A", "B"],
        ),
        (
            [da.random.random((1, 1, 1)), da.random.random((2, 2, 2))],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(da.random.random((1, 1, 1))),
                xr.DataArray(da.random.random((2, 2, 2))),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["A", "B"],
        ),
        # Test filled in xarray(s)
        # no metadata should be overwritten
        (
            xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "TYX",
            None,
        ),
        (
            xr.DataArray(
                np.random.rand(1, 1, 1), dims=list("CYX"), coords={"C": ["A"]}
            ),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "CYX",
            ["A"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1),
            "TYX",
            None,
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 2, 2),
            "CYX",
            ["A", "B"],
        ),
        # Test non-standard dimensions
        (
            np.random.rand(1, 1, 1),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ABD",
            None,
        ),
        (
            da.random.random((1, 1, 1)),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ABD",
            None,
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1)),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ABD",
            None,
        ),
        (
            xr.DataArray(da.random.random((1, 1, 1))),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1),
            "ABD",
            None,
        ),
        # Test that we can support many dimensions if dim_order is provided
        (
            np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0",),
            (1, 2, 3, 4, 5, 6, 7, 8),
            "ABCDEFGH",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((1, 2, 3, 4, 5, 6, 7, 8)),
            ],
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 2, 3, 4, 5, 6, 7, 8),
            "ABCDEFGH",
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((2, 3, 4, 5, 6, 7, 8, 9)),
            ],
            "ABCDEFGH",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (2, 3, 4, 5, 6, 7, 8, 9),
            "ABCDEFGH",
            ["Channel:1:0", "Channel:1:1", "Channel:1:2", "Channel:1:3"],
        ),
        # Test that we allow metadata kwargs to be passed as lists
        # even if they "expand" out to a single scene
        (
            np.random.rand(1, 1),
            ["AX"],
            None,
            "Image:0",
            ("Image:0",),
            (1, 1),
            "AX",
            None,
        ),
        (
            np.random.rand(1, 1, 1, 1),
            None,
            [["A"]],
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1),
            "CZYX",
            ["A"],
        ),
        # Test mismatching mapping of arrays to dim_order
        pytest.param(
            [np.random.rand(1, 1)],
            ["YX", "BAD"],
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        pytest.param(
            [np.random.rand(1, 1), np.random.rand(1, 1, 1, 1)],
            ["YX", "STILL", "BAD"],
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Test mismatching mapping of arrays to channel_names
        pytest.param(
            [np.random.rand(1, 1, 1, 1)],
            None,
            [["A"], ["B"]],
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        pytest.param(
            [np.random.rand(1, 1, 1, 1), (1, 1, 1, 1, 1)],
            None,
            [["A"], ["B"], ["C"]],
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Test dim string compare to dims shape
        pytest.param(
            np.random.rand(1, 1),
            "BAD",
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=ValueError),
        ),
        # Test channel names length to size of channel dim
        pytest.param(
            np.random.rand(1, 1, 1, 1),
            None,
            ["B", "A", "D"],
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=ValueError),
        ),
        # Test channel names provided when no channel dim present
        pytest.param(
            np.random.rand(1, 1, 1),
            None,
            ["B", "A", "D"],
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=ValueError),
        ),
        # Test that without dim_order and with more than five dims, it raises an error
        # Our guess dim order only support up to five dims
        pytest.param(
            np.random.rand(1, 2, 3, 4, 5, 6),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            "hello world",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_arraylike_reader(
    image: Union[types.MetaArrayLike, List[types.MetaArrayLike]],
    dim_order: Optional[Union[str, List[str]]],
    channel_names: Optional[Union[List[str], List[List[str]]]],
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dims: str,
    expected_channel_names: Optional[List[str]],
) -> None:
    # Init
    image_container = ArrayLikeReader(
        image, dim_order=dim_order, channel_names=channel_names
    )

    run_image_container_checks(
        image_container=image_container,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.float64),
        expected_dims_order=expected_dims,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(None, None, None),
        # we allow both None and Dict because the user can pass an already initialized
        # xarray DataArray which has metadata as a dict
        expected_metadata_type=(
            type(None),
            dict,
        ),
    )


@pytest.mark.parametrize(
    "image, "
    "dim_order, "
    "channel_names, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dims, "
    "expected_channel_names",
    [
        # Check no metadata provided 2D
        # these are really simple just None checks
        (
            np.random.rand(1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Check no metadata provided 4D
        # these check that channel names are created for all
        # and specifically for xr that channel names are overwritten
        (
            np.random.rand(1, 1, 1, 1),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(1, 1, 1, 1)),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Test many scene, same dim_order, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            "CYX",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1"],
        ),
        # Test many scene, different dim_order, different channel_names, second scene
        (
            [np.random.rand(1, 1, 1), np.random.rand(2, 2, 2)],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1)),
                xr.DataArray(np.random.rand(2, 2, 2)),
            ],
            [None, "CYX"],
            [None, ["A", "B"]],
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        # Test filled in xarray(s)
        # no metadata should be overwritten
        (
            xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(
                np.random.rand(1, 1, 1), dims=list("CYX"), coords={"C": ["A"]}
            ),
            None,
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            [
                xr.DataArray(np.random.rand(1, 1, 1), dims=list("TYX")),
                xr.DataArray(
                    np.random.rand(2, 2, 2), dims=list("CYX"), coords={"C": ["A", "B"]}
                ),
            ],
            None,
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 2, 1, 2, 2),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["A", "B"],
        ),
        # Test non-standard dimensions
        (
            np.random.rand(2, 2, 2),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        (
            xr.DataArray(np.random.rand(2, 2, 2)),
            "ABD",
            None,
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
        ),
        # Test that we can support many dimensions if dims is provided
        (
            np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0",),
            (1, 3, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((1, 2, 3, 4, 5, 6, 7, 8)),
            ],
            "ABCDEFGH",
            None,
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 3, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2"],
        ),
        (
            [
                np.random.rand(1, 2, 3, 4, 5, 6, 7, 8),
                da.random.random((2, 3, 4, 5, 6, 7, 8, 9)),
            ],
            "ABCDEFGH",
            None,
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 4, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0", "Channel:1:1", "Channel:1:2", "Channel:1:3"],
        ),
        # Test that without dims and with more than five dims, it raises an error
        # Our guess dim order only support up to five dims
        pytest.param(
            np.random.rand(1, 2, 3, 4, 5, 6),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            "hello world",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_aicsimage_from_array(
    image: Union[types.MetaArrayLike, List[types.MetaArrayLike]],
    dim_order: Optional[str],
    channel_names: Optional[List[str]],
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dims: str,
    expected_channel_names: List[str],
) -> None:
    # Init
    image_container = AICSImage(image, dim_order=dim_order, channel_names=channel_names)

    run_image_container_checks(
        image_container=image_container,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.float64),
        expected_dims_order=expected_dims,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(None, None, None),
        # we allow both None and Dict because the user can pass an already initialized
        # xarray DataArray which has metadata as a dict
        expected_metadata_type=(
            type(None),
            dict,
        ),
    )


@pytest.fixture
def arange_data() -> list[xr.DataArray]:
    dims = list("STCZYX")
    shape = (3, 4, 5, 6, 7, 8)
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)
    coords = {
        "S": [f"Image:{i}" for i in range(shape[0])],
        "C": ["bf", "dapi", "gfp", "yfp", "dsred"],
    }
    return [arr.drop_vars("S") for arr in xr.DataArray(data, dims=dims, coords=coords)]


def test_get_stack_defaults(arange_data: list[xr.DataArray]) -> None:
    alr = ArrayLikeReader(arange_data)
    data = xr.concat(arange_data, dim="I")

    # numpy
    stack = alr.get_stack()
    assert np.allclose(stack, data)

    # dask
    dask_stack = alr.get_dask_stack()
    assert np.allclose(dask_stack, data).compute()

    # xarray dask
    xarray_dask_stack = alr.get_xarray_dask_stack()
    xr.testing.assert_allclose(xarray_dask_stack, data)

    # xarray numpy
    xarray_stack = alr.get_xarray_stack()
    xr.testing.assert_allclose(xarray_stack, data)


def test_get_stack_different_shape(arange_data: list[xr.DataArray]) -> None:
    arange_data[1] = arange_data[1][:3, :4, :5, :6]
    alr = ArrayLikeReader(arange_data)
    data = xr.concat(arange_data[::2], dim="I")

    with pytest.raises(exceptions.UnexpectedShapeError):
        alr.get_stack()

    stack = alr.get_stack(drop_non_matching_scenes=True)
    np.testing.assert_allclose(data.data, stack)

    stack = alr.get_stack(select_scenes=(0, 2))
    np.testing.assert_allclose(data.data, stack)

    stack = alr.get_xarray_stack(drop_non_matching_scenes=True)
    xr.testing.assert_allclose(data, stack)


def test_get_stack_different_dtype(arange_data: list[xr.DataArray]) -> None:
    arange_data[1] = arange_data[1].astype("int64")
    alr = ArrayLikeReader(arange_data)
    data = xr.concat(arange_data[::2], dim="I")

    with pytest.raises(TypeError):
        alr.get_stack()

    stack = alr.get_stack(drop_non_matching_scenes=True)
    np.testing.assert_allclose(stack, data.data)

    stack = alr.get_stack(select_scenes=(0, 2))
    np.testing.assert_allclose(stack, data.data)

    stack = alr.get_xarray_stack(drop_non_matching_scenes=True)
    xr.testing.assert_allclose(stack, data)


def test_get_stack_scene_coords(arange_data: list[xr.DataArray]) -> None:
    alr = ArrayLikeReader(arange_data)
    data = xr.concat(arange_data, dim="I")
    data = data.assign_coords({"I": [f"Image:{i}" for i in range(3)]})

    stack = alr.get_xarray_stack(scene_coord_values="names")
    xr.testing.assert_allclose(stack, data)

    # test different scene char
    stack = alr.get_xarray_stack(scene_coord_values="names", scene_character="S")
    xr.testing.assert_allclose(stack, data.rename({"I": "S"}))

    with pytest.raises(ValueError):
        stack = alr.get_xarray_stack(scene_coord_values="names", scene_character="T")

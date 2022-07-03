#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers.default_reader import DefaultReader
from aicsimageio.readers.reader import Reader

from ...conftest import LOCAL, get_resource_full_path, host
from ...image_container_test_utils import run_image_file_checks


@host
@pytest.mark.parametrize(
    "filename, set_scene, expected_shape, expected_dims_order",
    [
        ("example.bmp", "Image:0", (480, 640, 4), "YXS"),
        ("example.png", "Image:0", (800, 537, 4), "YXS"),
        ("example.jpg", "Image:0", (452, 400, 3), "YXS"),
        ("example.gif", "Image:0", (72, 268, 268, 4), "TYXS"),
        (
            "example_invalid_frame_count.mp4",
            "Image:0",
            (55, 1080, 1920, 3),
            "TYXS",
        ),
        (
            "example_valid_frame_count.mp4",
            "Image:0",
            (72, 272, 272, 3),
            "TYXS",
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.png",
            "Image:1",
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_default_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_shape: Tuple[int, ...],
    expected_dims_order: str,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=DefaultReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.uint8),
        expected_dims_order=expected_dims_order,
        expected_channel_names=None,
        expected_physical_pixel_sizes=(None, None, None),
        expected_metadata_type=dict,
    )


def test_ffmpeg_header_fail() -> None:
    with pytest.raises(IOError):
        # Big Buck Bunny
        DefaultReader("https://archive.org/embed/archive-video-files/test.mp4")


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_metadata_type",
    [
        (
            "example.png",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 800, 537, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (None, None, None),
            dict,
        ),
        (
            "example.gif",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 268, 268, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (None, None, None),
            dict,
        ),
        (
            "example_valid_frame_count.mp4",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 272, 272, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (None, None, None),
            dict,
        ),
    ],
)
def test_aicsimage(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_image_file_checks(
        ImageContainer=AICSImage,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "set_dims, "
    "set_channel_names, "
    "expected_dims, "
    "expected_channel_names, "
    "expected_shape",
    [
        # First check to show nothing changes
        (
            "example.gif",
            "Image:0",
            None,
            None,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0"],
            (72, 1, 1, 268, 268, 4),
        ),
        # Check just dims to see default channel name creation
        (
            "example.gif",
            "Image:0",
            "ZYXC",
            None,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3"],
            (1, 4, 72, 268, 268),
        ),
        # Check setting both as simple definitions
        (
            "example.gif",
            "Image:0",
            "ZYXC",
            ["Red", "Green", "Blue", "Alpha"],
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Red", "Green", "Blue", "Alpha"],
            (1, 4, 72, 268, 268),
        ),
        # Check providing too many dims
        pytest.param(
            "example.gif",
            "Image:0",
            "ABCDEFG",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check providing too many channels
        pytest.param(
            "example.gif",
            "Image:0",
            "ZYXC",
            ["A", "B", "C"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
        # Check providing channels but no channel dim
        pytest.param(
            "example.gif",
            "Image:0",
            None,
            ["A", "B", "C"],
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.ConflictingArgumentsError),
        ),
    ],
)
def test_set_coords(
    filename: str,
    set_scene: str,
    set_dims: Optional[Union[str, List[str]]],
    set_channel_names: Optional[Union[List[str], List[List[str]]]],
    expected_dims: str,
    expected_channel_names: List[str],
    expected_shape: Tuple[int, ...],
) -> None:
    # As a reminder, AICSImage always has certain dimensions
    # If you provide a dimension that isn't one of those,
    # it will only be available on the reader, not the AICSImage object.

    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri, dim_order=set_dims, channel_names=set_channel_names)

    # Set scene
    img.set_scene(set_scene)

    # Compare AICSImage results
    assert img.dims.order == expected_dims
    assert img.channel_names == expected_channel_names
    assert img.shape == expected_shape


@pytest.mark.parametrize(
    "filename, set_reader, extra_kwargs, expected_dims, expected_shape",
    [
        # See shape to see why you should use TiffReader :)
        (
            "actk.ome.tiff",
            DefaultReader,
            {},
            dimensions.DEFAULT_DIMENSION_ORDER,
            (390, 1, 1, 233, 345),
        ),
    ],
)
def test_set_reader(
    filename: str,
    set_reader: Type[Reader],
    extra_kwargs: Dict[str, Any],
    expected_dims: str,
    expected_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Init
    img = AICSImage(uri, reader=set_reader, **extra_kwargs)

    # Assert basics
    assert img.dims.order == expected_dims
    assert img.shape == expected_shape


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        ("example.png", (1, 1, 1, 800, 537, 4)),
    ],
)
def test_no_scene_prop_access(
    filename: str,
    expected_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Construct image and check no scene call with property access
    img = AICSImage(uri)
    assert img.shape == expected_shape

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Tuple, Union

import dask.array as da
import numpy as np
import pytest

###############################################################################


LOCAL = "LOCAL"
REMOTE = "REMOTE"

LOCAL_RESOURCES_DIR = Path(__file__).parent / "resources"
REMOTE_RESOURCES_DIR = "s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources"  # noqa: E501

LOCAL_RESOURCES_WRITE_DIR = Path(__file__).parent / "writer_products"
REMOTE_RESOURCES_WRITER_DIR = "s3://aics-modeling-packages-test-resources/fake/dir"


def get_resource_full_path(filename: str, host: str) -> Union[str, Path]:
    if host is LOCAL:
        return LOCAL_RESOURCES_DIR / filename

    return f"{REMOTE_RESOURCES_DIR}/{filename}"


def get_resource_write_full_path(filename: str, host: str) -> Union[str, Path]:
    if host is LOCAL:
        LOCAL_RESOURCES_WRITE_DIR.mkdir(parents=True, exist_ok=True)
        return LOCAL_RESOURCES_WRITE_DIR / filename

    return f"{REMOTE_RESOURCES_WRITER_DIR}/{filename}"


host = pytest.mark.parametrize("host", [LOCAL])


def np_random_from_shape(shape: Tuple[int, ...], **kwargs: Any) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(shape: Tuple[int, ...], **kwargs: Any) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor", [np_random_from_shape, da_random_from_shape]
)

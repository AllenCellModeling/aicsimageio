#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.reader import Reader

###############################################################################


def run_image_read_checks(
    ReaderClass: Reader,
    resources_dir: Path,
    filename: str,
    chunk_dims: List[str],
    select_scene: int,
    expected_shape: Tuple[int],
    expected_dims: str,
    expected_dtype: np.dtype,
):
    # Get file
    f = resources_dir / filename

    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Read file
    reader = ReaderClass(f, chunk_by_dims=chunk_dims, S=select_scene)

    # Check that there are no open file pointers after init
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert reader.dims == expected_dims
    assert reader.metadata
    assert reader.shape == expected_shape
    assert reader.dask_data.shape == expected_shape
    assert reader.get_size(expected_dims) == expected_shape
    assert reader.dtype == expected_dtype

    # Will error because those dimensions don't exist in the file
    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        assert reader.get_size("ABCDEFG") == expected_shape

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check array
    assert reader.data.shape == expected_shape

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]

    return reader

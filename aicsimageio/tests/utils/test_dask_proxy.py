from contextlib import contextmanager
from typing import Iterator

import dask.array as da
import numpy as np
import pytest

from aicsimageio.utils.dask_proxy import DaskArrayProxy

FILE_OPEN = False
OPEN_COUNT = 0


@contextmanager
def open_file() -> Iterator[None]:
    global FILE_OPEN
    global OPEN_COUNT
    FILE_OPEN = True
    OPEN_COUNT += 1
    try:
        yield
    finally:
        FILE_OPEN = False


@pytest.fixture
def dask_arr() -> da.Array:
    global OPEN_COUNT
    OPEN_COUNT = 0
    called = [0]

    def get_chunk() -> np.ndarray:
        if not FILE_OPEN:
            raise OSError("You didn't open the file!")
        nonlocal called
        called[0] += 1
        return np.random.rand(10, 10)[np.newaxis, np.newaxis]

    d = da.map_blocks(get_chunk, chunks=((1,) * 10, (1,) * 10, 10, 10), dtype=float)
    d.called = called
    return d


def test_array(dask_arr: da.Array) -> None:
    with pytest.raises(OSError):
        dask_arr.compute()

    assert OPEN_COUNT == 0

    with open_file():
        assert dask_arr.compute().shape == (10, 10, 10, 10)

    assert OPEN_COUNT == 1


def test_proxy(dask_arr: da.Array) -> None:
    with pytest.raises(OSError):
        dask_arr.compute()

    prox = DaskArrayProxy(dask_arr, open_file)

    assert OPEN_COUNT == 0
    assert prox.compute().shape == (10, 10, 10, 10)
    assert OPEN_COUNT == 1
    assert dask_arr.called[0] == 100

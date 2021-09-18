import warnings
from contextlib import contextmanager
from typing import Iterator, Tuple

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

    def get_chunk(block_id: Tuple[int, ...]) -> np.ndarray:
        if not FILE_OPEN:
            warnings.warn("You didn't open the file!")
        nonlocal called

        if not isinstance(block_id, np.ndarray):
            called[0] += 1
        return np.arange(100).reshape(10, 10)[np.newaxis, np.newaxis]

    d = da.map_blocks(get_chunk, chunks=((1,) * 10, (1,) * 10, 10, 10), dtype=float)
    d.called = called
    return d


@pytest.fixture
def proxy(dask_arr: da.Array) -> DaskArrayProxy:
    return DaskArrayProxy(dask_arr, open_file)


def test_array(dask_arr: da.Array) -> None:
    with pytest.warns(UserWarning):
        dask_arr.compute()

    assert OPEN_COUNT == 0

    with open_file():
        assert dask_arr.compute().shape == (10, 10, 10, 10)

    assert OPEN_COUNT == 1


def test_proxy_compute(proxy: DaskArrayProxy) -> None:
    assert OPEN_COUNT == 0
    ary = proxy.compute()
    assert isinstance(ary, np.ndarray)
    assert ary.shape == (10, 10, 10, 10)
    assert OPEN_COUNT == 1
    assert proxy.__wrapped__.called[0] == 100


def test_proxy_asarray(proxy: DaskArrayProxy) -> None:
    assert OPEN_COUNT == 0
    ary = np.asarray(proxy)
    assert isinstance(ary, np.ndarray)
    assert ary.shape == (10, 10, 10, 10)
    assert OPEN_COUNT == 1
    assert proxy.__wrapped__.called[0] == 100


def test_proxy_getitem(dask_arr: da.Array, proxy: DaskArrayProxy) -> None:
    a = dask_arr[0, 1:3]
    b = proxy[0, 1:3]
    assert isinstance(a, da.Array)
    assert isinstance(b, DaskArrayProxy)
    np.testing.assert_array_equal(a.compute(), b.compute())


def test_proxy_methods(dask_arr: da.Array, proxy: DaskArrayProxy) -> None:
    dmean = proxy.mean()
    assert isinstance(dmean, DaskArrayProxy)
    assert isinstance(dmean.compute(), float)
    assert dmean.compute() == dask_arr.mean().compute()

    # non array-returning methods don't return proxies
    assert isinstance(proxy.to_svg(), str)


def test_proxy_ufunc(dask_arr: da.Array, proxy: DaskArrayProxy) -> None:
    amean = np.mean(dask_arr)
    pmean = np.mean(proxy)
    assert isinstance(amean, da.Array)
    assert isinstance(pmean, DaskArrayProxy)
    assert amean.compute() == pmean.compute()


def test_proxy_repr(dask_arr: da.Array, proxy: DaskArrayProxy) -> None:
    assert repr(dask_arr) == repr(proxy)
    assert repr(dask_arr.mean) == repr(proxy.mean)

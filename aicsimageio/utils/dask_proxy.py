"""
experimental Dask array proxy for file IO
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

import dask.array as da
import numpy as np
from wrapt import ObjectProxy

if TYPE_CHECKING:
    from typing_extensions import Protocol

    # fmt: off
    class CheckableContext(Protocol):
        @property
        def closed(self) -> bool: ...  # noqa: E704
        def __enter__(self) -> CheckableContext: ...  # noqa: E704
        def __exit__(self, *a: Any) -> None: ...  # noqa: E704
    # fmt: on


class DaskArrayProxy(ObjectProxy):
    """Dask array wrapper that provides a 'file open' context when computing.

    This is necessary when the dask array contains a delayed underlying reader function
    that requires the file to be open. We don't want to open/close the file on
    every single chunk.  So this wrap the `compute` and `__array__` method in an
    open-file context manager.  For __getattr__, we return an `ArrayMethodProxy` that
    again wraps the resulting object in a DaskArrayProxy if it is a dask array.

    Experimental!
    The state held by the `file_ctx` may be problematic for dask distributed.

    Parameters
    ----------
    wrapped : da.Array
        the dask array that requires some file
    file_ctx : ContextManager
        A context in which the file is open.
        IMPORTANT: the context must be reusable, and preferably re-entrant:
        https://docs.python.org/3/library/contextlib.html#reentrant-context-managers
    """

    __wrapped__: da.Array

    def __init__(self, wrapped: da.Array, file_ctx: CheckableContext) -> None:
        super().__init__(wrapped)
        self._file_ctx = file_ctx

    def __getitem__(self, key: Any) -> DaskArrayProxy:
        return DaskArrayProxy(self.__wrapped__.__getitem__(key), self._file_ctx)

    def __getattr__(self, key: Any) -> Any:
        attr = getattr(self.__wrapped__, key)
        return _ArrayMethodProxy(attr, self._file_ctx) if callable(attr) else attr

    def __repr__(self) -> str:
        return repr(self.__wrapped__)

    def compute(self, **kwargs: Any) -> np.ndarray:
        with self._file_ctx if self._file_ctx.closed else nullcontext():  # type: ignore
            return self.__wrapped__.compute(**kwargs)

    def __array__(self, dtype: str = None, **kwargs: Any) -> np.ndarray:
        with self._file_ctx if self._file_ctx.closed else nullcontext():  # type: ignore
            return self.__wrapped__.__array__(dtype, **kwargs)


class _ArrayMethodProxy:
    """Wraps method on a dask array and returns a DaskArrayProxy if the result of the
    method is a dask array.  see details in DaskArrayProxy docstring."""

    def __init__(self, method: Callable, file_ctx: CheckableContext) -> None:
        self.method = method
        self._file_ctx = file_ctx

    def __repr__(self) -> str:
        return repr(self.method)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with self._file_ctx if self._file_ctx.closed else nullcontext():  # type: ignore
            result = self.method(*args, **kwds)

        if isinstance(result, da.Array):
            return DaskArrayProxy(result, self._file_ctx)
        return result

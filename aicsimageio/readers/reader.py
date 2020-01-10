#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import dask.array as da
import numpy as np

from .. import types
from ..constants import Dimensions


class Reader(ABC):
    _dask_data = None
    _data = None
    _dims = None
    _metadata = None

    def __init__(self, file: types.ImageLike, **kwargs):
        # This will both fully expand and enforce that the filepath exists
        file = Path(file).expanduser().resolve(strict=True)

        # This will check if the above enforced filepath is a directory
        if file.is_dir():
            raise IsADirectoryError(file)

        self._file = file

    @staticmethod
    def guess_dim_order(shape: Tuple[int]) -> str:
        return Dimensions.DefaultOrder[len(Dimensions.DefaultOrder) - len(shape):]

    @classmethod
    def is_this_type(cls, data: types.ImageLike) -> bool:
        # Check path
        if isinstance(data, (str, Path)):
            # This will both fully expand and enforce that the filepath exists
            f = Path(data).expanduser().resolve(strict=True)

            # This will check if the above enforced filepath is a directory
            if f.is_dir():
                raise IsADirectoryError(f)

            # Return and close the open pointer
            with open(f, "rb") as read_bytes:
                return cls._is_this_type(read_bytes)

        # Convert bytes to BytesIO
        if isinstance(data, bytes):
            data = io.BytesIO(data)

        # Check type
        if isinstance(data, io.BytesIO):
            return cls._is_this_type(data)

        # Special cases
        if isinstance(data, (np.ndarray, da.core.Array)):
            return cls._is_this_type(data)

        # Raise because none of the above returned
        raise TypeError(
                f"Reader only accepts types: [str, pathlib.Path, bytes, io.BytesIO, numpy or dask array]. "
                f"Received: {type(data)}"
            )

    @staticmethod
    @abstractmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        pass

    @property
    @abstractmethod
    def dask_data(self) -> da.core.Array:
        pass

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self.dask_data.compute()
        return self._data

    @property
    @abstractmethod
    def dims(self) -> str:
        pass

    @property
    @abstractmethod
    def metadata(self) -> Any:
        pass

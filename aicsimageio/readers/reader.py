#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import io
from pathlib import Path
from typing import Any, NamedTuple, Union

from .. import types


class LoadResults(NamedTuple):
    data: types.SixDArray
    dims: str
    metadata: Any


class Reader(ABC):

    @staticmethod
    def convert_to_bytes_io(file: Union[types.PathLike, types.BytesLike]) -> io.BytesIO:
        # Check path
        if isinstance(file, (str, Path)):
            # This will both fully expand and enforce that the filepath exists
            f = Path(file).expanduser().resolve(strict=True)

            # This will check if the above enforced filepath is a directory
            if f.is_dir():
                raise IsADirectoryError(f)

            # Convert to BytesIO
            with open(f, "rb") as read_in:
                return io.BytesIO(read_in)

        # Convert bytes
        elif isinstance(file, bytes):
            return io.BytesIO(file)

        # Set bytes
        elif isinstance(file, io.BytesIO):
            return file

        # Raise
        else:
            raise TypeError(
                f"Reader only accepts types: [str, pathlib.Path, bytes, io.BytesIO], recieved: {type(file)}"
            )

    def __init__(self, file: Union[types.PathLike, types.BytesLike]):
        # Lazy loaded
        self._bytes = None
        self._loaded_results = None

        # Convert to BytesIO
        self._bytes = self.convert_to_bytes_io(file)

    @property
    @abstractmethod
    def data(self) -> types.SixDArray:
        pass

    @abstractmethod
    @staticmethod
    def is_this_type(file: Union[types.PathLike, types.BytesLike]) -> bool:
        pass

    @property
    @abstractmethod
    def dims(self) -> str:
        pass

    @property
    @abstractmethod
    def metadata(self) -> Any:
        pass

    @abstractmethod
    def _load_from_bytes(self) -> LoadResults:
        pass

    def load(self) -> LoadResults:
        return LoadResults(self.data, self.dims, self.metadata)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._bytes.close()

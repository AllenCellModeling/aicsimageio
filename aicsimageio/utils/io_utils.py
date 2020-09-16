#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileOpener
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

from ..types import PathLike

###############################################################################

def pathlike_to_fs(
    uri: PathLike,
    enforce_exists: bool = False,
) -> Union[AbstractBufferedFile, LocalFileOpener]:
    """
    Find and return the appropriate filesystem and path from a path-like object.

    Parameters
    ----------
    uri: PathLike
        The local or remote path or uri.
    enforce_exists: bool
        Check whether or not the resource exists, if not, raise FileNotFoundError.

    Returns
    -------
    abstract_file: Union[AbstractBufferedFile, LocalFileOpener]
        A file like object to operate on.

    Raises
    ------
    FileNotFoundError: If enforce_exists is provided value True and the resource is not
    found or is unavailable.
    """
    # Convert paths to string to be handled by url_to_fs
    if isinstance(uri, Path):
        uri = str(uri)

    # Get details
    fs, path = url_to_fs(uri)

    # Check file exists
    if enforce_exists:
        if not fs.exists(path):
            raise FileNotFoundError(f"{fs.protocol}://{path}")

    # Get and store details
    return fs.open(path)

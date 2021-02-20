#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple

from fsspec.core import url_to_fs
from fsspec.spec import AbstractFileSystem

from ..types import PathLike

###############################################################################


def pathlike_to_fs(
    uri: PathLike,
    enforce_exists: bool = False,
) -> Tuple[AbstractFileSystem, str]:
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
    fs: AbstractFileSystem
        The filesystem to operate on.
    path: str
        The full path to the target resource.

    Raises
    ------
    FileNotFoundError
        If enforce_exists is provided value True and the resource is not found or is
        unavailable.
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
    # We do not return an AbstractBufferedFile (i.e. fs.open) as we do not want to have
    # any open file buffers _after_ any API call. API calls must themselves call
    # fs.open and complete their function during the context of the opened buffer.
    return fs, path

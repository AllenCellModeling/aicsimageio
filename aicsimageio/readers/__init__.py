#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Type

# To add a new reader add it both to TYPE_CHECKING and _READERS

if TYPE_CHECKING:
    from .array_like_reader import ArrayLikeReader  # noqa: F401
    from .bioformats_reader import BioformatsReader  # noqa: F401
    from .czi_reader import CziReader  # noqa: F401
    from .lif_reader import LifReader  # noqa: F401
    from .ome_tiff_reader import OmeTiffReader  # noqa: F401
    from .reader import Reader
    from .tiff_reader import TiffReader  # noqa: F401


# add ".relativepath.ClassName"
_READERS = (
    ".array_like_reader.ArrayLikeReader",
    ".bioformats_reader.BioformatsReader",
    ".czi_reader.CziReader",
    ".lif_reader.LifReader",
    ".ome_tiff_reader.OmeTiffReader",
    ".tiff_reader.TiffReader",
)
_LOOKUP = {k.rsplit(".", 1)[-1]: k for k in _READERS}
__all__ = list(_LOOKUP)


def __getattr__(name: str) -> Type[Reader]:
    if name in _LOOKUP:
        from importlib import import_module

        path, clsname = _LOOKUP[name].rsplit(".", 1)
        mod = import_module(path, __name__)
        return getattr(mod, clsname)
    raise ImportError(
        f"cannot import name {name!r} from {__name__!r}({__file__}). "
        f"Readers include: {__all__}"
    )

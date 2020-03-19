#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from pluggy import HookimplMarker

from . import utils

###############################################################################

napari_hook_implementation = HookimplMarker("napari")

###############################################################################


@napari_hook_implementation
def napari_get_reader(path: utils.PathLike) -> Optional[utils.ReaderFunction]:
    return utils.get_reader(path, compute=False)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr

from .dimensions import Dimensions
from .types import PhysicalPixelSizes

###############################################################################


class ImageContainer(ABC):
    @property
    @abstractmethod
    def scenes(self) -> Tuple[str, ...]:
        pass

    @property
    def current_scene(self) -> str:
        pass

    @property
    def current_scene_index(self) -> int:
        pass

    @abstractmethod
    def set_scene(self, scene_id: Union[str, int]) -> None:
        pass

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        pass

    @property
    def xarray_data(self) -> xr.DataArray:
        pass

    @property
    def dask_data(self) -> da.Array:
        pass

    @property
    def data(self) -> np.ndarray:
        pass

    @property
    def dtype(self) -> np.dtype:
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    def dims(self) -> Dimensions:
        pass

    @abstractmethod
    def get_image_dask_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> da.Array:
        pass

    @abstractmethod
    def get_image_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        pass

    @property
    def metadata(self) -> Any:
        pass

    @property
    def channel_names(self) -> Optional[List[str]]:
        pass

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        pass

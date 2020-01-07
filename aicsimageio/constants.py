#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Dimensions:
    B = "B"
    Scene = "S"
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"
    DefaultOrderList = [Scene, Time, Channel, SpatialZ, SpatialY, SpatialX]
    DefaultOrder = "".join(DefaultOrderList)

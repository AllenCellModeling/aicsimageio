#!/usr/bin/env/python


import os
import numpy as np
import xarray as xr
import tifffile as tiff


DATA_SHAPE = (2,3,4,5,6,7) # STCZYX

def test_glob_reader(): 
    data = np.arange(np.prod(DATA_SHAPE), dtype='uint16').reshape(DATA_SHAPE)
    try:
        os.mkdir("fake_data")
    except FileExistsError:
        pass

    for s in range(DATA_SHAPE[0]):
        for t in range(DATA_SHAPE[1]):
            for c in range(DATA_SHAPE[2]):
                for z in range(DATA_SHAPE[3]):
                    im = data[s,t,c,z]
                    tiff.imsave(f"fake_data/S{s}_T{t}_C{c}_Z{z}.tif", im, dtype=np.uint16)
                        

    dims = list("STCZYX")
    coords = {'C':[f'Channel:0:{i}' for i in range(DATA_SHAPE[2])]}
    x_data = xr.DataArray(data, dims=dims, coords=coords)

#! usr/env/bin/python
import os
import numpy as np
import xarray as xr
import tifffile as tiff
import aicsimageio
from pathlib import Path

TIFF_3D = False 
DATA_SHAPE = (2,3,4,5,6,7) # STCZYX


def make_fake_data_2d(path):

    data = np.arange(np.prod(DATA_SHAPE), dtype='uint16').reshape(DATA_SHAPE)

    dims = list("STCZYX")
    image_ids = [aicsimageio.metadata.utils.generate_ome_image_id(i) for i in range(DATA_SHAPE[0])]
    
    x_data = xr.DataArray(data, dims=dims)#, coords=coords)
    
    for s in range(DATA_SHAPE[0]):
        for t in range(DATA_SHAPE[1]):
            for c in range(DATA_SHAPE[2]):
                    for z in range(DATA_SHAPE[3]):
                        im = data[s,t,c,z]
                        tiff.imsave(str(path/f"S{s}_T{t}_C{c}_Z{z}.tif"), im, dtype=np.uint16)
    return x_data

def test_glob_reader_2d(tmp_path: Path):
    reference = make_fake_data_2d(tmp_path)
    gr = aicsimageio.readers.GlobReader(str(tmp_path/"*.tif"))

    for i, s in enumerate(gr.scenes):
        gr.set_scene(s)
        assert  np.all(reference.isel(S=i).data == gr.xarray_dask_data.data)
        assert  np.all(reference.isel(S=i).data == gr.xarray_data.data)



#    im = data[s,t,c]
#        tiff.imsave(f"fake_data/S{s}_T{t}_C{c}.tif", im, dtype=np.uint16)

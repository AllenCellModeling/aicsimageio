#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict

###############################################################################

# The order of the readers in this impl dict is important.
#
# Example:
# if TiffReader was placed before OmeTiffReader,
# we would never hit the OmeTiffReader

# Additionally while so many formats can be read by base-imageio
# Our custom reader may be more well-suited for interactions
# Example:
# DefaultReader supports LSM, and all the similar "Tiff-Like"
# formats, but TiffReader does as well and it has better
# dask chunking + metadata parsing than DefaultReader for those formats


FORMAT_IMPLEMENTATIONS: Dict[str, str] = {
    "array-like": "aicsimageio.readers.array_like_reader.ArrayLikeReader",
    "ome.tiff": "aicsimageio.readers.ome_tiff_reader.OmeTiffReader",
    "ome.tif": "aicsimageio.readers.ome_tiff_reader.OmeTiffReader",
    "tiff": "aicsimageio.readers.tiff_reader.TiffReader",
    "tif": "aicsimageio.readers.tiff_reader.TiffReader",
    "czi": "aicsimageio.readers.czi_reader.CziReader",
    "lif": "aicsimageio.readers.lif_reader.LifReader",
    # BASE-IMAGEIO FORMATS (with tifffile + non-existant removals)
    #
    # Pulled using:
    # from imageio import formats
    # routes = {}
    # for f in formats:
    #   for ext in f.extensions:
    #       routes[ext[1:]] = "aicsimageio.readers.default_reader.DefaultReader"
    #
    # for f in [
    #   "tif", "tiff", "stk", "lsm", "sgi", "nonexistentext", "foobar"
    # ]:
    #   routes.pop(f)
    #
    "bmp": "aicsimageio.readers.default_reader.DefaultReader",
    "bufr": "aicsimageio.readers.default_reader.DefaultReader",
    "cur": "aicsimageio.readers.default_reader.DefaultReader",
    "dcx": "aicsimageio.readers.default_reader.DefaultReader",
    "dds": "aicsimageio.readers.default_reader.DefaultReader",
    "ps": "aicsimageio.readers.default_reader.DefaultReader",
    "eps": "aicsimageio.readers.default_reader.DefaultReader",
    "fit": "aicsimageio.readers.default_reader.DefaultReader",
    "fits": "aicsimageio.readers.default_reader.DefaultReader",
    "fli": "aicsimageio.readers.default_reader.DefaultReader",
    "flc": "aicsimageio.readers.default_reader.DefaultReader",
    "fpx": "aicsimageio.readers.default_reader.DefaultReader",
    "ftc": "aicsimageio.readers.default_reader.DefaultReader",
    "ftu": "aicsimageio.readers.default_reader.DefaultReader",
    "gbr": "aicsimageio.readers.default_reader.DefaultReader",
    "gif": "aicsimageio.readers.default_reader.DefaultReader",
    "grib": "aicsimageio.readers.default_reader.DefaultReader",
    "h5": "aicsimageio.readers.default_reader.DefaultReader",
    "hdf": "aicsimageio.readers.default_reader.DefaultReader",
    "icns": "aicsimageio.readers.default_reader.DefaultReader",
    "ico": "aicsimageio.readers.default_reader.DefaultReader",
    "im": "aicsimageio.readers.default_reader.DefaultReader",
    "iim": "aicsimageio.readers.default_reader.DefaultReader",
    "jfif": "aicsimageio.readers.default_reader.DefaultReader",
    "jpe": "aicsimageio.readers.default_reader.DefaultReader",
    "jpg": "aicsimageio.readers.default_reader.DefaultReader",
    "jpeg": "aicsimageio.readers.default_reader.DefaultReader",
    "jp2": "aicsimageio.readers.default_reader.DefaultReader",
    "j2k": "aicsimageio.readers.default_reader.DefaultReader",
    "jpc": "aicsimageio.readers.default_reader.DefaultReader",
    "jpf": "aicsimageio.readers.default_reader.DefaultReader",
    "jpx": "aicsimageio.readers.default_reader.DefaultReader",
    "j2c": "aicsimageio.readers.default_reader.DefaultReader",
    "mic": "aicsimageio.readers.default_reader.DefaultReader",
    "mpo": "aicsimageio.readers.default_reader.DefaultReader",
    "msp": "aicsimageio.readers.default_reader.DefaultReader",
    "pcd": "aicsimageio.readers.default_reader.DefaultReader",
    "pcx": "aicsimageio.readers.default_reader.DefaultReader",
    "pxr": "aicsimageio.readers.default_reader.DefaultReader",
    "png": "aicsimageio.readers.default_reader.DefaultReader",
    "pbm": "aicsimageio.readers.default_reader.DefaultReader",
    "pgm": "aicsimageio.readers.default_reader.DefaultReader",
    "ppm": "aicsimageio.readers.default_reader.DefaultReader",
    "psd": "aicsimageio.readers.default_reader.DefaultReader",
    "bw": "aicsimageio.readers.default_reader.DefaultReader",
    "rgb": "aicsimageio.readers.default_reader.DefaultReader",
    "rgba": "aicsimageio.readers.default_reader.DefaultReader",
    "ras": "aicsimageio.readers.default_reader.DefaultReader",
    "tga": "aicsimageio.readers.default_reader.DefaultReader",
    "wmf": "aicsimageio.readers.default_reader.DefaultReader",
    "emf": "aicsimageio.readers.default_reader.DefaultReader",
    "xbm": "aicsimageio.readers.default_reader.DefaultReader",
    "xpm": "aicsimageio.readers.default_reader.DefaultReader",
    "cut": "aicsimageio.readers.default_reader.DefaultReader",
    "exr": "aicsimageio.readers.default_reader.DefaultReader",
    "g3": "aicsimageio.readers.default_reader.DefaultReader",
    "hdr": "aicsimageio.readers.default_reader.DefaultReader",
    "iff": "aicsimageio.readers.default_reader.DefaultReader",
    "lbm": "aicsimageio.readers.default_reader.DefaultReader",
    "jng": "aicsimageio.readers.default_reader.DefaultReader",
    "jif": "aicsimageio.readers.default_reader.DefaultReader",
    "jxr": "aicsimageio.readers.default_reader.DefaultReader",
    "wdp": "aicsimageio.readers.default_reader.DefaultReader",
    "hdp": "aicsimageio.readers.default_reader.DefaultReader",
    "koa": "aicsimageio.readers.default_reader.DefaultReader",
    "pfm": "aicsimageio.readers.default_reader.DefaultReader",
    "pct": "aicsimageio.readers.default_reader.DefaultReader",
    "pict": "aicsimageio.readers.default_reader.DefaultReader",
    "pic": "aicsimageio.readers.default_reader.DefaultReader",
    "3fr": "aicsimageio.readers.default_reader.DefaultReader",
    "arw": "aicsimageio.readers.default_reader.DefaultReader",
    "bay": "aicsimageio.readers.default_reader.DefaultReader",
    "bmq": "aicsimageio.readers.default_reader.DefaultReader",
    "cap": "aicsimageio.readers.default_reader.DefaultReader",
    "cine": "aicsimageio.readers.default_reader.DefaultReader",
    "cr2": "aicsimageio.readers.default_reader.DefaultReader",
    "crw": "aicsimageio.readers.default_reader.DefaultReader",
    "cs1": "aicsimageio.readers.default_reader.DefaultReader",
    "dc2": "aicsimageio.readers.default_reader.DefaultReader",
    "dcr": "aicsimageio.readers.default_reader.DefaultReader",
    "drf": "aicsimageio.readers.default_reader.DefaultReader",
    "dsc": "aicsimageio.readers.default_reader.DefaultReader",
    "dng": "aicsimageio.readers.default_reader.DefaultReader",
    "erf": "aicsimageio.readers.default_reader.DefaultReader",
    "fff": "aicsimageio.readers.default_reader.DefaultReader",
    "ia": "aicsimageio.readers.default_reader.DefaultReader",
    "iiq": "aicsimageio.readers.default_reader.DefaultReader",
    "k25": "aicsimageio.readers.default_reader.DefaultReader",
    "kc2": "aicsimageio.readers.default_reader.DefaultReader",
    "kdc": "aicsimageio.readers.default_reader.DefaultReader",
    "mdc": "aicsimageio.readers.default_reader.DefaultReader",
    "mef": "aicsimageio.readers.default_reader.DefaultReader",
    "mos": "aicsimageio.readers.default_reader.DefaultReader",
    "mrw": "aicsimageio.readers.default_reader.DefaultReader",
    "nef": "aicsimageio.readers.default_reader.DefaultReader",
    "nrw": "aicsimageio.readers.default_reader.DefaultReader",
    "orf": "aicsimageio.readers.default_reader.DefaultReader",
    "pef": "aicsimageio.readers.default_reader.DefaultReader",
    "ptx": "aicsimageio.readers.default_reader.DefaultReader",
    "pxn": "aicsimageio.readers.default_reader.DefaultReader",
    "qtk": "aicsimageio.readers.default_reader.DefaultReader",
    "raf": "aicsimageio.readers.default_reader.DefaultReader",
    "raw": "aicsimageio.readers.default_reader.DefaultReader",
    "rdc": "aicsimageio.readers.default_reader.DefaultReader",
    "rw2": "aicsimageio.readers.default_reader.DefaultReader",
    "rwl": "aicsimageio.readers.default_reader.DefaultReader",
    "rwz": "aicsimageio.readers.default_reader.DefaultReader",
    "sr2": "aicsimageio.readers.default_reader.DefaultReader",
    "srf": "aicsimageio.readers.default_reader.DefaultReader",
    "srw": "aicsimageio.readers.default_reader.DefaultReader",
    "sti": "aicsimageio.readers.default_reader.DefaultReader",
    "targa": "aicsimageio.readers.default_reader.DefaultReader",
    "wap": "aicsimageio.readers.default_reader.DefaultReader",
    "wbmp": "aicsimageio.readers.default_reader.DefaultReader",
    "wbm": "aicsimageio.readers.default_reader.DefaultReader",
    "webp": "aicsimageio.readers.default_reader.DefaultReader",
    "mov": "aicsimageio.readers.default_reader.DefaultReader",
    "avi": "aicsimageio.readers.default_reader.DefaultReader",
    "mpg": "aicsimageio.readers.default_reader.DefaultReader",
    "mpeg": "aicsimageio.readers.default_reader.DefaultReader",
    "mp4": "aicsimageio.readers.default_reader.DefaultReader",
    "mkv": "aicsimageio.readers.default_reader.DefaultReader",
    "wmv": "aicsimageio.readers.default_reader.DefaultReader",
    "bsdf": "aicsimageio.readers.default_reader.DefaultReader",
    "dcm": "aicsimageio.readers.default_reader.DefaultReader",
    "ct": "aicsimageio.readers.default_reader.DefaultReader",
    "mri": "aicsimageio.readers.default_reader.DefaultReader",
    "npz": "aicsimageio.readers.default_reader.DefaultReader",
    "swf": "aicsimageio.readers.default_reader.DefaultReader",
    "fts": "aicsimageio.readers.default_reader.DefaultReader",
    "fz": "aicsimageio.readers.default_reader.DefaultReader",
    "gipl": "aicsimageio.readers.default_reader.DefaultReader",
    "ipl": "aicsimageio.readers.default_reader.DefaultReader",
    "mha": "aicsimageio.readers.default_reader.DefaultReader",
    "mhd": "aicsimageio.readers.default_reader.DefaultReader",
    "nhdr": "aicsimageio.readers.default_reader.DefaultReader",
    "nia": "aicsimageio.readers.default_reader.DefaultReader",
    "nrrd": "aicsimageio.readers.default_reader.DefaultReader",
    "nii": "aicsimageio.readers.default_reader.DefaultReader",
    "niigz": "aicsimageio.readers.default_reader.DefaultReader",
    "img": "aicsimageio.readers.default_reader.DefaultReader",
    "imggz": "aicsimageio.readers.default_reader.DefaultReader",
    "vtk": "aicsimageio.readers.default_reader.DefaultReader",
    "hdf5": "aicsimageio.readers.default_reader.DefaultReader",
    "mnc": "aicsimageio.readers.default_reader.DefaultReader",
    "mnc2": "aicsimageio.readers.default_reader.DefaultReader",
    "mgh": "aicsimageio.readers.default_reader.DefaultReader",
    "dicom": "aicsimageio.readers.default_reader.DefaultReader",
    "gdcm": "aicsimageio.readers.default_reader.DefaultReader",
    "ecw": "aicsimageio.readers.default_reader.DefaultReader",
    "lfr": "aicsimageio.readers.default_reader.DefaultReader",
    "lfp": "aicsimageio.readers.default_reader.DefaultReader",
    "spe": "aicsimageio.readers.default_reader.DefaultReader",
    "nd2": "aicsimageio.readers.bioformats_reader.BioformatsReader",
}

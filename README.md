# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Docs/badge.svg)](https://allencellmodeling.github.io/aicsimageio)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aicsimageio)

A Python library for reading and writing image data with specific support for handling bio-formats.

---

## Features
* Supports reading metadata and imaging data for:
    * `CZI`
    * `OME-TIFF`
    * `TIFF`
    * Any additional format supported by `imageio`
* Supports writing metadata and imaging data for:
    * `OME-TIFF`

## Installation
**Stable Release:** `pip install aicsimageio`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/aicsimageio.git`

## Documentation
For full package documentation please visit
[allencellmodeling.github.io/aicsimageio](https://allencellmodeling.github.io/aicsimageio/index.html).

## Quick Start

### Full Image Reading
```python
from aicsimageio import AICSImage, imread

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.data  # returns 6D STCZYX numpy array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
img.size("STC")  # returns tuple of dimensions sizes for just STZ
img.get_image_data("CZYX", S=0, T=0)  # returns 4D CZYX numpy array

# Get 6D STCZYX numpy array
data = imread("my_file.tiff")
```

### Delayed Slice Image Reading
```python
from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.dask_data  # returns 6D STCZYX dask array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
img.size("STC")  # returns tuple of dimensions sizes for just STZ
img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array

# Get 6D STCZYX dask array
data = imread_dask("my_file.tiff")

# Read specified portion of dask array
lazy_s0t0 = img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array
s0t0 = lazy_s0t0.compute()  # returns 4D CZYX numpy array

# Or use normal numpy array slicing
lazy_s0t0 = data[0, 0, :]
s0t0 = lazy_s0t0.compute()
```

### Speed up IO and Processing with Dask Client and Clusters
```python
from aicsimageio import AICSImage, dask_utils

# Create a local dask cluster for the duration of the context manager
with AICSImage("filename.ome.tiff") as img:
    # do your work like normal

# Specify arguments for the local cluster initialization
with AICSImage("filename.ome.tiff", nworkers=4) as img:
    # do your work like normal

# Connect to a dask client for the duration of the context manager
with AICSImage("filename.ome.tiff", address="tcp://localhost:12345") as img:
    # do your work like normal

# Or spawn a cluster and client outside of a context manager
# This uses the same "address" and dask kwargs as above
# If you pass an address in, it will only shutdown the client and not the cluster
# as it is assumed that the cluster you are connecting to is shared between multiple people.
cluster, client = dask_utils.spawn_cluster_and_client()

img1 = AICSImage("1.tiff")
img2 = AICSImage("2.tiff")
img3 = AICSImage("3.tiff")

# And shut it down after
cluster, client = dask_utils.shutdown_cluster_and_client(cluster, client)
```

### Metadata Reading
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.metadata  # returns the metadata object for this image type
img.get_channel_names()  # returns a list of string channel names if found in the metadata
```

### Napari Interactive Viewer
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.view_napari()  # launches napari GUI and viewer
```

## Notes
* Image `data` and `dask_data` are always returned as six dimensional in dimension order `STCZYX`
or `Scene`, `Time`, `Channel`, `Z`, `Y`, and `X`.
* Each file format may use a different metadata parser it is dependent on the reader's implementation.
* The `AICSImage` object will only pull the `Scene`, `Time`, `Channel`, `Z`, `Y`, `X` dimensions from the reader.
If your file has dimensions outside of those, use the base reader classes `CziReader`, `OmeTiffReader`, `TiffReader`,
or `DefaultReader`.
* We make some choices for the user based off the image data during `img.view_napari`. If you don't want this
behavior, simply pass the `img.dask_data` into `napari.view_image` instead.

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

***Free software: BSD-3-Clause***

# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Documentation/badge.svg)](https://allencellmodeling.github.io/aicsimageio)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aicsimageio)

Delayed Parallel Image Reading for Microscopy Images in Python

---

## Features
* Supports reading metadata and imaging data for:
    * `CZI`
    * `OME-TIFF`
    * `TIFF`
    * `LIF`
    * Any additional format supported by [`imageio`](https://github.com/imageio/imageio)
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
img.get_image_data("CZYX", S=0, T=0)  # returns 4D CZYX numpy array

# Get 6D STCZYX numpy array
data = imread("my_file.tiff")
```

### Delayed Image Reading
```python
from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.dask_data  # returns 6D STCZYX dask array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
img.size("STC")  # returns tuple of dimensions sizes for just STC
img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array

# Read specified portion of dask array
lazy_s0t0 = img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array
s0t0 = lazy_s0t0.compute()  # returns 4D CZYX numpy array

# Or use normal numpy array slicing
lazy_data = imread_dask("my_file.tiff")
lazy_s0t0 = lazy_data[0, 0, :]
s0t0 = lazy_s0t0.compute()
```

#### Quick Start Notes
In short, if the word "dask" appears in the function or property name, the function
utilizes delayed reading. If not, the requested image will be loaded immediately and
the internal implementation may result in loading the entire image even if only a small
chunk was requested. Currently, `AICSImage.data` and `AICSImage.get_image_data` load
and cache the entire image in memory before performing their operation.
`AICSImage.dask_data` and `AICSImage.get_image_dask_data` do not load any image data
until the user calls `compute` on the `dask.Array` object and only the requested chunk
will be loaded into memory instead of the entire image.

### Speed up IO and Processing with Dask Clients and Clusters
If you have already spun up a `distributed.Client` object in your Python process or
your processing is running on a distributed worker, great, you will naturally gain IO
and processing gains. If you haven't done that or don't know what either of those are,
there are some utility functions to help construct and manage these for you.

```python
from aicsimageio import AICSImage, dask_utils

# Create a local dask cluster and client for the duration of the context manager
with AICSImage("filename.ome.tiff") as img:
    # do your work like normal
    print(img.dask_data.shape)

# Specify arguments for the local cluster initialization
with AICSImage("filename.ome.tiff", dask_kwargs={"nworkers": 4}) as img:
    # do your work like normal
    print(img.dask_data.shape)

# Connect to a dask client for the duration of the context manager
with AICSImage(
    "filename.ome.tiff",
    dask_kwargs={"address": "tcp://localhost:12345"}
) as img:
    # do your work like normal
    print(img.dask_data.shape)

# Or spawn a local cluster and / or connect to a client outside of a context manager
# This uses the same "address" and dask kwargs as above
# If you pass an address in, it will create and shutdown the client
# and no cluster will be created.
# Similar to AICSImage, these objects will be connected and useable
# for the lifespan of the context manager.
with dask_utils.cluster_and_client() as (cluster, client):

    img1 = AICSImage("1.tiff")
    img2 = AICSImage("2.tiff")
    img3 = AICSImage("3.tiff")

    # Do your image processing work
```

**Note:** The `AICSImage` context manager and the `dask_utils` module require that the
processing machine or container have networking capabilities enabled to function
properly.


### Metadata Reading
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.metadata  # returns the metadata object for this image type
img.get_channel_names()  # returns a list of string channel names found in the metadata
```

### Napari Interactive Viewer
[napari](https://github.com/Napari/napari) is a fast, interactive, multi-dimensional
image viewer for python and it is pretty useful for imaging data that this package
tends to interact with.
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.view_napari()  # launches napari GUI and viewer
```

We have also released
[napari-aicsimageio](https://github.com/AllenCellModeling/napari-aicsimageio), a plugin
that allows use of all the functionality described here, but in the `napari` default
viewer itself.


## Notes
* Image `data` and `dask_data` are always returned as six dimensional in dimension
order `STCZYX` or `Scene`, `Time`, `Channel`, `Z`, `Y`, and `X`.
* Each file format may use a different metadata parser it is dependent on the reader's
implementation.
* The `AICSImage` object will only pull the `Scene`, `Time`, `Channel`, `Z`, `Y`, `X`
dimensions from the reader.
If your file has dimensions outside of those, use the base reader classes `CziReader`,
`OmeTiffReader`, `TiffReader`, or `DefaultReader`.
* We make some choices for the user based off the image data during `img.view_napari`.
If you don't want this behavior, simply pass the `img.dask_data` into
`napari.view_image` instead.

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

_Free software: BSD-3-Clause_

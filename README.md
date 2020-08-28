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
img.data  # returns 5D TCZYX numpy array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.data  # returns 5D TCZYX numpy array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order

# Get 5D TCZYX numpy array
data = imread("my_file.tiff")
```

### Delayed Image Slice Reading
```python
from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.dask_data  # returns 5D TCZYX dask array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.dask_data  # returns 5D TCZYX dask array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Read specified portion of dask array
lazy_t0 = img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array
t0 = lazy_t0.compute()  # returns 4D CZYX numpy array

# Get a 5D TCZYX dask array
lazy_data = imread_dask("my_file.tiff")
lazy_t0 = lazy_data[0, :]
t0 = lazy_t0.compute()
```

### Speed up IO and Processing with Dask Clients and Clusters
If you have already spun up a `distributed.Client` object in your Python process or
your processing is running on a distributed worker, great, you will naturally gain IO
and processing gains. If you haven't done that or don't know what either of those are,
there are some utility functions to help construct and manage these for you.

```python
from aicsimageio import AICSImage, dask_utils

# Spawn a local cluster
# These objects will be connected and useable for the lifespan of the context manager.
with dask_utils.cluster_and_client() as (cluster, client):

    img1 = AICSImage("1.tiff")
    img2 = AICSImage("2.tiff")
    img3 = AICSImage("3.tiff")

    # Do your image processing work

# Connect to a remote cluster
# If you pass an address in, it will create and shutdown the client and no cluster will
# be created. These objects will be connected and usable for the lifespan of the
# context manager.
with dask_utils.cluster_and_client(address="tcp://localhost:1234") as (cluster, client):

    img1 = AICSImage("1.tiff")
    img2 = AICSImage("2.tiff")
    img3 = AICSImage("3.tiff")

    # Do your image processing work
```

**Note:** The `dask_utils` module require that the processing machine or container have
networking capabilities enabled to function properly.


### Metadata Reading
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.metadata  # returns the metadata object for this image type
img.channel_names  # returns a list of string channel names found in the metadata
```

### Napari Interactive Viewer
[napari](https://github.com/Napari/napari) is a fast, interactive, multi-dimensional
image viewer for python and it is pretty useful for imaging data that this package
tends to interact with. If you would like the distributed reading and delayed benefits
of `aicsimageio` while using `napari` please install
[`napari-aicsimageio`](https://github.com/AllenCellModeling/napari-aicsimageio).


## Performance Considerations
* **If your image fits into memory and you are not using a distributed cluster:** use
`AICSImage.data` or `Reader.data` which are generally optimal. You can also use this to
preload the data before using `get_image_data`.
* **If your image is too large to fit into memory:** use `AICSImage.get_image_data` to
get a `numpy` array or `AICSImage.get_image_dask_data` to get a `dask` array for a
specific chunk of data from the image.
* **If you are using a distributed cluster with more than ~6 workers:** all functions
and properties in the library are generally optimal.
* **If you are using a distributed cluster with less than ~6 workers:** use
`aicsimageio.use_dask(False)`. From our testing, 6 workers is the bare minimum for
read time reduction compared to no cluster usage.
* When using a `dask` array, it is important to know when to `compute` or
`persist` data and when to keep chaining computation.
[Here is a good rundown on the trade offs.](https://stackoverflow.com/questions/41806850/dask-difference-between-client-persist-and-client-compute#answer-41807160)


## Notes
* Image `data` and `dask_data` are always returned as five dimensional in dimension
order `TCZYX` or, `Time`, `Channel`, `Z`, `Y`, and `X`.
* Each file format may use a different metadata parser it is dependent on the reader's
implementation.
* The `AICSImage` object will only pull the `Scene`, `Time`, `Channel`, `Z`, `Y`, `X`
dimensions from the reader. If your file has dimensions outside of those, use the base
reader classes.

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

_Free software: BSD-3-Clause_

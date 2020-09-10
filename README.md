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
img = AICSImage("my_file.tiff")  # loads the first scene found
img.data  # returns 5D TCZYX numpy array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get current operating scene
img.current_scene

# Get valid scene ids from the file
img.scenes

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.data  # returns 5D TCZYX numpy array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get 5D TCZYX numpy array
data = imread("my_file.tiff")
```

### Delayed Image Reading
```python
from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # loads the first scene found
img.dask_data  # returns 5D TCZYX dask array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Get current operating scene
img.current_scene

# Get valid scene ids from the file
img.scenes

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.dask_data  # returns 5D TCZYX dask array
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Read specified portion of dask array
lazy_t0 = img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array
t0 = lazy_t0.compute()  # returns 4D CZYX numpy array

# Get a 5D TCZYX dask array
lazy_data = imread_dask("my_file.tiff")
lazy_t0 = lazy_data[0, :]
t0 = lazy_t0.compute()
```

#### Quick Start Notes
In short, if the word "dask" appears in the function or property name, the function
utilizes delayed reading, if not, the underlying operation is backed by the image fully
read into memory. I.E. `AICSImage.data` and `AICSImage.get_image_data` load the entire
image into memory before performing their operation, and `AICSImage.dask_data` and
`AICSImage.get_image_dask_data` do not load any image data until the user calls
`compute` on the `dask.Array` object and only the requested chunk will be loaded into
memory instead of the entire image.

### Metadata Reading
```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.metadata  # returns the metadata object for this image type
img.channel_names  # returns a list of string channel names found in the metadata
img.physical_pixel_size.Z  # returns the Z dimension pixel size as found in the metadata
img.physical_pixel_size.Y  # returns the Y dimension pixel size as found in the metadata
img.physical_pixel_size.X  # returns the X dimension pixel size as found in the metadata
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

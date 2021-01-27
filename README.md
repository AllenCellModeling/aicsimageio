# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Documentation/badge.svg)](https://allencellmodeling.github.io/aicsimageio)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aicsimageio)

Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure
Python

---

## Features

- Supports reading metadata and imaging data for:
  - `CZI`
  - `OME-TIFF`
  - `TIFF`
  - `LIF`
  - Any additional format supported by [imageio](https://github.com/imageio/imageio)
- Supports writing metadata and imaging data for:
  - `OME-TIFF`
- Supports reading from and writing to any
  [fsspec](https://github.com/intake/filesystem_spec) supported file system:
  _ Local paths (i.e. `my-file.png`)
  _ HTTP URLs (i.e. `https://my-domain.com/my-file.png`)
  _ [s3fs](https://github.com/dask/s3fs) (i.e. `s3://my-bucket/my-file.png`)
  _ [gcsfs](https://github.com/dask/gcsfs) (i.e. `gcs://my-bucket/my-file.png`) \* See the [list of known implementations](https://filesystem-spec.readthedocs.io/en/latest/?badge=latest#implementations).

## Installation

**Stable Release:** `pip install aicsimageio`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/aicsimageio.git`

## Documentation

For full package documentation please visit
[allencellmodeling.github.io/aicsimageio](https://allencellmodeling.github.io/aicsimageio/index.html).

## Quickstart

### Full Image Reading

```python
from aicsimageio import AICSImage, imread

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.data  # returns 5D TCZYX numpy array
img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get the id of the current operating scene
img.current_scene

# Get a list valid scene ids
img.scenes

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.data  # returns 5D TCZYX numpy array
img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get 5D TCZYX numpy array
data = imread("my_file.tiff")  # optionally provide a scene id, default first
```

### Delayed Image Reading

```python
from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.dask_data  # returns 5D TCZYX dask array
img.xarray_dask_data  # returns 5D TCZYX xarray data array backed by dask array
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Get the id of the current operating scene
img.current_scene

# Get a list valid scene ids
img.scenes

# Change scene
img.set_scene(1)

# Same operations on a different scene
img.dask_data  # returns 5D TCZYX dask array
img.xarray_dask_data  # returns 5D TCZYX xarray data array backed by dask array
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array

# Read a specified portion of dask array
lazy_t0 = img.get_image_dask_data("CZYX", T=0)  # returns 4D CZYX dask array
t0 = lazy_t0.compute()  # returns 4D CZYX numpy array

# Get a 5D TCZYX dask array
lazy_data = imread_dask("my_file.tiff")  # optionally provide a scene id, default first
lazy_t0 = lazy_data[0, :]
t0 = lazy_t0.compute()
```

### Remote Image Reading

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("http://my-website.com/my_file.tiff")
img = AICSImage("s3://my-bucket/my_file.tiff")
img = AICSImage("gcs://my-bucket/my_file.tiff")

# All other normal operations work just fine
```

### Metadata Reading

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.metadata  # returns the metadata object for this image type
img.channel_names  # returns a list of string channel names found in the metadata
img.physical_pixel_size.Z  # returns the Z dimension pixel size as found in the metadata
img.physical_pixel_size.Y  # returns the Y dimension pixel size as found in the metadata
img.physical_pixel_size.X  # returns the X dimension pixel size as found in the metadata
```

#### Quickstart Notes

In short, if the word "dask" appears in the function or property name, the function
utilizes delayed reading. If not, the requested image will be loaded immediately and
the internal implementation may result in loading the entire image even if only a small
chunk was requested. Currently, `AICSImage.data`, `AICSImage.xarray_data`, and
`AICSImage.get_image_data` load and cache the entire image in memory before performing
their operation. `AICSImage.dask_data`, `AICSImage.xarray_dask_data`, and
`AICSImage.get_image_dask_data` do not load any image data until the user calls
`compute` on the `dask.Array` object and only the requested chunk will be loaded into
memory instead of the entire image.

## Performance Considerations

- **The quickest read operation will always be `.data` on a local file.** All other
  operations come with _some_ minor overhead. We try to minimize this overhead wherever
  possible.
- **If your image fits in memory:** use `AICSImage.data`, `AICSImage.get_image_data`,
  or `Reader` equivalents.
- **If your image is too large to fit in memory:** use `AICSImage.dask_data`,
  `AICSImage.get_image_dask_data`, or `Reader` equivalents.
- **If your image does not support native chunk reading:** it may not be best to read
  chunks from a remote source. While possible, the format of the image matters a lot for
  chunked read performance.

## Benchmarks

AICSImageIO is benchmarked using [asv](https://asv.readthedocs.io/en/stable/).
You can find the benchmark results for every commit to `master` starting at the 4.0
release on our
[benchmarks page](https://AllenCellModeling.github.io/aicsimageio/_benchmarks/index.html).

## Napari Interactive Viewer

[napari](https://github.com/Napari/napari) is a fast, interactive, multi-dimensional
image viewer for python and it is pretty useful for imaging data that this package
tends to interact with.

We have also released
[napari-aicsimageio](https://github.com/AllenCellModeling/napari-aicsimageio), a plugin
that allows use of all the functionality described in this library, but in the `napari`
default viewer itself.

## Notes

- Each file format may use a different metadata parser as it is dependent on the
  format's reader class implementation.
- The `AICSImage` object will only pull the `Scene`, `Time`, `Channel`, `Z`, `Y`, `X`
  dimensions from the reader. If your file has dimensions outside of those, use the base
  `Reader` classes.

## Development

See our
[developer resources](https://allencellmodeling.github.io/aicsimageio/developer_resources)
for information related to developing the code.

_Free software: BSD-3-Clause_

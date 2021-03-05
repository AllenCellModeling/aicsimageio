# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Main/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/aicsimageio/)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aicsimageio)

Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python

---

## Features

-   Supports reading metadata and imaging data for:
    -   `CZI`
    -   `OME-TIFF`
    -   `TIFF`
    -   `LIF`
    -   Any additional format supported by [imageio](https://github.com/imageio/imageio)
        (`PNG`, `GIF`, etc.)
-   Supports writing metadata and imaging data for:
    -   `OME-TIFF`
-   Supports reading and writing to
    [fsspec](https://github.com/intake/filesystem_spec) supported file systems
    wherever possible:

    -   Local paths (i.e. `my-file.png`)
    -   HTTP URLs (i.e. `https://my-domain.com/my-file.png`)
    -   [s3fs](https://github.com/dask/s3fs) (i.e. `s3://my-bucket/my-file.png`)
    -   [gcsfs](https://github.com/dask/gcsfs) (i.e. `gcs://my-bucket/my-file.png`)

    See the [list of known implementations](https://filesystem-spec.readthedocs.io/en/latest/?badge=latest#implementations).

## Installation

**Stable Release:** `pip install aicsimageio`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/aicsimageio.git`

## Documentation

For full package documentation please visit
[allencellmodeling.github.io/aicsimageio](https://allencellmodeling.github.io/aicsimageio/index.html).

## Quickstart

### Full Image Reading

```python
from aicsimageio import AICSImage

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
img.set_scene("Image:1")

# Use the same operations on a different scene
# ...
```

#### Full Image Reading Notes

The `.data` and `.xarray_data` properties will load the whole image into memory.
The `.get_image_data` function will load the whole image into memory and then retrieve
the specified chunk.

### Delayed Image Reading

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.dask_data  # returns 5D TCZYX dask array
img.xarray_dask_data  # returns 5D TCZYX xarray data array backed by dask array
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order

# Pull only a specific chunk in-memory
lazy_t0 = img.get_image_dask_data("CZYX", T=0)  # returns out-of-memory 4D dask array
t0 = lazy_t0.compute()  # returns in-memory 4D numpy array

# Get the id of the current operating scene
img.current_scene

# Get a list valid scene ids
img.scenes

# Change scene
img.set_scene("Image:1")

# Use the same operations on a different scene
# ...
```

#### Delayed Image Reading Notes

The `.dask_data` and `.xarray_dask_data` properties and the `.get_image_dask_data`
function will not load any piece of the imaging data into memory until you specifically
call `.compute` on the returned Dask array. In doing so, you will only then load the
selected chunk in-memory.

### Xarray Coordinate Plane Attachment

If `aicsimageio` finds coordinate information for the spatial-temporal dimensions of
the image in metadata, you can use
[xarray](http://xarray.pydata.org/en/stable/index.html) for indexing by coordinates.

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.ome.tiff")

# Get the first ten seconds (not frames)
first_ten_seconds = img.xarray_data[:10]  # returns an xarray.DataArray

# Get the first ten major units (usually micrometers, not indices) in Z
first_ten_mm_in_z = img.xarray_data[:, :, :10]

# Get the first ten major units (usually micrometers, not indices) in Y
first_ten_mm_in_y = img.xarray_data[:, :, :, :10]

# Get the first ten major units (usually micrometers, not indices) in X
first_ten_mm_in_x = img.xarray_data[:, :, :, :, :10]
```

See
[xarray.DataArray documentation](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray)
for more details.

### Remote Image Reading

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("http://my-website.com/my_file.tiff")
img = AICSImage("s3://my-bucket/my_file.tiff")
img = AICSImage("gcs://my-bucket/my_file.tiff")

# All other normal operations work just fine
```

Remote reading requires that the file-system specification implementation for the
target backend is installed.

-   For `s3`: `pip install s3fs`
-   For `gs`: `pip install gcsfs`

See the [list of known implementations](https://filesystem-spec.readthedocs.io/en/latest/?badge=latest#implementations).

### Metadata Reading

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.metadata  # returns the metadata object for this file format (XML, JSON, etc.)
img.channel_names  # returns a list of string channel names found in the metadata
img.physical_pixel_size.Z  # returns the Z dimension pixel size as found in the metadata
img.physical_pixel_size.Y  # returns the Y dimension pixel size as found in the metadata
img.physical_pixel_size.X  # returns the X dimension pixel size as found in the metadata
```

### Format Conversion to OME-TIFF

```python
from aicsimageio import AICSImage

AICSImage("my_file.czi").save("my_file.ome.tiff")
```

### Base Reader Specification

All base readers (`TiffReader`, `OmeTiffReader`, `DefaultReader`, etc.) all follow the
same base specification. Each reader will have documentation for functions and properties
specific to itself while
[the base Reader documentation](./aicsimageio.readers.html#module-aicsimageio.readers.reader)
is the best place to get an overview of all functions and properties available to all
base Reader classes.

## Performance Considerations

-   **If your image fits in memory:** use `AICSImage.data`, `AICSImage.xarray_data`,
    `AICSImage.get_image_data`, or `Reader` equivalents.
-   **If your image is too large to fit in memory:** use `AICSImage.dask_data`,
    `AICSImage.xarray_dask_data`, `AICSImage.get_image_dask_data`, or `Reader` equivalents.
-   **If your image does not support native chunk reading:** it may not be best to read
    chunks from a remote source. While possible, the format of the image matters a lot for
    chunked read performance.

## Benchmarks

AICSImageIO is benchmarked using [asv](https://asv.readthedocs.io/en/stable/).
You can find the benchmark results for every commit to `main` starting at the 4.0
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

## Development

See our
[developer resources](https://allencellmodeling.github.io/aicsimageio/developer_resources)
for information related to developing the code.

_Free software: BSD-3-Clause_

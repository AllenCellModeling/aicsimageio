# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Main/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/aicsimageio/)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/main/graph/badge.svg)](https://app.codecov.io/gh/AllenCellModeling/aicsimageio/branch/main)

Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python

---

## Features

-   Supports reading metadata and imaging data for:
    -   `OME-TIFF`
    -   `TIFF`
    -   `CZI` -- (`pip install aicsimageio[czi]`)
    -   `LIF` -- (`pip install aicsimageio[lif]`)
    -   `PNG`, `GIF`, [etc.](https://github.com/imageio/imageio) -- (`pip install aicsimageio[base-imageio]`)
-   Supports writing metadata and imaging data for:
    -   `OME-TIFF`
    -   `PNG`, `GIF`, [etc.](https://github.com/imageio/imageio) -- (`pip install aicsimageio[base-imageio]`)
-   Supports reading and writing to
    [fsspec](https://github.com/intake/filesystem_spec) supported file systems
    wherever possible:

    -   Local paths (i.e. `my-file.png`)
    -   HTTP URLs (i.e. `https://my-domain.com/my-file.png`)
    -   [s3fs](https://github.com/dask/s3fs) (i.e. `s3://my-bucket/my-file.png`)
    -   [gcsfs](https://github.com/dask/gcsfs) (i.e. `gcs://my-bucket/my-file.png`)

    See [Cloud IO Support](#cloud-io-support) for more details.

## Installation

**Stable Release:** `pip install aicsimageio`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/aicsimageio.git`

AICSImageIO is supported on Windows, Mac, and Ubuntu.
For other platforms, you will likely need to build from source.

#### Extra Format Installation

TIFF and OME-TIFF reading and writing is always available after
installing `aicsimageio`, but extra supported formats can be
optionally installed using `[...]` syntax.

-   For a single additional supported format (i.e. CZI): `pip install aicsimageio[czi]`
-   For multiple additional supported formats: `pip install aicsimageio[czi,lif]`
-   For all additional supported formats: `pip install aicsimageio[all]`

## Documentation

For full package documentation please visit
[allencellmodeling.github.io/aicsimageio](https://allencellmodeling.github.io/aicsimageio/index.html).

## Quickstart

### Full Image Reading

If your image fits in memory:

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

# Change scene using name
img.set_scene("Image:1")
# Or by scene index
img.set_scene(img.scenes[1])

# Use the same operations on a different scene
# ...
```

#### Full Image Reading Notes

The `.data` and `.xarray_data` properties will load the whole scene into memory.
The `.get_image_data` function will load the whole scene into memory and then retrieve
the specified chunk.

### Delayed Image Reading

If your image doesn't fit in memory:

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

# Change scene using name
img.set_scene("Image:1")
# Or by scene index
img.set_scene(img.scenes[1])

# Use the same operations on a different scene
# ...
```

#### Delayed Image Reading Notes

The `.dask_data` and `.xarray_dask_data` properties and the `.get_image_dask_data`
function will not load any piece of the imaging data into memory until you specifically
call `.compute` on the returned Dask array. In doing so, you will only then load the
selected chunk in-memory.

### Mosaic Image Reading

Read stitched data or single tiles as a dimension.

Readers that support mosaic tile stitching:

-   `LifReader`

#### AICSImage

If the file format reader supports stitching mosaic tiles together, the
`AICSImage` object will default to stitching the tiles back together.

```python
img = AICSImage("very-large-mosaic.lif")
img.dims.order  # T, C, Z, big Y, big X, (S optional)
img.dask_data  # Dask chunks fall on tile boundaries, pull YX chunks out of the image
```

This behavior can be manually turned off:

```python
img = AICSImage("very-large-mosaic.lif", reconstruct_mosaic=False)
img.dims.order  # M (tile index), T, C, Z, small Y, small X, (S optional)
img.dask_data  # Chunks use normal ZYX
```

If the reader does not support stitching tiles together the M tile index will be
available on the `AICSImage` object:

```python
img = AICSImage("some-unsupported-mosaic-stitching-format.ext")
img.dims.order  # M (tile index), T, C, Z, small Y, small X, (S optional)
img.dask_data  # Chunks use normal ZYX
```

#### Reader

If the file format reader detects mosaic tiles in the image, the `Reader` object
will store the tiles as a dimension.

If tile stitching is implemented, the `Reader` can also return the stitched image.

```python
reader = LifReader("ver-large-mosaic.lif")
reader.dims.order  # M, T, C, Z, tile size Y, tile size X, (S optional)
reader.dask_data  # normal operations, can use M dimension to select individual tiles
reader.mosaic_dask_data  # returns stitched mosaic - T, C, Z, big Y, big, X, (S optional)
```

#### Single Tile Absolute Positioning

There are functions available on both the `AICSImage` and `Reader` objects
to help with single tile positioning:

```python
img = AICSImage("very-large-mosaic.lif")
img.mosaic_tile_dims  # Returns a Dimensions object with just Y and X dim sizes
img.mosaic_tile_dims.Y  # 512 (for example)

# Get the tile start indices (top left corner of tile)
y_start_index, x_start_index = img.get_mosaic_tile_position(12)
```

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

### Xarray Coordinate Plane Attachment

If `aicsimageio` finds coordinate information for the spatial-temporal dimensions of
the image in metadata, you can use
[xarray](http://xarray.pydata.org/en/stable/index.html) for indexing by coordinates.

```python
from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.ome.tiff")

# Get the first ten seconds (not frames)
first_ten_seconds = img.xarray_data.loc[:10]  # returns an xarray.DataArray

# Get the first ten major units (usually micrometers, not indices) in Z
first_ten_mm_in_z = img.xarray_data.loc[:, :, :10]

# Get the first ten major units (usually micrometers, not indices) in Y
first_ten_mm_in_y = img.xarray_data.loc[:, :, :, :10]

# Get the first ten major units (usually micrometers, not indices) in X
first_ten_mm_in_x = img.xarray_data.loc[:, :, :, :, :10]
```

See `xarray`
["Indexing and Selecting Data" Documentation](http://xarray.pydata.org/en/stable/indexing.html)
for more information.

### Cloud IO Support

[File-System Specification (fsspec)](https://github.com/intake/filesystem_spec) allows
for common object storage services (S3, GCS, etc.) to act like normal filesystems by
following the same base specification across them all. AICSImageIO utilizes this
standard specification to make it possible to read directly from remote resources when
the specification is installed.

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

### Saving to OME-TIFF

The simpliest method to save your image as an OME-TIFF file with key pieces of
metadata is to use the `save` function.

```python
from aicsimageio import AICSImage

AICSImage("my_file.czi").save("my_file.ome.tiff")
```

**Note:** By default `aicsimageio` will generate only a portion of metadata to pass
along from the reader to the OME model. This function currently does not do a full
metadata translation.

For finer grain customization of the metadata, scenes, or if you want to save an array
as an OME-TIFF, the writer class can also be used to customize as needed.

```python
import numpy as np
from aicsimageio.writers import OmeTiffWriter

image = np.random.rand(10, 3, 1024, 2048)
OmeTiffWriter.save(image, "file.ome.tif", dim_order="ZCYX")
```

See
[OmeTiffWriter documentation](./aicsimageio.writers.html#aicsimageio.writers.ome_tiff_writer.OmeTiffWriter.save)
for more details.

#### Other Writers

In most cases, `AICSImage.save` is usually a good default but there are other image
writers available. For more information, please refer to
[our writers documentation](https://allencellmodeling.github.io/aicsimageio/aicsimageio.writers.html).

## Benchmarks

AICSImageIO is benchmarked using [asv](https://asv.readthedocs.io/en/stable/).
You can find the benchmark results for every commit to `main` starting at the 4.0
release on our
[benchmarks page](https://AllenCellModeling.github.io/aicsimageio/_benchmarks/index.html).

## Development

See our
[developer resources](https://allencellmodeling.github.io/aicsimageio/developer_resources)
for information related to developing the code.

## Citation

If you find `aicsimageio` useful, please cite this repository as:

> AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

_Free software: BSD-3-Clause_

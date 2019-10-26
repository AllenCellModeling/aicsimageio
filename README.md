# AICSImageIO

[![Build Status](https://github.com/AllenCellModeling/aicsimageio/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/aicsimageio/actions)
[![Documentation](https://github.com/AllenCellModeling/aicsimageio/workflows/Docs/badge.svg)](https://allencellmodeling.github.io/aicsimageio)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/aicsimageio/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aicsimageio)

A Python library for reading and writing image data with specific support for handling bio-formats.

---

## Features
* Supports reading metadata and imaging data from file path or buffered bytes for:
    * `CZI`
    * `OME-TIFF`
    * `TIFF`
    * Any additional format supported by `imageio`
* Supports writing metadata and imaging data for:
    * `OME-TIFF`
    * `TIFF`
    * Any additional format supported by `imageio`

### Disclaimer:
This package is under heavy revision in preparation for version 3.0.0 release. The quick start below is representative
of how to interact with the package under 3.0.0 and not under the current stable release.

## Quick Start
```python
from aicsimageio import AICSImage, imread

# For numpy array of image data
im = imread("/path/to/your/file_or_buffer.ome.tiff")

# For AICSImage object that
im = AICSImage("/path/to/your/file_or_buffer.ome.tiff")

# To specify a known dimension order
im = AICSImage("/path/to/your/file_or_buffer.ome.tiff", known_dims="SYX")

# Image data is stored in `data` attribute
im.data  # returns the image data numpy array

# Image dimension sizes can be obtained via properties:
im.size_z  # returns the size of the Z dimension. X,Y,Z,C,T, and S supported.

# Image dimensions can also be obtained as a tuple in two ways:
im.size("ZYX")  # returns a tuple containing the Z, Y, and X sizes only
im.get_image_data(out_orientation="ZYX").shape  # returns same as above

# Image metadata is stored in `metadata` attribute
im.metadata  # returns whichever metadata parser best suits the file format

# Subsets or transposes of the image data can be requested:
im.get_image_data(out_orientation="ZYX")  # returns a 3d data block containing only the ZYX dimensions

```

## Notes
* Image data numpy arrays are always returned as six dimensional in dimension order `STCZYX`
or `Scene`, `Time`, `Channel`, `Z`, `Y`, and `X`.
* Each file format may use a different metadata parser it is dependent on the reader's implementation.

## Installation
**Stable Release:** `pip install aicsimageio`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/aicsimageio.git`

## Documentation
For full package documentation please visit [allencellmodeling.github.io/aicsimageio](https://allencellmodeling.github.io/aicsimageio/index.html).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

***Free software: BSD-3-Clause***

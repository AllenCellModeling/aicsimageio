# Mission and Values

This document is meant to help guide decisions about the future of AICSImageIO, be it
in terms of whether to accept new functionality, changes to existing functionality,
changes to package administrative tasks, etc. It serves as a reference for core
developers in guiding their work, and, as an introduction for newcomers who want to
learn where the project is headed and what the team's values are. You can also learn
how the project is managed by looking at our [governance model](./GOVERNANCE.md).

## Mission

AICSImageIO aims to provide a **consistent intuitive API for reading in or out-of-memory
image pixel data and metadata** for the many existing proprietary microscopy file
formats, and, an **easy-to-use API for converting from proprietary file formats to an
open, common, standard** -- all using either language agnostic or pure Python tooling.

In short:
> AICSImageIO provides a method to fully read and convert from an existing proprietary
> microscopy file format to, or _emulate_, a Python representation of the community
> standard metadata model regardless of image size, format, or location.

(The current community standard for microscopy images is the
[Open Microscopy Environment](https://www.openmicroscopy.org/))

We hope to accomplish this by:
* being **easy to use and install**. We will take extra care to ensure that this library
is easy to use and fully installable on Windows, Mac-OS, and Ubuntu.
* being **well-documented** with our entire API having up-to-date, useful docstrings
and additionally providing examples of more complex use-cases when possible.
* providing a **consistent and stable API** to users by following
[semantic versioning](https://semver.org/) and limiting the amount of breaking changes
introduced unless necessary for the future robustness or scalability of the library.
* sustaining **comparable or better performance when compared to more tailored file
format reading libraries**. We will regularly run benchmarks utilizing a set of varied
size images from all the file formats the library is capable of reading.
* **working closely with the microscopy community** while deciding on standards and best
practices for open, accessible, file formats and imaging and in deciding which
proprietary file formats and metadata selection are in need of support.

## Values
* We are **inclusive**. We welcome and mentor newcomers who are making their first
contribution and strive to grow our most dedicated contributors into core developers. We
have a [Code of Conduct](./CODE_OF_CONDUCT.md) to ensure that the AICSImageIO remains
a welcoming place for all.
* We are **community-driven**. We respond to feature requests and proposals on our
[issue tracker](https://github.com/AllenCellModeling/aicsimageio/issues) and make
decisions that are driven by our user's requirements.
* We focus on **file IO and metadata conversion**, leaving image analysis functionality
and visualization to other libraries.
* We aim to **develop new methods of metadata extraction and conversion**, instead of
duplicating existing, or porting, from other libraries primarily by creating **language
agnostic methods** for metadata manipulation.
* We value **simple, readable implementations**. Readable code that is easy to
understand, for newcomers and maintainers alike, makes it easier to contribute new code
as well as prevent bugs.
* We value **education and documentation**. All functions should have docstrings,
preferably with examples, and major functionality should be explained in our tutorials.
Core developers can take an active role in finishing documentation examples.
* We **minimize [magic](https://en.wikipedia.org/wiki/Magic_(programming))** and always
provide a way for users to opt out of magical behaviour and guessing by providing
explicit ways to control functionality.

## Acknowledgments
We share a lot of our mission and values with the `napari` project, and acknowledge the
influence of their mission and values statement on this document.

Additionally, much of the work produced for this library is built on the shoulders of
giants. Notably:
* [Christoph Gohlke](https://www.lfd.uci.edu/~gohlke/) -- maintainer of `tifffile`,
`czifile`, and the `imagecodecs` libraries
* [Paul Watkins](https://github.com/elhuhdron) -- original creator of `pylibczi`
* [OME and Bio-Formats Team](https://github.com/ome/bioformats) -- proprietary
microscopy file format conversion and standards development
* [Python-Bio-Formats Team](https://github.com/CellProfiler/python-bioformats) --
Python Java Bridge for Bio-Formats and original implementations of OME Python
representation
* [imageio Team](https://github.com/imageio/imageio) -- robust, expansive, cross
platform image reading
* [Dask Team](https://dask.org/) -- delayed and out-of-memory parallel array
manipulation
* [xarray Team](https://github.com/pydata/xarray) -- coordinate and metadata attached
array handling and manipulation

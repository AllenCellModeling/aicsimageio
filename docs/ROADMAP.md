# AICSImageIO Roadmap

## For 4.* Series of Releases - September 2020
The `aicsimageio` roadmap captures current development priorities within the project
and should serve as a guide for core developers, to encourage contribution for new
contributors, and to provide insight to external developers who are interested in using
`aicsimageio` in their work.

The [mission](./MISSION_AND_VALUES.HTML) of `aicsimageio` is to provide a method to
fully read and convert from an existing proprietary microscopy file format to, or
_emulate_, a Python representation of the community standard metadata model regardless
of image size, format, or location. To work towards this mission, we have set a few
high-level priorities over the upcoming months:

* Enable **reading and writing to remote sources** quick and easy
* Make **writing n-dimensional images** with partial or complete metadata **simple**
* Make **converting and accessing metadata** as easy as possible with a **standard API**
* Expand the list of currently available **proprietary file format readers**

## Enable Reading and Writing to Remote Sources Quick and Easy
While most microscopy labs and computational scientists currently work with entirely
local datasets, it is becoming increasingly common for work to be conducted entirely on
the cloud. To address this, we plan to allow remote paths or pointers to data directly
into the high-level functions and objects of this library.

Fortunately, many existing frameworks and libraries can handle this for us to a large
extent but do come with some concerns about how to do so efficiently and easily,
specifically on the chunked / out-of-memory reading side.

In-memory reading from remote sources is relatively simple for all formats. Out-of
memory reading for formats not-initially designed to handled remote friendly chunked
reading are the crux of the issue (i.e. TIFF, CZI, LIF, etc.). Adding remote reading is
great for an initial 4.0 addition but _optimizing_ chunked remote reading wherever
possible in future work of the 4.* series is a high priority.

Our [benchmarks](./BENCHMARKS.md) will be incredibly useful in tracking and
maintaining our performance.

## Make Writing n-dimensional Images with Partial or Complete Metadata Simple
N-dimensional image writing _with_ metadata has largely been ignored outside of the work
done by the proprietary file format developers themselves. To do so for the general
computational scientist requires much more flexibility in what to allow as valid
metadata, and, the bare minimum needed to deem a file as having "useful" metadata.

"Useful" is largely defined by the community so we will not define it ourselves, but,
looking to the community and answering the call for a simple n-dimensional image writer
_with "useful" metadata attachment_ is where we can contribute.

In general, we have followed the [OME](https://www.openmicroscopy.org/) team in
metadata specification and have participated in the OME Community Meeting's in
discussing the exact question of "what is 'useful' metadata?" Keeping with them, we will
continue to improve the OME-TIFF writing experience for both _updating_ image metadata
or generating bare minimum metadata for a research / processing result.

In addition, addressing the problems with reading and writing _remote_ data mentioned
previously, we plan to add writing to a remote source as baseline functionality, making
it efficient, and adding more file format writers to the library as more file formats
better equipped to handle chunked remote reading become available, i.e.
[Zarr](https://zarr.readthedocs.io/en/stable/) and OME-Zarr.

## Make Converting and Accessing Metadata as Easy as Possible with a Standard API
As more and more proprietary file format readers are added to the library all with their
own metadata schema, it becomes harder and harder for computational scientists to
quickly switch from one dataset to another simply based off of their dataset's file
format. Additionally, we recognize the work already done by the
[OME](https://www.openmicroscopy.org/) and
[Python Bio-Formats](https://github.com/CellProfiler/python-bioformats) teams in making
this even remotely possible to begin with. We want to expand on their existing work and
address the problems of scaling to multiple programming languages, making it easier for
non-computational users to contribute, and more.

To address this we are planning a large chunk of work around "language-agnostic metadata
schema conversion." We have already begun to prototype this work with
[CZI-to-OME-XSLT](https://github.com/allencellmodeling/czi-to-ome-xslt), a repository
that can be used in any language that has [XSLT](https://en.wikipedia.org/wiki/XSLT)
support or a usage library.

For us this would mean two things:
1. a more generalized form of metadata schema conversion that multiple languages can use
and contribute back to instead of duplicated work in many languages
2. a simple system to convert metadata schema, allowing for a unified access API

While we will not make it a requirement that with the addition of a new proprietary
file format reader to the library, the contributor must also add a sub-module to a
metadata schema conversion repository, we will however _highly encourage it._ We
believe that this strategy will allow us in the long-run a more maintainable code base
as well as a simple end-user API because, in general, metadata conversion is cheap and
fast, allowing us to convert to a common standard (OME) on request and handle all
metadata queries using the same schema.

In the case where no conversion XSLT or similar "language-agnostic method" sub-module
is added, we will still make an effort to extract the metadata in code while still
conforming to the standard API for the time being.

## Expand the List of Currently Available Proprietary File Format Readers
Looking to our needs at the Allen Institute for Cell Science first, we will continue to
add new readers to the library as needed. However, we will happily accept, review, and
help maintain new proprietary file format readers to the library from contributors.

With a few exceptions, in general the only requirement of a new file format to be
supported would be that the reader must be able to accept local and remote requests as
per our previously stated goals.

Specifically for the Allen Institute for Cell Science, we would like to add readers for:
* Slidebook (`.sld`)
* Zarr (`.zarr`)
* OME-Zarr (`.ome.zarr`)

Our greatest hope would be that the proprietary file format developer themselves added
new readers, and, as previously described, metadata conversion functionality.
Historically however, this has not been the case. We will work on fostering
relationships with the developers of the priority file formats where possible to
change this, whether it is for `aicsimageio` or another library -- open access and open
standards are still better, regardless of `aicsimageio` or not.

## About This Document
This document is meant to be a snapshot or high-level objectives and reasoning for the
library during our 4.* series of releases starting in September 2020.

For more low-level implementation details, features, bugs, documentation requests, etc,
please see our [issue tracker](https://github.com/AllenCellModeling/aicsimageio/issues).

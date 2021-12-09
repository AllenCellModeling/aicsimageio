# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!

Ready to contribute? Here's how to set up `aicsimageio` for local development.

1.  Fork the `aicsimageio` repo on GitHub.

2.  Clone your fork locally:

        git clone https://{your_name_here}@github.com/aicsimageio.git

3.  Check out the submodules used in this repo:

        git submodule update --init --recursive

4.  Install the project in editable mode (and preferably in a virtual environment):

        cd aicsimageio/
        pip install -e .[dev]

5.  Download the test resources:

        python scripts/download_test_resources.py

    If you need to upload new resources please let a core maintainer know.

    If you cannot download the test resources, feel free to open a Draft Pull Request
    to the main `aicsimageio` repository as our GitHub Actions will allow you to test
    with the resources downloaded.

6.  Create a branch for local development:

        git checkout -b {your_development_type}/short-description

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

7.  When you're done making changes, check that your changes pass linting and
    tests, including testing other Python versions with make (!! Important !!):

        make build

    If you have any AWS credentials configured and would like to run the full
    remote IO test suite:

        make build-with-remote

8.  (Optional) If you'd like to have linting checks run automatically prior to
    every `git commit`, then you can install a commit-hook that runs
    [pre-commit](https://pre-commit.com/), as follows:

        pip install pre-commit
        pre-commit install

9.  Commit your changes and push your branch to GitHub:

        git add .
        git commit -m "Resolves gh-###. Your detailed description of your changes."
        git push origin {your_development_type}/short-description

10. Submit a pull request through the GitHub website.

## Updating the Submodule
If there are updates to the submodule repo that you would like to integrate into AICSImageIO, follow these steps:
1. Create a new branch in this repo:
```
git checkout -b {your_development_type}/short-description
```

2. Pull in the changes from the submodule:
```
git submodule update --remote --merge
```

3. Commit the changes caused by the above command.

4. Create a PR with these changes.

## Adding a New Custom Reader

### Basics

AICSImageIO is broken into the `Reader` base class and the `AICSImage` higher
level "standardization" class. `Reader` classes should always provide
access in some sense to the "raw" data. The `AICSImage` class wraps the
`Reader` classes to then select and transform the "raw" data into our
AICS / near-OME standard model.

### Reader Class Implementation

The `Reader` base class can be thought of as a wrapper around
`xarray` since we store most or all data and metadata on
`xarray.DataArray` objects.
Because of this, it may be useful to read the
[xarray documentation](http://xarray.pydata.org/en/stable/).

The `Reader` base class in full can be found
[here](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/readers/reader.py).

New `Reader` classes are **required** to implement:

-   `_is_supported_image`: a function to check if the custom `Reader` class
    _can_ read the provided object (file, array, etc.)
-   `scenes`: a property which returns the Tuple of all scene names
    (or in OME terms, "images" or "series").
-   `_read_delayed`: a function which returns an `xarray.DataArray` object
    which is backed by a delayed `dask.array.Array`.
-   `_read_immediate`: a function which returns an `xarray.DataArray` object
    which is backed by an in-memory `numpy.ndarray`.

New `Reader` classes can **optionally** implement:

-   `_get_stitched_dask_mosaic`: a function to stitch mosaic tiles from a
    delayed dask array, otherwise raises `NotImplementedError`.
-   `_get_stitched_mosaic`: a function to stitch mosaic tiles from an
    in-memory numpy array, otherwise raises `NotImplementedError`.
-   `get_mosaic_tile_position`: a function to get a specific mosaic tile's position
    from it's tile index, otherwise raises `NotImplementedError`.
-   `ome_metadata`: a property which returns the format's metadata type converted to
    OME metadata, otherwise raises `NotImplementedError`.
-   `physical_pixel_sizes`: a property which returns the spatial dimensions pixel sizes,
    otherwise returns `(None, None, None)`

Please see the `Reader` code for docstrings with more information
for each of the above functions and properties.

### Custom Dependencies

If your `Reader` requires custom dependencies, add the custom dependencies
to our [format_libs lookup in setup.py](https://github.com/AllenCellModeling/aicsimageio/blob/main/setup.py#L26).

If you want your `Reader` to be used as a part of the `AICSImage`
object's attempted reader resolution (i.e. the `AICSImage` object iterates
through `Reader` objects until a `Reader` can read the provided object),
add your file format(s) + reader module path to our
[FORMAT_IMPLEMENTATIONS lookup](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/formats.py).

**Note:** the order of the `FORMAT_IMPLEMENTATIONS` lookup matters
--if `TiffReader` was put before `OmeTiffReader`,
`OmeTiffReader` would never be reached.

For an example of a `Reader` that requires custom dependencies, see our `CziReader` (specifically
[Reader dependency lookup during import of the module](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/readers/czi_reader.py#L24)).

### Testing

We provide standard functions for testing and validating data and metadata
as well as running serialization and deserialization and other checks for you.
These are:

-   [run_image_container_checks](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/tests/image_container_test_utils.py#L46)
    for single-scene
-   [run_multi_scene_image_read_checks](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/tests/image_container_test_utils.py#L179)
    for multi-scene.

See our [OmeTiffReader Tests](https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/tests/readers/test_ome_tiff_reader.py)
for an example parametrized usage of these standard testing functions.

Outside of using these standard testing functions, you should feel free to
add additional testing for you `Reader` class when needed or
for cases that you feel we should support.

### Optional Reader Benchmarking

If you want to benchmark your `Reader` over the course of `aicsimageio` patches,
add a `{YourReader}Suite` class to our
[benchmark_image_containers file](https://github.com/AllenCellModeling/aicsimageio/blob/main/benchmarks/benchmark_image_containers.py#L167).

**Note:** You may want to specifically choose files larger than 100MB to
benchmark against to make random IO spikes on the GitHub Action runner negligible.

### Documentation

Let people know your `Reader` is available for use! Document it's install pattern
in the README.

## Benchmarking

If you are working on a patch that would change a base reader it is recommended
to run `asv` to benchmark how the change compares to the current release.

To do so simply run `asv` in the top level directory of this repo.
You can create a specific comparison by running `asv continuous branch_a branch_b`.

For more information on `asv` and full commands please see
[their documentation](https://asv.readthedocs.io/en/stable/).

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
make prepare-release
git push
git push --tags
```

After all builds pass, GitHub Actions will automatically publish to PyPI.

**Note:** `make prepare-release` by default only bumps the patch number and
not a minor or major version.

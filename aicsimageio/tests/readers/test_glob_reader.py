#! usr/env/bin/python
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tifffile as tiff
import xarray as xr

import aicsimageio
from aicsimageio.readers.tiff_glob_reader import TiffGlobReader

DATA_SHAPE = (3, 4, 5, 6, 7, 8)  # STCZYX


def check_values(
    reader: aicsimageio.readers.TiffGlobReader, reference: xr.DataArray
) -> None:
    for i, s in enumerate(reader.scenes):
        reader.set_scene(s)
        np.testing.assert_array_equal(reference.isel(S=i).data, reader.xarray_data.data)


def make_fake_data_2d(path: Path, as_mm: bool = False) -> xr.DataArray:
    """
    Parameters
    ----------
    path : [Path]
        Folder to save data in
    as_mm : [bool]
        Whether to save the data in with Micromanager MDA naming conventions.

    Returns
    -------
    x_data : [xr.DataArray]
    """

    data = np.arange(np.prod(DATA_SHAPE), dtype="uint16").reshape(DATA_SHAPE)
    dims = list("STCZYX")

    x_data = xr.DataArray(data, dims=dims)

    os.mkdir(str(path / "2d_images"))
    for s, t, c, z in product(*(range(x) for x in DATA_SHAPE[:4])):

        im = data[s, t, c, z]
        if as_mm:
            name = f"img_channel{c}_position{s}_time{t}_z{z}.tif"
        else:
            name = f"S{s}_T{t}_C{c}_Z{z}.tif"
        tiff.imwrite(
            str(path / "2d_images" / name),
            im,
            dtype=np.uint16,
        )
    return x_data


def test_glob_reader_2d(tmp_path: Path) -> None:
    reference = make_fake_data_2d(tmp_path)
    gr = aicsimageio.readers.TiffGlobReader(str(tmp_path / "2d_images/*.tif"))

    assert gr.xarray_dask_data.data.chunksize == (1, 1) + DATA_SHAPE[-3:]

    check_values(gr, reference)


def test_index_alignment(tmp_path: Path) -> None:
    # Testing case where user has passed in a list of files
    # and a dataframe with non-continuous index

    # use as_mm to have an easily available Indexer function
    _ = make_fake_data_2d(tmp_path, as_mm=True)
    filenames = np.array(list((tmp_path / "2d_images").glob("*.tif")))
    # print(filenames)
    indexer = pd.Series(filenames).apply(TiffGlobReader.MicroManagerIndexer)

    # Keep only some of the Z
    # more realistic case is eliminating everything after a given time point
    # but this garuntees that our eliminated images will be embedded all through the
    # order rather than just at the end
    keep = indexer.Z < 5

    indexer = indexer.loc[keep]

    reader = TiffGlobReader(filenames[keep], indexer)

    # check that there are no nans
    # nans are a symptom of index misalignment
    assert not reader._all_files.isnull().any().any()


@pytest.mark.parametrize(
    "type_",
    [
        list,
        pd.Series,
        np.array,
        # should throw a TypeError instead of an unboundlocal error
        pytest.param(bytes, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_glob_types(type_: Any, tmp_path: Path) -> None:
    reference = make_fake_data_2d(tmp_path)
    filenames = list((tmp_path / "2d_images").glob("*.tif"))

    gr = TiffGlobReader(type_(filenames))
    check_values(gr, reference)


def test_mm_indexer(tmp_path: Path) -> None:
    _ = make_fake_data_2d(tmp_path, True)
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "2d_images/*.tif"), indexer=TiffGlobReader.MicroManagerIndexer
    )
    assert gr.dims.order == "TCZYX"
    assert gr.dims.shape == DATA_SHAPE[1:]


def make_fake_data_3d(path: Path) -> xr.DataArray:

    data = np.arange(np.prod(DATA_SHAPE), dtype="uint16").reshape(DATA_SHAPE)

    dims = list("STCZYX")

    x_data = xr.DataArray(data, dims=dims)

    os.mkdir(str(path / "3d_images"))

    shape_for_3d = (*DATA_SHAPE[:3], int(DATA_SHAPE[3] / 2))
    for s, t, c, z in product(*(range(x) for x in shape_for_3d)):
        im = data[s, t, c, 2 * z : 2 * (z + 1)]
        tiff.imwrite(
            str(path / f"3d_images/S{s}_T{t}_C{c}_Z{z}.tif"),
            im,
            dtype=np.uint16,
        )
    return x_data


def test_glob_reader_3d(tmp_path: Path) -> None:
    reference = make_fake_data_3d(tmp_path)

    # do not stack z dimension
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "3d_images/*Z0.tif"), single_file_dims=list("ZYX")
    )
    assert gr.xarray_dask_data.data.chunksize == (1, 1, 2, 7, 8)
    check_values(gr, reference.isel(Z=slice(0, 2)))

    # stack along z dimension but do not chunk
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "3d_images/*.tif"),
        single_file_dims=list("ZYX"),
        chunk_dims=list("TC"),
    )
    assert gr.xarray_dask_data.data.chunksize == (4, 5, 2, 7, 8)
    check_values(gr, reference)

    # stack along z and chunk along z
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "3d_images/*.tif"), single_file_dims=list("ZYX")
    )
    assert gr.xarray_dask_data.data.chunksize == (1, 1, 6, 7, 8)
    check_values(gr, reference)


def make_fake_data_4d(path: Path) -> xr.DataArray:

    data = np.arange(np.prod(DATA_SHAPE), dtype="uint16").reshape(DATA_SHAPE)

    dims = list("STCZYX")

    x_data = xr.DataArray(data, dims=dims)

    os.mkdir(str(path / "4d_images"))

    per_file_t = 2
    t_files = int(DATA_SHAPE[1] / per_file_t)

    per_file_z = 3
    z_files = int(DATA_SHAPE[3] / per_file_z)

    shape_for_4d = (DATA_SHAPE[0], t_files, DATA_SHAPE[2], z_files)
    for s, t, c, z in product(*(range(x) for x in shape_for_4d)):
        im = data[
            s,
            per_file_t * t : per_file_t * (t + 1),
            c,
            per_file_z * z : per_file_z * (z + 1),
        ]
        tiff.imwrite(
            str(path / f"4d_images/S{s}_T{t}_C{c}_Z{z}.tif"),
            im,
            dtype=np.uint16,
            photometric="MINISBLACK",
        )
    return x_data


def test_glob_reader_4d(tmp_path: Path) -> None:
    reference = make_fake_data_4d(tmp_path)

    # stack none
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "4d_images/*T0*Z0.tif"), single_file_dims=list("TZYX")
    )
    assert gr.xarray_dask_data.data.chunksize == (2, 1, 3, 7, 8)
    check_values(gr, reference.isel(T=slice(0, 2), Z=slice(0, 3)))

    # stack z and t - chunk z
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "4d_images/*.tif"), single_file_dims=list("TZYX")
    )
    assert gr.xarray_dask_data.data.chunksize == (2, 1, 6, 7, 8)
    check_values(gr, reference)

    # stack z and t - chunk z and t
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "4d_images/*.tif"),
        single_file_dims=list("TZYX"),
        chunk_dims=["T", "Z"],
    )
    assert gr.xarray_dask_data.data.chunksize == (4, 1, 6, 7, 8)
    check_values(gr, reference)

    # stack z an t - chunk ztc
    gr = aicsimageio.readers.TiffGlobReader(
        str(tmp_path / "4d_images/*.tif"),
        single_file_dims=list("TZYX"),
        chunk_dims=list("TCZ"),
    )
    assert gr.xarray_dask_data.data.chunksize == (4, 5, 6, 7, 8)
    check_values(gr, reference)


def test_aics_image(tmp_path: Path) -> None:

    _ = make_fake_data_4d(tmp_path)

    aicsimage_tiff = aicsimageio.AICSImage(tmp_path / "4d_images/S0_T0_C0_Z0.tif")
    assert isinstance(aicsimage_tiff.reader, aicsimageio.readers.tiff_reader.TiffReader)

    aicsimage_tiff_glob = aicsimageio.AICSImage(
        tmp_path / "4d_images/*.tif", single_file_dims=list("TZYX")
    )
    assert isinstance(aicsimage_tiff_glob.reader, aicsimageio.readers.TiffGlobReader)

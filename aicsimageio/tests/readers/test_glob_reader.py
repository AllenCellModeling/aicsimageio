#! usr/env/bin/python
import os
from pathlib import Path

import numpy as np
import tifffile as tiff
import xarray as xr

import aicsimageio

DATA_SHAPE = (3, 4, 5, 6, 7, 8)  # STCZYX


def check_values(
    reader: aicsimageio.readers.TiffGlobReader, reference: xr.DataArray
) -> None:
    for i, s in enumerate(reader.scenes):
        reader.set_scene(s)
        assert np.all(
            reference.isel(S=i).data == reader.xarray_dask_data.data
        ).compute()
        assert np.all(reference.isel(S=i).data == reader.xarray_data.data)


def make_fake_data_2d(path: Path) -> xr.DataArray:

    data = np.arange(np.prod(DATA_SHAPE), dtype="uint16").reshape(DATA_SHAPE)

    dims = list("STCZYX")

    x_data = xr.DataArray(data, dims=dims)

    os.mkdir(str(path / "2d_images"))

    for s in range(DATA_SHAPE[0]):
        for t in range(DATA_SHAPE[1]):
            for c in range(DATA_SHAPE[2]):
                for z in range(DATA_SHAPE[3]):
                    im = data[s, t, c, z]
                    tiff.imsave(
                        str(path / f"2d_images/S{s}_T{t}_C{c}_Z{z}.tif"),
                        im,
                        dtype=np.uint16,
                    )
    return x_data


def test_glob_reader_2d(tmp_path: Path) -> None:
    reference = make_fake_data_2d(tmp_path)
    gr = aicsimageio.readers.TiffGlobReader(str(tmp_path / "2d_images/*.tif"))

    assert gr.xarray_dask_data.data.chunksize == (1, 1) + DATA_SHAPE[-3:]

    check_values(gr, reference)


def make_fake_data_3d(path: Path) -> xr.DataArray:

    data = np.arange(np.prod(DATA_SHAPE), dtype="uint16").reshape(DATA_SHAPE)

    dims = list("STCZYX")

    x_data = xr.DataArray(data, dims=dims)

    os.mkdir(str(path / "3d_images"))

    for s in range(DATA_SHAPE[0]):
        for t in range(DATA_SHAPE[1]):
            for c in range(DATA_SHAPE[2]):
                for z in range(int(DATA_SHAPE[3] / 2)):
                    im = data[s, t, c, 2 * z : 2 * (z + 1)]
                    tiff.imsave(
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

    for s in range(DATA_SHAPE[0]):
        for t in range(t_files):
            for c in range(DATA_SHAPE[2]):
                for z in range(z_files):
                    im = data[
                        s,
                        per_file_t * t : per_file_t * (t + 1),
                        c,
                        per_file_z * z : per_file_z * (z + 1),
                    ]
                    tiff.imsave(
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

    aicsimage_tiff = aicsimageio.AICSImage(tmp_path / "3d_images/S0_T0_C0_Z0.tif")
    assert isinstance(aicsimage_tiff.reader, aicsimageio.readers.tiff_reader.TiffReader)

    aicsimage_tiff_glob = aicsimageio.AICSImage(
        tmp_path / "3d_images/*.tif", single_file_dims=list("ZYX")
    )
    assert isinstance(aicsimage_tiff_glob.reader, aicsimageio.readers.TiffGlobReader)

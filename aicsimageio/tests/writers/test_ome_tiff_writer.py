#!/usr/bin/env python

import os
import pytest

import numpy as np

from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.exceptions import InvalidDimensionOrderingError

filename = "ometif_test_output.ome.tif"

image = np.random.rand(1, 40, 3, 128, 256).astype(np.uint16)


def test_writerShapeComparison(resources_dir):
    """
    Test to check that OmeTiffWriter saves arrays that are reflexive with OmeTiffReader
    """
    writer = OmeTiffWriter(resources_dir / filename, overwrite_file=True)
    writer.save(image)
    writer.close()

    with OmeTiffReader(resources_dir / filename) as test_output_reader:
        output = test_output_reader.data

    assert output.shape == image.shape[1:]


def test_loadAssertionError(resources_dir):
    """
    Test to check if save() will only accept 3, 4, 5 dimensions for data
    """
    image_to_save = np.ones((1, 2, 3, 4, 5, 6))
    with pytest.raises(Exception):
        writer = OmeTiffWriter(resources_dir / filename, overwrite_file=True)
        writer.save(image_to_save)


def test_overwriteFile(resources_dir):
    """
    Test to check if save() can overwrite a file
    """
    with OmeTiffWriter(resources_dir / filename, overwrite_file=True) as writer:
        writer.save(image)


def test_dontOverwriteFile(resources_dir):
    """
    Test to check if save() will raise error when user does not want to overwrite a file that exists
    """
    with pytest.raises(Exception):
        with OmeTiffWriter(resources_dir / filename) as writer:
            writer.save(image)


def test_noopOverwriteFile(resources_dir):
    """
    Test to check if save() silently no-ops when user does not want to overwrite exiting file
    """
    with open(resources_dir / filename, "w") as f:
        f.write("test")
    with OmeTiffWriter(resources_dir / filename, overwrite_file=False) as writer:
        writer.save(image)
    with open(resources_dir / filename, "r") as f:
        line = f.readline().strip()
        assert "test" == line


def test_big_tiff():
    x = np.zeros((10, 10))
    assert OmeTiffWriter._size_of_ndarray(data=x) == 10 * 10 * x.itemsize


@pytest.mark.parametrize(
    "dims, expected_t, expected_c, expected_z, expected_y, expected_x",
    [
        ("TCZYX", 1, 40, 3, 128, 256),
        ("TZCYX", 1, 3, 40, 128, 256),
        ("CZTYX", 3, 1, 40, 128, 256),
        ("CTZYX", 40, 1, 3, 128, 256),
        ("ZCTYX", 3, 40, 1, 128, 256),
        ("ZTCYX", 40, 3, 1, 128, 256),
        ("STCZYX", 1, 40, 3, 128, 256),
        (
            pytest.param(
                "XY",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
        (
            pytest.param(
                "ABCD",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
        (
            pytest.param(
                "XXXXX",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
    ],
)
def test_dimensionOrder(
    resources_dir, dims, expected_t, expected_c, expected_z, expected_y, expected_x
):
    writer = OmeTiffWriter(resources_dir / filename, overwrite_file=True)
    writer.save(image, dimension_order=dims)
    writer.close()

    with OmeTiffReader(resources_dir / filename) as test_output_reader:
        output = test_output_reader.data
        t = test_output_reader.size_t()
        c = test_output_reader.size_c()
        z = test_output_reader.size_z()
        y = test_output_reader.size_y()
        x = test_output_reader.size_x()

    os.remove(resources_dir / filename)

    assert output.shape == image.shape[1:]
    assert x == expected_x
    assert y == expected_y
    assert z == expected_z
    assert c == expected_c
    assert t == expected_t

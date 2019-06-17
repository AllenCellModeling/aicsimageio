#!/usr/bin/env python

import os
import unittest

from aicsimageio.readers import CziReader
from aicsimageio.tests.transformation import *


class TestCziReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with CziReader(
            os.path.join(dir_path, "..", "img", "T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi")
        ) as reader:
            z_index = m.floor(reader.size_z() / 2)
            c_index = m.floor(reader.size_c() / 2)
            t_index = m.floor(reader.size_t() / 2)
            cls.load = reader.data
            cls.load_image = np.ndarray(
                [
                    1,
                    reader.size_t(),
                    reader.size_c(),
                    reader.size_z(),
                    reader.size_y(),
                    reader.size_x(),
                    1,
                ]
            )

        # Set up for file with no Z (only 5 Dimension)
        with CziReader(
            os.path.join(dir_path, "..", "img", "test_5_dimension.czi")
        ) as reader:
            cls.loadNoZ = reader.data

    @classmethod
    def tearDownClass(cls):
        pass

    """
    Test to check the dimensionality of the array loaded by CziReader
    This should be 4 dimensional, ZCYX, or 5 dimensional, TZCYX.
    """

    def test_loadDimension(self):
        self.assertTrue(len(self.load.shape) == 7, msg="Shape is not 7 D")

    """
    Test to check the dimensionality of the array loaded by CziReader
    This should be 5 dimensional, TZCYX
    """

    def test_loadImageDimensions(self):
        self.assertEqual(len(self.load_image.shape), 7)

    """
    Test to check the dimensionality of the array loaded by CziReader for an image with no Z dimension
    This should be 5 dimensional, TZCYX
    """

    def test_loadImageDimensionsNoZ(self):
        self.assertEqual(len(self.loadNoZ.shape), 5)

    """
    Test to check the dimensionality of the array loaded by CziReader for an image with no Z dimension
    This should be 4 dimensional, ZCYX, or 5 dimensional, TZCYX.
    """

    def test_loadDimensionNoZ(self):
        self.assertTrue(
            len(self.loadNoZ.shape) == 4 or len(self.loadNoZ.shape) == 5,
            msg="Shape is not 4 or 5",
        )

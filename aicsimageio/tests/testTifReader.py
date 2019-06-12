#!/usr/bin/env python

# authors: Dan Toloudis     danielt@alleninstitute.org
#          Zach Crabtree    zacharyc@alleninstitute.org

import math as m
import os
import unittest
import numpy as np

from aicsimageio.readers import TifReader


class TestTifReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        cls.reader0 = TifReader(os.path.join(cls.dir_path, 'img', 'img40_1_dna.tif'))
        cls.reader1 = TifReader(os.path.join(cls.dir_path, 'img', 'TestXYCZ_imagej.tif'))


    """
    Test to check the dimensionality of the array loaded by TifReader
    This should be 2 dimensional, YX
    """
    def test_tifLoadImageDimension(self):
        z_index = int(m.floor(self.reader0.size_z() / 2))
        self.assertEqual(len(self.reader0.load_slice(z=z_index).shape), 2)

    """
    Test to check the dimensionality of the array loaded by TifReader
    This should be 5 dimensional, TCZYX where C = rgb channels
    """
    def test_tifLoadDimension(self):
        self.assertEqual(len(self.reader0.load().shape), 5)

    """
    Test to check that loading the image through load() and load_image() doesn't
    change the output or dimensionality
    """
    def test_tifLoadComparisonTest(self):
        # TZCYX
        loaded_image_slices = np.ndarray([1, self.reader0.size_z(), 1, self.reader0.size_y(),
                                          self.reader0.size_x()], dtype=self.reader0.dtype())
        for i in range(self.reader0.size_z()):
            loaded_image_slices[0, i, 0, :, :] = self.reader0.load_slice(z=i)
        loaded_image = self.reader0.load()

        self.assertTrue(np.array_equal(loaded_image, loaded_image_slices))

    """
    Test to check the dimensionality of a tif file that has ImageJ metadata embedded.
    This shows that a plain tif can be loaded as 4D from tifffile.py
    """
    def test_tifLoadImageJ(self):
        image = self.reader1.load()
        self.assertEqual(len(image.shape), 5)
        self.assertTrue(image.shape[0] == 1)
        self.assertTrue(image.shape[1] == 4)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(image.shape[3] == 300)
        self.assertTrue(image.shape[4] == 400)

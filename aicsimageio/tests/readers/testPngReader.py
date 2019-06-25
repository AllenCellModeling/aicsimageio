#!/usr/bin/env python

import os
import unittest

from aicsimageio.readers.default_reader import DefaultReader


class TestPngReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with DefaultReader(os.path.join(dir_path, "..", 'img', 'img40_1.png')) as reader:
            cls.input = reader.data

    """
    Test to assure that the png is always read in as a 3 dimensional array
    Will return XYC
    """
    def test_shapeOutput(self):
        self.assertEqual(len(self.input.shape), 3)
        self.assertEqual(self.input.shape[2], 3)

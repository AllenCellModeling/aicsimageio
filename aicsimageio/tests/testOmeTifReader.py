#!/usr/bin/env python

# authors: Dan Toloudis     danielt@alleninstitute.org
#          Zach Crabtree    zacharyc@alleninstitute.org

import os
import unittest

import numpy as np

from aicsimageio import OmeTifReader


class TestOmeTifReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        with OmeTifReader(os.path.join(cls.dir_path, 'img', 'img40_1.ome.tif')) as reader:
            cls.load = reader.load()
            cls.slice = reader.load_slice()
            cls.load_image = np.ndarray([reader.size_t(), reader.size_z(), reader.size_c(), reader.size_y(),
                                         reader.size_x()])
            for i in range(reader.size_t()):
                for j in range(reader.size_z()):
                    for k in range(reader.size_c()):
                        cls.load_image[i, j, k, :, :] = reader.load_slice(t=i, z=j, c=k)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_omeTifLoadShapeCorrectDimensions(self):
        self.assertEqual(len(self.load.shape), 5)

    def test_omeTifLoadSliceShapeCorrectDimensions(self):
        self.assertEqual(len(self.slice.shape), 2)

    def test_omeTifLoadCompareLoadImage(self):
        self.assertTrue(np.array_equal(self.load, self.load_image))

    def test_omeTifEmptyFileError(self):
        with self.assertRaises(Exception):
            with OmeTifReader('fakefile') as reader:
                reader.load()

    def test_notOmeTifFile(self):
        with self.assertRaises(Exception):
            with OmeTifReader(os.path.join(self.dir_path, 'img', 'T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi')) as reader:
                reader.load()

    def test_loadSampleOmeTif(self):
        names = [
            'single-channel.ome.tif',
            'multi-channel.ome.tif',
            'z-series.ome.tif',
            'time-series.ome.tif',
            'multi-channel-z-series.ome.tif',
            'multi-channel-time-series.ome.tif',
            '4D-series.ome.tif',
            'multi-channel-4D-series.ome.tif']
        dims = [
            #T  Z  C    Y    X
            (1, 1, 1, 167, 439),
            (1, 1, 3, 167, 439),
            (1, 5, 1, 167, 439),
            (7, 1, 1, 167, 439),
            (1, 5, 3, 167, 439),
            (7, 1, 3, 167, 439),
            (7, 5, 1, 167, 439),
            (7, 5, 3, 167, 439)
        ]
        for i, x in enumerate(names):
            with OmeTifReader(os.path.join(self.dir_path, 'img', x)) as reader:
                data = reader.load()
                self.assertEqual(data.shape, dims[i])


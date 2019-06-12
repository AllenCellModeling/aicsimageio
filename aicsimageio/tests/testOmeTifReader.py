#!/usr/bin/env python

import os
import unittest

from aicsimageio.readers import OmeTifReader


class TestOmeTifReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        with OmeTifReader(
            os.path.join(cls.dir_path, "img", "img40_1.ome.tif")
        ) as reader:
            cls.load = reader.data
            cls.load_sizes = [
                reader.size_t(),
                reader.size_c(),
                reader.size_z(),
                reader.size_y(),
                reader.size_x(),
            ]
            cls.dims = reader.dims
            cls.metadata = reader.metadata

    @classmethod
    def tearDownClass(cls):
        pass

    def test_omeTifLoadShapeCorrectDimensions(self):
        self.assertEqual(len(self.load.shape), 4)

    def test_omeTifEmptyFileError(self):
        with self.assertRaises(Exception):
            with OmeTifReader("fakefile") as reader:
                reader.data

    def test_notOmeTifFile(self):
        with self.assertRaises(Exception):
            with OmeTifReader(
                os.path.join(
                    self.dir_path, "img", "T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi"
                )
            ) as reader:
                reader.data

    def test_loadSampleOmeTif(self):
        names = [
            "single-channel.ome.tif",
            "multi-channel.ome.tif",
            "z-series.ome.tif",
            "time-series.ome.tif",
            "multi-channel-z-series.ome.tif",
            "multi-channel-time-series.ome.tif",
            "4D-series.ome.tif",
            "multi-channel-4D-series.ome.tif",
        ]
        dims = [
            (167, 439),
            (3, 167, 439),
            (5, 167, 439),
            (7, 167, 439),
            (3, 5, 167, 439),
            (7, 3, 167, 439),
            (7, 5, 167, 439),
            (7, 3, 5, 167, 439),
        ]
        for i, x in enumerate(names):
            with OmeTifReader(os.path.join(self.dir_path, "img", x)) as reader:
                data = reader.data
                self.assertEqual(data.shape, dims[i])

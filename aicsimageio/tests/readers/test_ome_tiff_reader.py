#!/usr/bin/env python

import os
import unittest

from aicsimageio.readers import OmeTiffReader


class TestOmeTifReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        with OmeTiffReader(
            os.path.join(cls.dir_path, "..", "resources", "s_1_t_1_c_1_z_1.ome.tiff")
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

    def test_omeTifLoadShapeCorrectDimensions(self):
        self.assertEqual(len(self.load.shape), 2)

    def test_omeTifEmptyFileError(self):
        with self.assertRaises(Exception):
            with OmeTiffReader("fakefile") as reader:
                assert reader.data

    def test_notOmeTifFile(self):
        with self.assertRaises(Exception):
            with OmeTiffReader(
                os.path.join(
                    self.dir_path, "resources", "s_1_t_1_c_1_z_1.czi"
                )
            ) as reader:
                assert reader.data

    def test_loadSampleOmeTif(self):
        names = [
            "s_1_t_1_c_1_z_1.ome.tiff",
            "s_3_t_1_c_3_z_5.ome.tiff"
        ]
        dims = [
            (325, 475),
            (5, 3, 325, 475),
        ]
        for i, x in enumerate(names):
            with OmeTiffReader(os.path.join(self.dir_path, "..", "resources", x)) as reader:
                assert reader.is_ome()
                data = reader.data
                self.assertEqual(data.shape, dims[i])

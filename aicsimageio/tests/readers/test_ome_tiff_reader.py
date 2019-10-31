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
            "s_1_t_1_c_10_z_1.ome.tiff",
            "s_3_t_1_c_3_z_5.ome.tiff"
        ]
        dims = [
            (325, 475),
            (10, 1736, 1776),
            (5, 3, 325, 475),
        ]
        dim_orders = [
            "YX",
            "CYX",  # Inferred from metadata not shape
            "ZCYX",
        ]
        physical_pixel_sizes = [
            (1.0833333333333333, 1.0833333333333333, 1.0),
            (1.0, 1.0, 1.0),
            (1.0833333333333333, 1.0833333333333333, 1.0),
        ]
        for i, x in enumerate(names):
            with OmeTiffReader(os.path.join(self.dir_path, "..", "resources", x)) as reader:
                assert reader.is_ome()
                data = reader.data
                self.assertEqual(data.shape, dims[i])
                self.assertEqual(reader.dims, dim_orders[i])
                self.assertEqual(reader.get_physical_pixel_size(), physical_pixel_sizes[i])

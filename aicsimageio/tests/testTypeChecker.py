import unittest
import os

from aicsimageio import TypeChecker

class TestTypeChecker(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.__init__(self)

    def runTest(self):
        # this method has to be included in a testgroup in order for it be run
        self.assertTrue(True)

    def test_png(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checker = TypeChecker(os.path.join(dir_path, 'img', 'img40_1.png'))
        self.assertFalse(checker.is_czi)
        self.assertFalse(checker.is_tiff)
        self.assertFalse(checker.is_ome)

    def test_tiff(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checker = TypeChecker(os.path.join(dir_path, 'img', 'img40_1_dna.tif'))
        self.assertFalse(checker.is_czi)
        self.assertTrue(checker.is_tiff)
        self.assertFalse(checker.is_ome)

    def test_ome_tiff(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checker = TypeChecker(os.path.join(dir_path, 'img', 'img40_1.ome.tif'))
        self.assertFalse(checker.is_czi)
        self.assertTrue(checker.is_tiff)
        self.assertTrue(checker.is_ome)

    def test_czi(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checker = TypeChecker(os.path.join(dir_path, 'img', 'T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi'))
        self.assertTrue(checker.is_czi)
        self.assertFalse(checker.is_tiff)
        self.assertFalse(checker.is_ome)

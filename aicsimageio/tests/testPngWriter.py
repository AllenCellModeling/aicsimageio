#!/usr/bin/env python

# Author: Zach Crabtree zacharyc@alleninstitute.org

import os
import unittest

from aicsimageio import PngReader, PngWriter
from .transformation import *


class TestPngWriter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        cls.file = os.path.join(cls.dir_path, 'img', 'pngwriter_test_output.png')
        cls.writer = PngWriter(cls.file, overwrite_file=True)
        # unfortunately, the rounding is necessary - scipy.fromimage() only returns integer values for pixels
        cls.image = np.round(transform(np.random.rand(40, 3, 128, 256)))
        if not os.path.isfile(cls.file):
            open(cls.file, 'a').close()

    @classmethod
    def tearDownClass(cls):
        cls.writer.close()
        os.remove(cls.file)

    """
    Test saves an image and compares it with a previously saved image.
    This iotest should assure that the png save() method does not transpose any dimensions as it saves
    """
    def test_pngSaveComparison(self):
        self.writer.save(self.image.astype('uint8'))
        reader = PngReader(self.file)
        output_image = reader.load()
        self.assertTrue(np.array_equal(self.image, output_image))
        reader.close()

    """
    Test saves an image with various z, c, and t.
    The extra parameters should not change the output from save()'s output
    """
    def test_pngSaveImageComparison(self):
        self.writer.save_slice(self.image.astype('uint8'), z=1, c=2, t=3)
        reader = PngReader(self.file)
        output_image = reader.load()
        self.assertTrue(np.array_equal(self.image, output_image))
        reader.close()

    """
    Test to check if save() can overwrite a file
    """
    def test_overwriteFile(self):
        print(self.file)
        with PngWriter(self.file, overwrite_file=True) as writer:
            writer.save(self.image.astype('uint8'))

    """
    Test to check if save() will raise error when user does not want to overwrite a file that exists
    """
    def test_dontOverwriteFile(self):
        with self.assertRaises(Exception):
            with PngWriter(self.file) as writer:
                writer.save(self.image)

    """
    Test to check if save() silently no-ops when user does not want to overwrite exiting file
    """
    def test_noopOverwriteFile(self):
        with open(self.file, 'w') as f:
            f.write("test")
        with PngWriter(self.file, overwrite_file=False) as writer:
            writer.save(self.image)
        with open(self.file, 'r') as f:
            line = f.readline().strip()
            self.assertEqual("test", line)

    """
    Test saves an image with a single xy plane
    This iotest assures that the pixels are written to the correct orientation
    """
    def test_twoDimensionalImages(self):
        image = np.ndarray([2, 2], dtype=np.uint8)
        image[0, 0] = 255
        image[0, 1] = 0
        image[1, 0] = 0
        image[1, 1] = 255
        self.writer.save(image)
        with PngReader(self.file) as reader:
            loaded_image = reader.load()
            self.assertTrue(np.array_equal(image, loaded_image))

    """
    Test saves an image with a single xy plane, but gives one channel
    This iotest assures that the channels are repeated when written with less than 3 channels
    """
    def test_threeDimensionalImages(self):
        image = np.zeros([1, 2, 2], dtype=np.uint8)
        image[0, 0, 0] = 255
        image[0, 0, 1] = 0
        image[0, 1, 0] = 0
        image[0, 1, 1] = 255
        self.writer.save(image)
        with PngReader(self.file) as reader:
            all_channels = reader.load()
            channel_r = all_channels[0, :, :]
            channel_g = all_channels[1, :, :]
            channel_b = all_channels[2, :, :]
            self.assertTrue(np.array_equal(channel_r, channel_g) and np.array_equal(channel_g, channel_b) and np.array_equal(channel_r, image[0, :, :]))

    """
    Test attempts to save an image with zcyx dims
    This should fail because the pngwriter does not accept images with more than 3 dims
    """
    def test_fourDimensionalImages(self):
        image = np.random.rand(1, 2, 3, 4)
        # the pngwriter cannot handle 4d images, and should thus throw an error
        with self.assertRaises(Exception):
            self.writer.save(image)

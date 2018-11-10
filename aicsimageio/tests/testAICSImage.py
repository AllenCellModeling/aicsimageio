# author: Zach Crabtree zacharyc@alleninstitute.org

import unittest
import numpy as np
import random
import os
import pathlib

from aicsimageio import AICSImage


class TestAicsImage(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.__init__(self)

    def runTest(self):
        # this method has to be included in a testgroup in order for it be run
        self.assertTrue(True)


    def test_transposedOutput(self):
        # arrange
        input_shape = random.sample(range(1, 10), 5)
        stack = np.zeros(input_shape)
        image = AICSImage(stack, dims="TCZYX")
        # act
        output_array = image.get_image_data("XYZCT")
        stack = stack.transpose((4, 3, 2, 1, 0))
        # assert
        self.assertEqual(output_array.shape, stack.shape)

    def test_p_transpose(self):
        input_shape = random.sample(range(1, 10), 5)
        stack = np.zeros(input_shape)
        ## Load randsom shape matrix as image with defined order
        image = AICSImage(stack, dims="TCZYX")
        # act
        ## Shuffle the AICS image matrix order
        output_array = AICSImage.p_transpose(image.data, image.dims, "YZXCT")
        ## Shuffle the input matrix block the same way
        stack = stack.transpose((3, 2, 4, 1, 0))
        # assert
        self.assertEqual(output_array.shape, stack.shape)

    def test_transposed2Output(self):
        # arrange
        ## Create a random shape matrix
        input_shape = random.sample(range(1, 10), 5)
        stack = np.zeros(input_shape)
        ## Load randsom shape matrix as image with defined order
        image = AICSImage(stack, dims="TCZYX")
        # act
        ## Shuffle the AICS image matrix order
        output_array = image.get_image_data("YZXCT")
        ## Shuffle the input matrix block the same way
        stack = stack.transpose((3, 2, 4, 1, 0))
        # assert
        self.assertEqual(output_array.shape, stack.shape)
        #self.assertEqual(output_array.all, stack.all)


    def test_slicedOutput(self):
        # arrange
        input_shape = random.sample(range(1, 20), 5)
        t_max, c_max = input_shape[0], input_shape[1]
        t_rand, c_rand = random.randint(0, t_max-1), random.randint(0, c_max-1)
        stack = np.zeros(input_shape)
        stack[t_rand, c_rand] = 1
        image = AICSImage(stack, dims="TCZYX")
        print("\ninput.shape: ", input_shape)
        print("\nimage.shape: ", image.shape)
        # act
        output_array = image.get_image_data("ZYX", T=t_rand, C=c_rand)
        print("output_array.shape: ", output_array.shape)
        # assert
        self.assertEqual(output_array.all(), 1)
        self.assertEqual(stack[t_rand, c_rand, :, :, :].shape, output_array.shape)

    def test_fewDimensions(self):
        input_shape = random.sample(range(1, 20), 3)
        stack = np.zeros(input_shape)
        image = AICSImage(stack, dims="CTX")
        self.assertEqual(image.data.shape, image.shape)

    def test_fromFileName(self):
        # arrange and act
        dir_path = os.path.dirname(os.path.realpath(__file__))
        image = AICSImage(os.path.join(dir_path, 'img', 'img40_1.ome.tif'))
        # assert
        self.assertIsNotNone(image)

    def test_fromInvalidFileName(self):
        # arrange, act, assert
        with self.assertRaises(IOError):
            AICSImage("fakeimage.ome.tif")

    def test_fromInvalidDataType(self):
        with self.assertRaises(TypeError):
            AICSImage(pathlib.Path('foo.tiff'))

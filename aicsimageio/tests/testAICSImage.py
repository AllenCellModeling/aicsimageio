from pathlib import Path
import random

import numpy as np
import pytest

from aicsimageio.aics_image import AICSImage


TEST_DATA_DIR = Path(__file__).parent / "img"
TEST_OME = TEST_DATA_DIR / "img40_1.ome.tif"


class ImgContainer(object):
    def __init__(self, channels: int = 5, dims: str = "TCZYX"):
        self.input_shape = random.sample(range(1, 10), channels)
        stack = np.zeros(self.input_shape)
        self.dims = dims
        self.order = {
            c: i for i, c in enumerate(dims)
        }  # {'T': 0, 'C': 1, 'Z': 2, 'Y': 3, 'X': 4}
        self.image = AICSImage(stack, dims=self.dims)

    def remap(self, seq):
        return [self.order[c] for c in seq]

    def shuffle_shape(self, seq):
        new_shape = [self.input_shape[self.order[c]] for c in seq]
        return tuple(new_shape)

    def get_trand_crand(self):
        tmax = self.input_shape[self.order["T"]]
        cmax = self.input_shape[self.order["C"]]
        trand, crand = random.randint(0, tmax - 1), random.randint(0, cmax - 1)
        return trand, crand


@pytest.fixture
def example_img5():
    return ImgContainer()


@pytest.fixture
def example_img3ctx():
    return ImgContainer(3, "CTX")


def test_helper_class(example_img5):
    assert example_img5.remap("XYZCT") == [4, 3, 2, 1, 0]


def test_transposed_output(example_img5):
    output_array = example_img5.image.get_image_data("XYZCT")
    stack_shape = example_img5.shuffle_shape("XYZCT")
    assert output_array.shape == stack_shape


def test_transpose(example_img5):
    output_array = AICSImage._AICSImage__transpose(
        example_img5.image.data, example_img5.dims, "YZXCT"
    )
    stack_shape = example_img5.shuffle_shape("YZXCT")
    assert output_array.shape == stack_shape


def test_slice(example_img5):
    """
    Test if the private function AICSImage.__get_slice works correctly for a simple example
    """
    trand, crand = example_img5.get_trand_crand()
    im_shape = example_img5.image.shape
    # reset the slice in the data to have value 1
    example_img5.image.data[trand, crand] = 1
    slice_shape = [
        im_shape[i] for i in range(2, 5)
    ]  # take the shape for the data cube (3D)
    slice_dict = {
        "T": trand,
        "C": crand,
        "Z": slice(None, None),
        "Y": slice(None, None),
        "X": slice(None, None),
    }
    # slice_dict defines the sub-block to pull out
    output_array = AICSImage._AICSImage__get_slice(
        example_img5.image.data, "TCZYX", slice_dict
    )
    assert output_array.shape == tuple(slice_shape)  # check the shape is right
    assert output_array.all() == 1  # check the values are all 1 for the sub-block


def test_transposed_output_2(example_img5):
    order = "TCZYX"
    for _ in range(0, 20):
        new_order = "".join(random.sample(order, len(order)))
        image = example_img5.image.get_image_data(new_order)
        shape = example_img5.shuffle_shape(new_order)
        assert image.shape == shape


def test_sliced_output(example_img5):
    t_rand, c_rand = example_img5.get_trand_crand()
    example_img5.image.data[
        t_rand, c_rand
    ] = 1  # force the data block to 1's, (was 0's)
    output_array = example_img5.image.get_image_data("ZYX", T=t_rand, C=c_rand)
    assert output_array.all() == 1
    assert example_img5.image.data[t_rand, c_rand, :, :, :].shape == output_array.shape


def test_multiple_access(example_img5):
    t_rand, c_rand = example_img5.get_trand_crand()
    example_img5.image.data[
        t_rand, c_rand
    ] = 1  # force the data block to 1's, (was 0's)
    output_array = example_img5.image.get_image_data("ZYX", T=t_rand, C=c_rand)
    assert output_array.all() == 1
    assert example_img5.image.data[t_rand, c_rand, :, :, :].shape == output_array.shape
    output_two = example_img5.image.get_image_data("TCXYZ")
    out_two_shape = example_img5.shuffle_shape("TCXYZ")
    assert output_two.shape == out_two_shape


def test_few_dimensions(example_img3ctx):
    image = example_img3ctx.image
    assert image.data.shape == image.shape


def test_get_image_data_small_data(example_img3ctx):
    image = example_img3ctx.image.get_image_data("TCX")
    assert image.shape == example_img3ctx.shuffle_shape("TCX")


def test_bad_query(example_img3ctx):
    # constructed by fixture with "CTX"
    image = example_img3ctx.image.get_image_data()
    # returns a 5D block
    assert image.shape != example_img3ctx.shuffle_shape("TCX")


@pytest.mark.parametrize(
    "filepath",
    [
        TEST_OME,
        str(TEST_OME),
        pytest.param(
            TEST_DATA_DIR, marks=pytest.mark.raises(exception=IsADirectoryError)
        ),
        pytest.param(
            "fakeimage.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)
        ),
        pytest.param(
            b"not-a-string-path", marks=pytest.mark.raises(exception=TypeError)
        ),
        pytest.param(
            "/This/is/a/bogus/file.ome.tif",
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
    ],
)
def test_file_exceptions(filepath):
    image = AICSImage(filepath)
    assert image is not None

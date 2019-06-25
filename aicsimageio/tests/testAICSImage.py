import random

import numpy as np
import pytest
from aicsimageio import AICSImage, readers


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


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_helper_class(example_img5):
    assert example_img5.remap("XYZCT") == [4, 3, 2, 1, 0]


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_transposed_output(example_img5):
    output_array = example_img5.image.get_image_data("XYZCT")
    stack_shape = example_img5.shuffle_shape("XYZCT")
    assert output_array.shape == stack_shape


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_transpose(example_img5):
    output_array = AICSImage._AICSImage__transpose(
        example_img5.image.data, example_img5.dims, "YZXCT"
    )
    stack_shape = example_img5.shuffle_shape("YZXCT")
    assert output_array.shape == stack_shape


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_transposed_output_2(example_img5):
    order = "TCZYX"
    for _ in range(0, 20):
        new_order = "".join(random.sample(order, len(order)))
        image = example_img5.image.get_image_data(new_order)
        shape = example_img5.shuffle_shape(new_order)
        assert image.shape == shape


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_get_image_data_small_data(example_img3ctx):
    image = example_img3ctx.image.get_image_data("TCX")
    assert image.shape == example_img3ctx.shuffle_shape("TCX")


@pytest.mark.parametrize("filepath", [
        "img40_1.ome.tif",
        pytest.param("fakeimage.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
        pytest.param("a/bogus/file.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
])
def test_file_exceptions(image_dir, filepath):
    f = image_dir / filepath
    AICSImage(f)


def test_file_passed_was_directory(image_dir):
    with pytest.raises(IsADirectoryError):
        AICSImage(image_dir)


def test_file_passed_was_byte_string(image_dir):
    with pytest.raises(ValueError):
        img = AICSImage(b"not-a-valid-image-byte-array")
        img.data

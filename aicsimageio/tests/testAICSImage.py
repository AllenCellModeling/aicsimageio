from pathlib import Path
import random

import numpy as np
import pytest

from aicsimageio.aics_image import AICSImage
from aicsimageio.exceptions import ConflictingArgsError


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


@pytest.mark.skip(reason='Waiting on ndarray support in DefaultReader')
def test_helper_class(example_img5):
    assert example_img5.remap("XYZCT") == [4, 3, 2, 1, 0]


# test _reshape_data which is called when data is accessed
@pytest.mark.parametrize("kwargs,expected", [
    ({'data': np.zeros((10, 1, 5, 6, 200, 400)), 'given_dims': "STCZYX", 'return_dims': "CSZYX"}, (5, 10, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "STCZYX"}, (1, 1, 1, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "ZCYSXT"}, (6, 1, 200, 1, 400, 1)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXC', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=IndexError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZ', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgsError)),
])
def test_reshape_data(kwargs, expected):
    ans = AICSImage._reshape_data(**kwargs)
    assert ans.shape == expected


@pytest.mark.parametrize("kwargs,expected", [
    ({'data': np.zeros((1, 2, 3, 4, 5, 6)), 'given_dims': "STCZYX", 'return_dims': "XYZCTS"}, (6, 5, 4, 3, 2, 1)),
    ({'data': np.zeros((1, 2, 3)), 'given_dims': "ZYX", 'return_dims': "ZXY"}, (1, 3, 2)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXC'}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgsError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZ'}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgsError)),
])
def test_transpose_to_dims(kwargs, expected):
    data = AICSImage._transpose_to_dims(**kwargs)
    assert data.shape == expected


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
        TEST_OME,
        str(TEST_OME),
        pytest.param(TEST_DATA_DIR, marks=pytest.mark.raises(exception=IsADirectoryError)),
        pytest.param("fakeimage.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
        # pytest.param(b"not-a-valid-image-byte-array", marks=pytest.mark.raises(exception=ValueError)),
        pytest.param("/This/is/a/bogus/file.ome.tif", marks=pytest.mark.raises(exception=FileNotFoundError)),
])
def test_file_exceptions(filepath):
    image = AICSImage(filepath)
    assert image is not None


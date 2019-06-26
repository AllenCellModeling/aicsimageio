import pytest
import numpy as np
from aicsimageio.transforms import transpose_to_dims, reshape_data
from aicsimageio.exceptions import ConflictingArgumentsError


# test _reshape_data which is called when data is accessed
@pytest.mark.parametrize("input_dict, expected", [
    ({'data': np.zeros((10, 1, 5, 6, 200, 400)), 'given_dims': "STCZYX", 'return_dims': "CSZYX"}, (5, 10, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "STCZYX"}, (1, 1, 1, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "ZCYSXT"}, (6, 1, 200, 1, 400, 1)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXC', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=IndexError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZ', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZX', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
])
def test_reshape_data_shape(input_dict, expected):
    # since transpose to dims has keywords for all arguments I'm using a dictionary
    # and expanding it with the ** operator
    ans = reshape_data(**input_dict)
    assert ans.shape == expected


@pytest.mark.parametrize("input_dict, idx_in, idx_out", [
    ({'data': np.random.rand(10, 1, 5, 6, 200, 400), 'given_dims': "STCZYX", 'return_dims': "CSZYX"},
     (5, 0, 3, 3, ...), (3, 5, 3, ...)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "STCZYX"},
     (..., 100, 200), (0, 0, 0, ..., 100, 200)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "ZCYSXT"},
     (3, 100, ...), (3, 0, 100, 0, ..., 0)),
])
def test_reshape_data_values(input_dict, idx_in, idx_out):
    # for explanation of input_dict see first use above
    data = input_dict['data']
    slice_in = data[idx_in]
    ans = reshape_data(**input_dict)
    for a, b in zip(slice_in.flat, ans[idx_out].flat):
        assert a == b


@pytest.mark.parametrize("input_dict, expected", [
    ({'data': np.zeros((1, 2, 3, 4, 5, 6)), 'given_dims': "STCZYX", 'return_dims': "XYZCTS"}, (6, 5, 4, 3, 2, 1)),
    ({'data': np.zeros((1, 2, 3)), 'given_dims': "ZYX", 'return_dims': "ZXY"}, (1, 3, 2)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXC'}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZ'}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
])
def test_transpose_to_dims(input_dict, expected):
    # for explanation of input_dict see first use above
    data = transpose_to_dims(**input_dict)
    assert data.shape == expected

import pytest
import numpy as np
from aicsimageio.transforms import transpose_to_dims, reshape_data
from aicsimageio.exceptions import ConflictingArgsError


# test _reshape_data which is called when data is accessed
@pytest.mark.parametrize("kwargs,expected", [
    ({'data': np.zeros((10, 1, 5, 6, 200, 400)), 'given_dims': "STCZYX", 'return_dims': "CSZYX"}, (5, 10, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "STCZYX"}, (1, 1, 1, 6, 200, 400)),
    ({'data': np.zeros((6, 200, 400)), 'given_dims': "ZYX", 'return_dims': "ZCYSXT"}, (6, 1, 200, 1, 400, 1)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXC', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=IndexError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZ', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgsError)),
    pytest.param({'data': np.zeros((6, 200, 400)), 'given_dims': 'ZYX', 'return_dims': 'TYXCZX', 'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgsError)),
])
def test_reshape_data(kwargs, expected):
    ans = reshape_data(**kwargs)
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
    data = transpose_to_dims(**kwargs)
    assert data.shape == expected

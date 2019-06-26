import pytest
import numpy as np
from aicsimageio.transforms import transpose_to_dims, reshape_data
from aicsimageio.exceptions import ConflictingArgumentsError


# test _reshape_data which is called when data is accessed
@pytest.mark.parametrize("data, given_dims, return_dims, other_args, expected", [
    (np.zeros((10, 1, 5, 6, 200, 400)), "STCZYX", "CSZYX", {}, (5, 10, 6, 200, 400)),
    (np.zeros((6, 200, 400)), "ZYX", "STCZYX", {}, (1, 1, 1, 6, 200, 400)),
    (np.zeros((6, 200, 400)), "ZYX", "ZCYSXT", {}, (6, 1, 200, 1, 400, 1)),
    pytest.param(np.zeros((6, 200, 400)), 'ZYX', 'TYXC', {'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=IndexError)),
    pytest.param(np.zeros((6, 200, 400)), 'ZYX', 'TYXCZ', {'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
    pytest.param(np.zeros((6, 200, 400)), 'ZYX', 'TYXCZX', {'Z': 7}, 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
])
def test_reshape_data_shape(data, given_dims, return_dims, other_args, expected):
    # the other_args are being used to pass slice specific information by expanding with the ** operator
    ans = reshape_data(data=data, given_dims=given_dims, return_dims=return_dims, **other_args)
    assert ans.shape == expected


@pytest.mark.parametrize("data, given_dims, return_dims, idx_in, idx_out", [
    (np.random.rand(10, 1, 5, 6, 200, 400), "STCZYX", "CSZYX", (5, 0, 3, 3, ...), (3, 5, 3, ...)),
    (np.zeros((6, 200, 400)), "ZYX", "STCZYX", (..., 100, 200), (0, 0, 0, ..., 100, 200)),
    (np.zeros((6, 200, 400)), "ZYX", "ZCYSXT", (3, 100, ...), (3, 0, 100, 0, ..., 0)),
])
def test_reshape_data_values(data, given_dims, return_dims, idx_in, idx_out):
    slice_in = data[idx_in]
    ans = reshape_data(data=data, given_dims=given_dims, return_dims=return_dims)
    for a, b in zip(slice_in.flat, ans[idx_out].flat):
        assert a == b


@pytest.mark.parametrize("data, given_dims, return_dims, expected", [
    (np.zeros((1, 2, 3, 4, 5, 6)), "STCZYX", "XYZCTS", (6, 5, 4, 3, 2, 1)),
    (np.zeros((1, 2, 3)), "ZYX", "ZXY", (1, 3, 2)),
    pytest.param(np.zeros((6, 200, 400)), 'ZYX', 'TYXC', 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
    pytest.param(np.zeros((6, 200, 400)), 'ZYX', 'TYXCZ', 5,
                 marks=pytest.mark.raises(exception=ConflictingArgumentsError)),
])
def test_transpose_to_dims(data, given_dims, return_dims, expected):
    data = transpose_to_dims(data=data, given_dims=given_dims, return_dims=return_dims)
    assert data.shape == expected

import numpy as np
import pytest
import logging

from tpgmm._core.arrays import subscript, identity_like, get_subarray


class TestSubscript:
    def test_all_integers(self):
        result = subscript(1, 2, 3)
        expected = (slice(1, 2), slice(2, 3), slice(3, 4))
        assert result == expected

    def test_mixed_integers_and_lists(self):
        result = subscript(1, [2, 3], 4)
        expected = (slice(1, 2), [2, 3], slice(4, 5))
        assert result == expected


class TestIdentityLike:
    def test_square_last_axes(self):
        input_array = np.zeros((3, 3, 3))
        result = identity_like(input_array)
        expected = np.stack([np.eye(3)] * 3)
        np.testing.assert_array_equal(result, expected)

    def test_non_square_last_axes(self, caplog):
        input_array = np.zeros((3, 3, 4))
        with caplog.at_level(logging.ERROR):
            result = identity_like(input_array)
        assert result is None


class TestGetSubarray:
    def test_basic_extraction(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        axes = [0, 1]
        indices = [[0, 1], [1, 2]]
        result = get_subarray(data, axes, indices)
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_invalid_indices_raises(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        axes = [0, 1]
        with pytest.raises(AssertionError):
            get_subarray(data, axes, [0, 1])


class TestBackwardCompatImports:
    """Verify backward-compatible imports from tpgmm.utils.arrays still work."""

    def test_imports_from_utils(self):
        from tpgmm.utils.arrays import subscript, identity_like, get_subarray
        assert callable(subscript)
        assert callable(identity_like)
        assert callable(get_subarray)

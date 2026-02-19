import numpy as np

from tpgmm.utils.casting import str_to_list, str_to_ndarray, ssv_to_ndarray


class TestStrToList:
    def test_converts_bracketed_string(self):
        input_string = "[apple, banana, cherry, date]"
        result = str_to_list(input_string)
        expected = ["apple", "banana", "cherry", "date"]
        assert result == expected


class TestStrToNdarray:
    def test_converts_bracketed_floats(self):
        input_string = "[1.1, 2.2, 3.3, 4.4, 5.5]"
        result = str_to_ndarray(input_string)
        expected = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        np.testing.assert_array_equal(result, expected)


class TestSsvToNdarray:
    def test_converts_space_separated_values(self):
        input_string = "1.1 2.2 3.3 4.4 5.5"
        result = ssv_to_ndarray(input_string)
        expected = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        np.testing.assert_array_equal(result, expected)
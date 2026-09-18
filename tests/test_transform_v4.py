import unittest

import numpy as np
import pandas as pd

from idopnetwork.curve_fitting import preprocess, data_transformation


class TransformV4Tests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'a': [-2., 0., 2.], 'b': [7., 7., 7.]}, index=['x', 'y', 'z'])

    def test_column_shift_and_constant_minmax(self):
        expected = pd.DataFrame({'a': [1., 3., 5.], 'b': [1., 1., 1.]}, index=self.frame.index)
        pd.testing.assert_frame_equal(preprocess(self.frame, 'Z_min_add1'), expected)
        expected = pd.DataFrame({'a': [0., .5, 1.], 'b': [0., 0., 0.]}, index=self.frame.index)
        pd.testing.assert_frame_equal(data_transformation(self.frame, 'Minmax_0_1'), expected)

    def test_base10_and_numeric_coercion(self):
        frame = pd.DataFrame({'a': ['0', '9', '99', 'invalid']})
        result = preprocess(frame, 'Log10_1p')
        np.testing.assert_allclose(result.a, [0., 1., 2., np.nan], equal_nan=True)
        np.testing.assert_allclose(preprocess(frame, 'None').a, [0., 9., 99., np.nan], equal_nan=True)
        self.assertEqual(frame.a.iloc[-1], 'invalid')

    def test_composed_transforms_and_unknown_method(self):
        shifted = preprocess(self.frame, 'Z_min_add1')
        result = preprocess(shifted, 'Log10_1p')
        pd.testing.assert_frame_equal(result, np.log10(shifted + 1))
        with self.assertRaises(ValueError):
            preprocess(self.frame, 'unknown')


if __name__ == '__main__':
    unittest.main()

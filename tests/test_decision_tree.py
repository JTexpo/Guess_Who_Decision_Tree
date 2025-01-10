import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from guess_who_dt.decision_tree import _get_tree_split

class TestGetTreeSplit(unittest.TestCase):
    def test_even_split(self):
        dataset = np.array([
            [1, 0.4, 0.6],
            [2, 0.3, 0.7],
            [3, 0.5, 0.5],
            [4, 0.6, 0.4],
            [5, 0.7, 0.3]
        ])
        expected_result = (2.0, 2)
        best_variance_reduction, best_feature_index =  _get_tree_split(dataset)

        self.assertEqual(expected_result[0], best_variance_reduction)
        self.assertEqual(expected_result[1], best_feature_index)

    def test_uneven_split(self):
        dataset = np.array([
            [1, 0.1, 0.9],
            [2, 0.2, 0.8],
            [3, 0.3, 0.7],
            [4, 0.4, 0.6],
            [5, 0.5, 0.5]
        ])
        expected_result = (1.0, 2)
        
        best_variance_reduction, best_feature_index =  _get_tree_split(dataset)

        self.assertEqual(expected_result[0], best_variance_reduction)
        self.assertEqual(expected_result[1], best_feature_index)

    def test_one_feature(self):
        dataset = np.array([
            [1, 0.5],
            [2, 0.5],
            [3, 0.5],
            [4, 0.5],
            [5, 0.5]
        ])
        expected_result = ( float("-inf") , -1)

        best_variance_reduction, best_feature_index =  _get_tree_split(dataset)

        self.assertEqual(expected_result[0], best_variance_reduction)
        self.assertEqual(expected_result[1], best_feature_index)

    def test_multiple_features(self):
        dataset = np.array([
            [1, 0.4, 0.6, 0.2],
            [2, 0.3, 0.7, 0.1],
            [3, 0.5, 0.5, 0.3],
            [4, 0.6, 0.4, 0.4],
            [5, 0.7, 0.3, 0.5]
        ])
        expected_result = (2.0, 2)

        best_variance_reduction, best_feature_index =  _get_tree_split(dataset)

        self.assertEqual(expected_result[0], best_variance_reduction)
        self.assertEqual(expected_result[1], best_feature_index)

    def test_all_values_equal_to_0_5(self):
        dataset = np.array([
            [1, 0.5, 0.5],
            [2, 0.5, 0.5],
            [3, 0.5, 0.5],
            [4, 0.5, 0.5],
            [5, 0.5, 0.5]
        ])
        expected_result = (float("-inf"), -1)

        best_variance_reduction, best_feature_index =  _get_tree_split(dataset)

        self.assertEqual(expected_result[0], best_variance_reduction)
        self.assertEqual(expected_result[1], best_feature_index)

    
if __name__ == '__main__':
    unittest.main()
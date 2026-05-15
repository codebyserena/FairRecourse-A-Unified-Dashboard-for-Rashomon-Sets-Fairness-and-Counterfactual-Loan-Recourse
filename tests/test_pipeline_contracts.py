import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preprocessing import get_feature_metadata
from fairness import _sensitive_groups
from recourse import _permitted_ranges, _select_query_index


class PipelineContractTests(unittest.TestCase):
    def test_feature_metadata_has_expected_groups(self):
        meta = get_feature_metadata()
        self.assertEqual(meta['target'], 'default')
        self.assertIn('sex', meta['sensitive'])
        self.assertIn('limit_bal', meta['mutable'])
        self.assertIn('age', meta['immutable'])

    def test_select_query_prefers_rejected_or_split_applicant(self):
        app_results = pd.DataFrame(
            [
                {'applicant_index': 1, 'approval_rate': 0.9, 'instability_score': 1.0},
                {'applicant_index': 2, 'approval_rate': 0.5, 'instability_score': 0.8},
                {'applicant_index': 3, 'approval_rate': 0.4, 'instability_score': 0.7},
            ]
        )
        self.assertEqual(_select_query_index(app_results), 2)

    def test_permitted_ranges_use_training_bounds(self):
        X_train = pd.DataFrame({'limit_bal': [10000, 20000, 50000]})
        self.assertEqual(_permitted_ranges(X_train, ['limit_bal']), {'limit_bal': [10000.0, 50000.0]})

    def test_age_sensitive_groups_are_binned(self):
        X = pd.DataFrame({'age': [25, 35, 45, 55, 65]})
        groups = _sensitive_groups(X, 'age').astype(str).tolist()
        self.assertEqual(groups, ['under_30', '30_39', '40_49', '50_59', '60_plus'])


if __name__ == '__main__':
    unittest.main()

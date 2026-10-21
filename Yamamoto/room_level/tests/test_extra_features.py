"""python -m unittest discover -s tests -p 'test_extra_features.py' -v"""
import importlib
import unittest

import numpy as np
import pandas as pd

feat = importlib.import_module("analysis.17_fixed_window_features")


def episodes(rows):
    d = pd.DataFrame(rows, columns=["building_id", "unit_key", "station", "layout", "city", "rent_group", "start_date", "end_date"])
    d["start_date"] = pd.to_datetime(d["start_date"])
    d["end_date"] = pd.to_datetime(d["end_date"])
    return d


class ExtraFeaturesTest(unittest.TestCase):
    def test_active_count_includes_self_only_while_listed(self):
        gid = np.array([0, 0, 1])
        start = np.array([10, 12, 10])
        end = np.array([20, 15, 30])
        # 日 13 に、グループ0では2件とも掲載中、グループ1では1件
        got = feat._active_count(gid, start, end, np.array([13, 13, 13]))
        self.assertEqual(got.tolist(), [2, 2, 1])
        # 終了日 = その日は掲載中ではない
        self.assertEqual(feat._active_count(gid, start, end, np.array([15, 15, 15])).tolist(), [1, 1, 1])

    def test_history_uses_only_past_within_window(self):
        ep = episodes([
            ("B1", "B1::1", "K", "1K", "c", "1", "2021-01-01", "2021-01-10"),
            ("B1", "B1::1", "K", "1K", "c", "1", "2021-02-01", "2021-02-10"),
            ("B1", "B1::1", "K", "1K", "c", "1", "2023-01-01", "2023-01-10"),  # 前の掲載から365日超
        ])
        out = feat.add_features(ep).sort_values("start_date").reset_index(drop=True)
        self.assertEqual(out["prev_n"].tolist(), [0, 1, 0])
        self.assertTrue(np.isnan(out.loc[0, "prev_len_log"]))
        self.assertAlmostEqual(out.loc[1, "prev_len_log"], np.log1p(9))
        self.assertTrue(np.isnan(out.loc[2, "prev_len_log"]))

    def test_placeholder_ids_are_not_treated_as_one_room(self):
        ep = episodes([
            ("1", "1::1", "K", "1K", "c", "1", "2021-01-01", "2021-01-10"),
            ("1", "1::1", "K", "1K", "c", "1", "2021-03-01", "2021-03-10"),
        ])
        out = feat.add_features(ep)
        self.assertEqual(out["prev_n"].tolist(), [0, 0])
        self.assertEqual(out["bldg_active_log"].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

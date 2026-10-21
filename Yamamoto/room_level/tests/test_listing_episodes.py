"""python -m unittest discover -s tests -p 'test_listing_episodes.py' -v"""
import unittest

import pandas as pd

from room_lib.listing_episodes import build_episodes, build_episodes_fast


def make(records):
    data = pd.DataFrame(records, columns=[
        "id", "building_id", "unit_id", "layout", "area", "station", "start_date", "end_date"
    ])
    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    data["start_year"] = data["start_date"].dt.year
    data["start_month"] = data["start_date"].dt.month
    return data


class ListingEpisodesTest(unittest.TestCase):
    def records(self):
        return make([
            ("a", "B", "101", "1K", 20, "K", "2020-01-01", "2020-01-10"),
            ("b", "B", "101", "1K", 20, "K", "2020-01-09", "2020-01-20"),
            ("c", "B", "101", "1K", 20, "K", "2020-02-10", "2020-02-20"),
            ("d", "", "101", "1K", 20, "K", "2020-01-01", "2020-01-10"),
            ("e", "B", "102", "1K", 20, "K", "2020-01-01", "2020-01-10"),
            ("f", "B", "102", "2K", 20, "K", "2020-01-05", "2020-01-10"),
        ])

    def test_merge_overlaps_keep_relisting_separate(self):
        ep, audit = build_episodes(self.records(), max_gap_days=1)
        self.assertEqual(len(ep), 5)
        self.assertEqual(ep["source_rows"].sum(), 6)
        self.assertEqual(int((ep["source_rows"] == 2).sum()), 1)
        self.assertEqual(int(audit["metadata_conflict"].sum()), 1)
        merged = ep.loc[ep["source_rows"] == 2].iloc[0]
        self.assertEqual(str(merged["start_date"].date()), "2020-01-01")
        self.assertEqual(str(merged["end_date"].date()), "2020-01-20")

    def test_missing_id_never_grouped(self):
        d = self.records().iloc[[3, 3]].copy()
        d.loc[d.index[1], "id"] = "another"
        ep, _ = build_episodes(d, max_gap_days=14)
        self.assertEqual(len(ep), 2)

    def test_station_difference_alone_does_not_block_merge(self):
        d = make([
            ("a", "B", "1", "1K", 20, "K", "2020-01-01", "2020-01-10"),
            ("b", "B", "1", "1K", 20, "N", "2020-01-05", "2020-01-20"),
        ])
        ep, audit = build_episodes(d, max_gap_days=1)
        self.assertEqual(len(ep), 1)
        self.assertTrue(bool(audit["station_differs"].any()))
        self.assertFalse(bool(audit["metadata_conflict"].any()))

    def test_conflict_in_old_listing_does_not_block_clean_overlap_elsewhere(self):
        # 2019年に別の間取りで載っていた履歴があっても、2020年の重なる2件は統合できる。
        d = make([
            ("old", "B", "1", "2K", 30, "K", "2019-01-01", "2019-01-10"),
            ("a", "B", "1", "1K", 20, "K", "2020-01-01", "2020-01-10"),
            ("b", "B", "1", "1K", 20, "K", "2020-01-05", "2020-01-20"),
        ])
        ep, _ = build_episodes(d, max_gap_days=1)
        self.assertEqual(len(ep), 2)
        self.assertEqual(int((ep["source_rows"] == 2).sum()), 1)

    def test_fast_builder_matches_original(self):
        cols = ["first_source_id", "start_date", "end_date", "source_rows"]
        for gap in (0, 1, 7, 14):
            slow, _ = build_episodes(self.records(), max_gap_days=gap)
            fast = build_episodes_fast(self.records(), max_gap_days=gap)
            a = slow[cols].sort_values(cols).reset_index(drop=True)
            b = fast[cols].sort_values(cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)

    def test_area_tolerance(self):
        small = make([
            ("a", "B", "1", "1K", 20.0, "K", "2020-01-01", "2020-01-10"),
            ("b", "B", "1", "1K", 21.5, "K", "2020-01-05", "2020-01-20"),
        ])
        large = make([
            ("a", "B", "1", "1K", 20.0, "K", "2020-01-01", "2020-01-10"),
            ("b", "B", "1", "1K", 25.0, "K", "2020-01-05", "2020-01-20"),
        ])
        self.assertEqual(len(build_episodes(small)[0]), 1)
        self.assertEqual(len(build_episodes(large)[0]), 2)


if __name__ == "__main__":
    unittest.main()

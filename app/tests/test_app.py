"""Smoke tests for the summary-data-backed application. Run: python -m unittest discover -s app/tests"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class ResearchAppSmokeTests(unittest.TestCase):
    def test_exactly_40_real_groups(self):
        response = main.list_historical_groups()
        rows = response["groups"]
        self.assertEqual(response["total_groups"], 40)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({(x["layout"], x["age_band"], x["rent_band"]) for x in rows}), 40)
        for row in rows:
            self.assertGreaterEqual(row["count"], 100)
            self.assertTrue(0 <= row["day12_next7_rate"] <= 1)
            self.assertEqual(row["level"], "hi" if row["day12_next7_rate"] >= .15 else "lo")


    def test_local_listing_pagination_and_filters(self):
        import pandas as pd
        local = pd.DataFrame([
            {"display_id": 1, "layout": "1K", "age": 7.0, "rent": 72000.0,
             "area": 22.0, "walk_distance": 6.0, "station": "国分寺",
             "start_month": 6, "pub_start_date": "2023-06-01",
             "pub_end_date": "2023-06-12", "duration": 11.0,
             "addr1_2_name": "国分寺市"},
            {"display_id": 2, "layout": "2LDK", "age": 14.0, "rent": 115000.0,
             "area": 55.0, "walk_distance": 12.0, "station": "西国分寺",
             "start_month": 7, "pub_start_date": "2023-07-01",
             "pub_end_date": "2023-07-30", "duration": 29.0,
             "addr1_2_name": "国分寺市"},
        ])
        with patch.object(main, "_local_listings", return_value=local):
            response = main.list_local_listings(page=1, limit=24)
            self.assertTrue(response["ready"])
            self.assertEqual(response["total"], 2)
            self.assertEqual(response["items"][0]["rent_man"], 7.2)
            result = main.list_local_listings(page=1, limit=1, layout="1K")
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["items"][0]["id"], 1)
            self.assertEqual(main.get_local_listing(2)["layout"], "2LDK")

    def test_missing_local_file_preserves_summary_mode(self):
        with patch.object(main, "_local_listings", return_value=None):
            result = main.list_local_listings()
            self.assertFalse(result["ready"])
            self.assertEqual(result["items"], [])

    def test_analysis_without_model_is_historical_not_fake_prediction(self):
        req = main.AnalyzeRequest(layout="1K", age=7, rent_man=7.2,
                                  area=22, walk_distance=6, station="国分寺",
                                  start_month=6, elapsed_day=12)
        with patch.object(main, "_predict_probabilities", return_value=None):
            outcome = main.analyze_property(req)
        self.assertEqual(outcome["mode"], "historical")
        self.assertIsNone(outcome["risk"])
        self.assertEqual(outcome["historical"]["count"], 1387)

    def test_analyze_with_model_result(self):
        req = main.AnalyzeRequest(layout="1K", age=7, rent_man=7.2,
                                  area=22, walk_distance=6, station="国分寺",
                                  start_month=6, elapsed_day=12)
        with patch.object(main, "_predict_probabilities", return_value=[.25]):
            outcome = main.analyze_property(req)
        self.assertEqual(outcome["mode"], "model")
        self.assertEqual(outcome["level"], "hi")
        self.assertAlmostEqual(outcome["risk"], .25)

    def test_model_failure_preserves_historical_statistics(self):
        req = main.AnalyzeRequest(layout="1K", age=7, rent_man=7.2,
                                  area=22, walk_distance=6, station="国分寺",
                                  start_month=6, elapsed_day=12)
        with patch.object(main, "_predict_probabilities", side_effect=RuntimeError("test failure")):
            outcome = main.analyze_property(req)
        self.assertEqual(outcome["mode"], "historical")
        self.assertIsNotNone(outcome["model_error"])
        self.assertEqual(outcome["historical"]["count"], 1387)


if __name__ == "__main__":
    unittest.main()

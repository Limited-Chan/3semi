import unittest
import numpy as np
import pandas as pd

from analyze import (
    prepare, exact_stats, selected_stat_value, apply_scope,
    parse_years, parse_types, group_statistics,
    apply_equal_width_bands, normalize_band_config, build_band_schema, apply_band_schema,
)


def row(**kw):
    base = dict(
        id="a",
        pub_start_date="2020-03-01",
        pub_end_date="2020-03-11",
        bukken_type="3101",
        kenchiku_date="201903",
        madori_number_all="2",
        madori_kind_all_label="LDK",
        money_room="100000",
        house_area="40",
    )
    base.update(kw)
    return base


class ScientificCorrectness(unittest.TestCase):
    def test_duration_zero_leap_and_invalid(self):
        d = prepare(pd.DataFrame([
            row(pub_start_date="2020-02-28", pub_end_date="2020-03-01"),
            row(pub_end_date="2020-03-01"),
            row(pub_end_date="2020-02-28"),
            row(pub_end_date="bad"),
        ]))
        self.assertEqual(d.duration.iloc[0], 2)
        self.assertEqual(d.duration.iloc[1], 0)
        self.assertEqual(d.valid_dates.tolist(), [True, True, False, False])

    def test_age_is_at_listing_month(self):
        d = prepare(pd.DataFrame([
            row(),
            row(kenchiku_date="201912"),
            row(kenchiku_date="202004"),
            row(kenchiku_date="201900"),
            row(kenchiku_date="202003"),
        ]))
        np.testing.assert_allclose(d.age.iloc[:2], [1.0, 0.25])
        self.assertTrue(pd.isna(d.age.iloc[2]))
        self.assertTrue(d.age_build_after_listing.iloc[2])
        self.assertTrue(pd.isna(d.age.iloc[3]))
        self.assertEqual(d.age.iloc[4], 0)

    def test_layout_preserves_room_count_and_accepts_lk(self):
        d = prepare(pd.DataFrame([
            row(),
            row(madori_number_all="1", madori_kind_all_label="LK"),
            row(madori_number_all="２", madori_kind_all_label="ＤＫ"),
        ]))
        self.assertEqual(d.layout.tolist(), ["2LDK", "1LK", "2DK"])

    def test_invalid_layout_is_audited_not_guessed(self):
        d = prepare(pd.DataFrame([
            row(madori_number_all=""),
            row(madori_kind_all_label="UNKNOWN"),
        ]))
        self.assertTrue(d.layout.isna().all())
        self.assertTrue(d.layout_invalid.all())

    def test_numeric_nonpositive_becomes_missing(self):
        d = prepare(pd.DataFrame([
            row(money_room="0", house_area="-1"),
            row(money_room="unknown", house_area="40"),
        ]))
        self.assertTrue(pd.isna(d.rent.iloc[0]))
        self.assertTrue(pd.isna(d.rent.iloc[1]))
        self.assertTrue(pd.isna(d.area.iloc[0]))
        self.assertEqual(d.area.iloc[1], 40)

    def test_exact_stats_does_not_hide_mode_ties(self):
        s = exact_stats(pd.Series([1, 1, 2, 2, 3]))
        self.assertEqual(s["median"], 2)
        self.assertEqual(s["mode_values"], "1|2")
        self.assertEqual(s["mode_tie_count"], 2)
        self.assertEqual(s["mode_frequency"], 2)

    def test_selected_mode_is_missing_when_tied(self):
        value, issue = selected_stat_value(pd.Series([1, 1, 2, 2]), "mode")
        self.assertTrue(pd.isna(value))
        self.assertTrue(issue.startswith("mode_tie:"))

    def test_scope_all_types_means_no_type_filter(self):
        d = prepare(pd.DataFrame([
            row(id="a", bukken_type="3101"),
            row(id="b", bukken_type="9999"),
        ]))
        primary, flow = apply_scope(d, years=[2020], types=None, duplicates="keep")
        self.assertEqual(len(primary), 2)
        self.assertFalse(flow["step"].str.contains("property-type").any())

    def test_scope_type_filter_only_when_explicit(self):
        d = prepare(pd.DataFrame([
            row(id="a", bukken_type="3101"),
            row(id="b", bukken_type="9999"),
        ]))
        primary, _ = apply_scope(d, years=[2020], types=["3101"], duplicates="keep")
        self.assertEqual(primary.id.tolist(), ["a"])

    def test_duplicate_policy_is_explicit(self):
        d = prepare(pd.DataFrame([
            row(id="a"), row(id="a", pub_end_date="2020-03-20"), row(id="b")
        ]))
        kept, _ = apply_scope(d, [2020], None, "keep")
        excluded, _ = apply_scope(d, [2020], None, "exclude")
        self.assertEqual(len(kept), 3)
        self.assertEqual(excluded.id.tolist(), ["b"])

    def test_group_stats_are_from_raw_records(self):
        d = prepare(pd.DataFrame([
            row(id="a", money_room="50000", pub_end_date="2020-03-11"),
            row(id="b", money_room="50000", pub_end_date="2020-03-21"),
            row(id="c", money_room="60000", pub_end_date="2020-03-31"),
        ]))
        d = apply_equal_width_bands(d, normalize_band_config())
        tab = group_statistics(d, "rent_band", "median").set_index("rent_band")
        self.assertEqual(tab.loc["40-60k", "median"], 15)
        self.assertEqual(tab.loc["60-80k", "median"], 30)

    def test_equal_width_band_boundaries_without_overflow(self):
        cfg = normalize_band_config({
            "rent": {"step": 20000, "bins_per_figure": 10},
            "area": {"step": 10, "bins_per_figure": 10},
            "age": {"step": 5, "bins_per_figure": 10},
            "duration": {"step": 30, "bins_per_figure": 12},
        })
        d = prepare(pd.DataFrame([
            row(id="a", money_room="59999", house_area="29.9", kenchiku_date="201503"),
            row(id="b", money_room="60000", house_area="30", kenchiku_date="201003"),
            row(id="c", money_room="250000", house_area="120", kenchiku_date="194003"),
        ]))
        schema = build_band_schema(d, cfg)
        d = apply_band_schema(d, schema)
        self.assertEqual(str(d.rent_band.iloc[0]), "40-60k")
        self.assertEqual(str(d.rent_band.iloc[1]), "60-80k")
        self.assertEqual(str(d.rent_band.iloc[2]), "240-260k")
        self.assertEqual(str(d.area_band.iloc[0]), "20-30m2")
        self.assertEqual(str(d.area_band.iloc[1]), "30-40m2")
        self.assertEqual(str(d.area_band.iloc[2]), "120-130m2")
        self.assertEqual(str(d.age_band.iloc[0]), "5-10y")
        self.assertEqual(str(d.age_band.iloc[1]), "10-15y")
        self.assertTrue(all(not x.startswith(">=") for x in schema["rent"]["labels"]))
        self.assertTrue(all(not x.startswith(">=") for x in schema["area"]["labels"]))

    def test_schema_covers_exact_edge_max_with_next_bin(self):
        d = prepare(pd.DataFrame([
            row(id="a", money_room="200000", house_area="100"),
            row(id="b", money_room="100000", house_area="50"),
        ]))
        schema = build_band_schema(d, normalize_band_config())
        d2 = apply_band_schema(d, schema)
        self.assertEqual(str(d2.rent_band.iloc[0]), "200-220k")
        self.assertEqual(str(d2.area_band.iloc[0]), "100-110m2")

    def test_equal_width_config_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            normalize_band_config({"age": {"step": 0}})
        with self.assertRaises(ValueError):
            normalize_band_config({"age": {"bins_per_figure": 0}})

    def test_parse_scope_arguments(self):
        self.assertIsNone(parse_years("all"))
        self.assertEqual(parse_years("2020,2021"), [2020, 2021])
        self.assertIsNone(parse_types("all"))
        self.assertEqual(parse_types("3101,3102"), ["3101", "3102"])


if __name__ == "__main__":
    unittest.main()

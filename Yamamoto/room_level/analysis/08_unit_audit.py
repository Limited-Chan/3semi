"""住戸IDの利用可能性と、連続掲載の統合件数を出力(修正版)。

実行: ..\\.venv\\Scripts\\python.exe -m analysis.08_unit_audit
(Yamamoto/room_level ディレクトリから)

統合しなかった理由(間取りの矛盾/面積の矛盾)を、掲載件数つきで出力する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # リポジトリの直下(data/ がある場所)
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
from src.data import load_prepared
from room_lib.listing_episodes import build_episodes, prepare_room_keys

INPUT = ROOT / "data" / "prepared" / "kokubunji_raw.parquet"
OUTPUT = HERE / "results" / "08_unit_audit"
GAPS = [0, 1, 7, 14]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = load_prepared(INPUT)
    keyed = prepare_room_keys(raw)
    per_unit = keyed.loc[keyed["has_unit_key"]].groupby("unit_key").size()
    summary = []
    for gap in GAPS:
        episodes, seg = build_episodes(raw, max_gap_days=gap)
        scope = episodes[episodes["start_year"].isin([2020, 2021, 2022, 2023])]
        multi = seg[seg["source_rows"] > 1]
        blocked = multi[multi["metadata_conflict"]]
        summary.append({
            "max_gap_days": gap,
            "original_rows_all_years": len(raw),
            "unit_id_coverage": float(keyed["has_unit_key"].mean()),
            "units_with_repeated_listings": int((per_unit > 1).sum()),
            "overlapping_or_adjacent_segments": len(multi),
            "segments_merged": int(multi["merged"].sum()),
            "segments_blocked_by_conflict": len(blocked),
            "listings_in_blocked_segments": int(blocked["source_rows"].sum()),
            "blocked_by_layout": int(blocked["layout_conflict"].sum()),
            "blocked_by_area_only": int((blocked["area_conflict"] & ~blocked["layout_conflict"]).sum()),
            "merged_segments_with_station_difference": int(
                (multi["merged"] & multi["station_differs"]).sum()
            ),
            "episodes_all_years": len(episodes),
            "episodes_starting_2020_2023": len(scope),
            "merged_original_rows": int((episodes["source_rows"] - 1).sum()),
        })
        episodes.to_parquet(OUTPUT / f"episodes_gap{gap}.parquet", index=False)
        if gap == 1:
            seg.to_csv(OUTPUT / "segment_audit.csv", index=False, encoding="utf-8-sig")
            episodes.to_csv(OUTPUT / "episodes_gap1.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUT / "episode_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(table.T.to_string(header=False))
    print("\n注意: IDが同じことだけで実際の同一住戸と確定したわけではありません。")
    print("間取り・面積が食い違うセグメントと、ID欠損の掲載は統合していません。")


if __name__ == "__main__":
    main()

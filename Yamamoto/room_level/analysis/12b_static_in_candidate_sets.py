"""短い/長いで学習した属性だけの分類器(方式2)を、09と同じ「同時に掲載中の候補」の中で評価する(探索的)。
候補集合の定義は09bと同じ5通り。方式1(Model 2)・経過日数だけの方法と並べる。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
m = importlib.import_module("analysis.09_candidate_ranking")
s9b = importlib.import_module("analysis.09b_segment_sensitivity")
cmp = importlib.import_module("analysis.12_high_low_compare")


def main() -> None:
    episodes = m.add_bins(pd.read_parquet(m.AUDIT_DIR / "episodes_gap1.parquet"))
    train = m.make_survival_table(episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])].copy(), m.TRAIN_CUTOFF)
    static = cmp.fit_static(train)  # 2023-01-01までに分かる情報だけで、短い/長いを学習
    candidates = s9b.build_all_candidates()
    candidates["duration_static"] = static.predict_proba(cmp.features(candidates))[:, 1]
    methods = ["overall_km", "model2", "duration_static"]
    keep = candidates[methods + m.METHODS].apply(lambda s: np.isfinite(s.to_numpy(dtype=float))).all(axis=1)
    candidates = candidates.loc[keep].copy()
    m.METHODS = list(dict.fromkeys(m.METHODS + ["duration_static"]))
    rows = []
    for name, keys in s9b.SCENARIOS.items():
        d = candidates.copy()
        d["segment"] = d[keys].astype(str).agg(" | ".join, axis=1) if keys else "all"
        metrics, _ = m.method_metrics(d)
        mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
        row = {"候補の似かた": name, "比較した対の数": int(mixed.loc[mixed["method"] == "model2", "positive_negative_pairs"].sum())}
        for method in methods:
            g = mixed.loc[mixed["method"] == method]
            row[method] = float(np.average(g["within_group_auc"], weights=g["positive_negative_pairs"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    out = HERE / "results" / "12_high_low_compare"
    table.to_csv(out / "in_candidate_sets.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

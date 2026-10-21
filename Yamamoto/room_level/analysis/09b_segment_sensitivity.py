"""候補集合の「似かた」を変えて、属性が順位付けに足す情報を測る(探索的な感度分析)。

09は 間取り×築年数10年帯×家賃2万円帯 がすべて同じ物件どうしだけを比べる最も厳しい条件。
ここでは候補集合の定義を事前に5通り決め、結果の良し悪しに関係なくすべて報告する。
  A 間取り×築年数×家賃(09と同じ)  B 間取り×築年数  C 間取り×家賃  D 間取りだけ  E 同じ日に掲載中の全物件

実行例: python -m analysis.09b_segment_sensitivity
(先に 08_unit_audit を実行しておく)。大きい候補集合は区間(ブートストラップ)を出していない。
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

SCENARIOS = {
    "A 間取り×築年数×家賃(最も厳しい)": ["layout_group", "age_group", "rent_group"],
    "B 間取り×築年数(家賃は自由)": ["layout_group", "age_group"],
    "C 間取り×家賃(築年数は自由)": ["layout_group", "rent_group"],
    "D 間取りだけ": ["layout_group"],
    "E 同じ日に掲載中の全物件": [],
}
SHOW = ["overall_km", "age_layout_km", "model1", "model2", "knn30_median", "km_remaining_median"]


def build_all_candidates(gap: int = 1) -> pd.DataFrame:
    episodes = m.add_bins(pd.read_parquet(m.AUDIT_DIR / f"episodes_gap{gap}.parquet"))
    train = episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])].copy()
    train = m.make_survival_table(train, m.TRAIN_CUTOFF)
    model1, _ = m.fit_model(train, use_age_layout_combo=False)
    model2, _ = m.fit_model(train, use_age_layout_combo=True)
    overall_km = m.fit_km(train["survival_days"], train["event"])
    global_median = m.km_median(train["survival_days"], train["event"], m.REMAINING_CAP_DAYS)
    subgroup_km, group_medians = {}, {}
    for key, sub in train.groupby(["age_group", "layout_group"], dropna=False):
        k = (str(key[0]), str(key[1]))
        subgroup_km[k] = m.fit_km(sub["survival_days"], sub["event"])
        group_medians[k] = m.km_median(sub["survival_days"], sub["event"], global_median)
    knn = m.SimilarListingMedian(train, neighbors=30)
    frames = []
    for snapshot in m.SNAPSHOTS:
        c = m.create_candidates(episodes, snapshot)
        if c.empty:
            continue
        c = m.add_baselines(c, overall_km, subgroup_km, group_medians, global_median)
        c["knn30_median"] = -knn.predict_median(c)
        c["model1"] = m.predict_variable_elapsed(model1, c, False)
        c["model2"] = m.predict_variable_elapsed(model2, c, True)
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    candidates = build_all_candidates()
    finite = candidates[m.METHODS].apply(lambda s: np.isfinite(s.to_numpy(dtype=float))).all(axis=1)
    candidates = candidates.loc[finite].copy()
    rows = []
    for name, keys in SCENARIOS.items():
        d = candidates.copy()
        d["segment"] = d[keys].astype(str).agg(" | ".join, axis=1) if keys else "all"
        metrics, _ = m.method_metrics(d)
        mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
        sizes = metrics.drop_duplicates(["snapshot", "segment"])["n_candidates"]
        row = {"候補集合の定義": name, "候補集合数": int(sizes.size), "集合の大きさ(中央値)": float(sizes.median()),
               "比較した対の数": int(mixed.loc[mixed["method"] == "model2", "positive_negative_pairs"].sum())}
        for method in SHOW:
            g = mixed.loc[mixed["method"] == method]
            row[method] = float(np.average(g["within_group_auc"], weights=g["positive_negative_pairs"]))
        row["model2−overall_km"] = row["model2"] - row["overall_km"]
        rows.append(row)
    table = pd.DataFrame(rows)
    out = HERE / "results" / "09_candidate_ranking"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "segment_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    print("\n注意: 事前に決めた5通りをすべて報告。大きい集合は同じ物件が複数日に出るため区間は未計算。")


if __name__ == "__main__":
    main()

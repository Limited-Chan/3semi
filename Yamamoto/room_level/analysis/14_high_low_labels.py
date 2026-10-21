"""部屋単位・7/14日リスク(Model 2)から実際に「高・中・低」を出し、どのくらい当たるかを測る(探索的)。

手順: 2020〜2021年だけで学習 → 2022年の候補で境界(3分位)を決める → 2020〜2022年で再学習 →
      境界を固定して2023年の候補に適用 → 実際の終了率を測る。
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
r13 = importlib.import_module("analysis.13_room_level_priority")

HORIZONS = [7, 14]
OUT = HERE / "results" / "14_high_low_labels"


def candidates_for_year(episodes: pd.DataFrame, year: int) -> pd.DataFrame:
    frames = []
    for snap in m.SNAPSHOTS:
        snapshot = snap.replace(year=year)
        active = episodes.loc[
            (episodes["start_year"] == year) & (episodes["start_date"] <= snapshot) & (episodes["end_date"] > snapshot)
        ].copy()
        if active.empty:
            continue
        active["elapsed_day"] = (snapshot - active["start_date"]).dt.days.astype(int)
        for h in HORIZONS:
            active[f"target_{h}"] = (
                (active["end_date"] <= snapshot + pd.Timedelta(days=h)) & (active["end_date"] < m.OBSERVATION_END)
            ).astype(int)
        active["snapshot"] = snapshot.date().isoformat()
        frames.append(m.add_bins(active))
    return pd.concat(frames, ignore_index=True)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cw, q * cw[-1], side="left")), len(v) - 1)])


def label(score: np.ndarray, q1: float, q2: float) -> np.ndarray:
    return np.select([score < q1, score < q2], ["低", "中"], default="高")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = m.add_bins(pd.read_parquet(m.AUDIT_DIR / "episodes_gap1.parquet"))
    cut22, cut23 = pd.Timestamp("2022-01-01"), m.TRAIN_CUTOFF
    train_val = m.make_survival_table(episodes.loc[episodes["start_year"].isin([2020, 2021])].copy(), cut22)
    train_test = m.make_survival_table(episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])].copy(), cut23)
    model_val = r13.Hazard(True, []).fit(train_val)
    model_test = r13.Hazard(True, []).fit(train_test)
    val = candidates_for_year(episodes, 2022)
    test = candidates_for_year(episodes, 2023)
    print(f"境界を決める2022年の候補: {len(val):,}行, 評価する2023年の候補: {len(test):,}行")
    rows, within = [], []
    for h in HORIZONS:
        val[f"risk_{h}"] = model_val.risk(val, h)
        test[f"risk_{h}"] = model_test.risk(test, h)
        w = (1 / val.groupby("id")["id"].transform("size")).to_numpy(float)
        q1 = weighted_quantile(val[f"risk_{h}"].to_numpy(float), w, 1 / 3)
        q2 = weighted_quantile(val[f"risk_{h}"].to_numpy(float), w, 2 / 3)
        print(f"\n[{h}日] 2022年で決めた境界: 低 < {q1:.1%} <= 中 < {q2:.1%} <= 高")
        for year, d in [(2022, val), (2023, test)]:
            d = d.copy()
            d["label"] = label(d[f"risk_{h}"].to_numpy(float), q1, q2)
            base = d[f"target_{h}"].mean()
            for lab in ["低", "中", "高"]:
                s = d[d["label"] == lab]
                rows.append({"予測期間(日)": h, "年": year, "ラベル": lab, "割合": len(s) / len(d),
                             "実際の終了率": s[f"target_{h}"].mean(), "全体の終了率": base,
                             "全体比(倍)": s[f"target_{h}"].mean() / base})
            if year == 2023:
                hi, lo = d[d["label"] == "高"], d[d["label"] == "低"]
                events = d[f"target_{h}"].sum()
                rows.append({"予測期間(日)": h, "年": year, "ラベル": "高が捕まえた終了の割合", "割合": hi[f"target_{h}"].sum() / events,
                             "実際の終了率": np.nan, "全体の終了率": np.nan, "全体比(倍)": np.nan})
                # 似た候補の中で、高と低に分かれた物件の実際の終了率の差
                for name, keys in s9b.SCENARIOS.items():
                    if name[0] not in "ADE":
                        continue
                    seg = d[keys].astype(str).agg(" | ".join, axis=1) if keys else pd.Series("all", index=d.index)
                    diffs, wts = [], []
                    for _, g in d.assign(seg=seg).groupby(["snapshot", "seg"]):
                        a, b = g[g["label"] == "高"], g[g["label"] == "低"]
                        if len(a) and len(b) and len(g) >= 3:
                            diffs.append(a[f"target_{h}"].mean() - b[f"target_{h}"].mean())
                            wts.append(min(len(a), len(b)))
                    within.append({"予測期間(日)": h, "候補の似かた": name,
                                   "高と低の両方がある候補集合": len(diffs),
                                   "高−低の実終了率の差(pt)": 100 * float(np.average(diffs, weights=wts)) if diffs else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "labels_by_year.csv", index=False, encoding="utf-8-sig")
    wtable = pd.DataFrame(within)
    wtable.to_csv(OUT / "within_set_high_low_gap.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 高・中・低ごとの実際の終了率 ===")
    print(table.round(3).to_string(index=False))
    print("\n=== 同じ候補集合の中で、高と判定された物件と低と判定された物件の差(2023年) ===")
    print(wtable.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

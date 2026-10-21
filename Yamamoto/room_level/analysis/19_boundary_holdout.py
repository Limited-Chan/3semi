"""高・低の境界を2022年で決め、2023年に固定して確かめる(未使用年での確認)。

手順:
  1. 2021年開始だけで学習した Model 2 で、2022年の候補(5つの日付)を採点する
  2. 2022年の結果から境界を決める
       案a: 目的から決めた絶対の境界(7日: 15%、14日: 25%)
       案b: 2022年の上位30%に入る点数(=高が約3割になる境界)
  3. その境界を動かさず、2021〜2022年開始で学習した Model 2 の2023年の候補(17番の保存分)に当てる
見るもの: 高の割合、高/低の実終了率、同じ候補の中での差(間取りだけ同じ / 最も厳しい)
注意: 2023年は境界を選ぶのに使っていない。ただし何度も見ているデータなので「探索的」。
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
s17 = importlib.import_module("analysis.17_fixed_window_features")
from src.data import prepare
from room_lib.listing_episodes import build_episodes_fast

OUT = HERE / "results" / "19_boundary_holdout"
CAND_2023 = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
HORIZONS = [7, 14]
OBJECTIVE_T = {7: 0.15, 14: 0.25}
SNAPSHOTS_2022 = pd.to_datetime(["2022-04-01", "2022-06-01", "2022-08-01", "2022-10-01", "2022-12-01"])
HIGH_SHARE = 0.30
SETS = {"D 間取りだけ同じ": ["city", "layout_group"],
        "A 間取り×築年数×家賃": ["city", "layout_group", "age_group", "rent_group"]}
rng = np.random.default_rng(20261016)


def score_2022() -> pd.DataFrame:
    raw = prepare(s17.s15.extract_tama())
    raw["city"] = raw["addr1_2_name"]
    ep = build_episodes_fast(raw, max_gap_days=1)
    ep["city"] = ep["addr1_2_name"]
    ep = m.add_bins(ep)
    cutoff = pd.Timestamp("2022-01-01")
    train = m.make_survival_table(ep.loc[ep["start_year"] == 2021].copy(), cutoff)
    print(f"学習(2021年開始のみ): {len(train):,}件")
    model = s17.Hazard("lr", []).fit(train)
    frames = []
    ev = ep.loc[ep["city"].isin(s17.NEAR)]
    for snap in SNAPSHOTS_2022:
        c = ev.loc[(ev["start_year"] == 2022) & (ev["start_date"] <= snap) & (ev["end_date"] > snap)].copy()
        c["elapsed_day"] = (snap - c["start_date"]).dt.days.astype(int)
        for h in HORIZONS:
            c[f"target_{h}"] = ((c["end_date"] <= snap + pd.Timedelta(days=h)) & (c["end_date"] < m.OBSERVATION_END)).astype(int)
            c[f"base_{h}"] = model.risk(c, h)
        c["snapshot"] = snap.date().isoformat()
        frames.append(m.add_bins(c))
        print(f"{snap.date()}: 候補 {len(c):,}件")
    return pd.concat(frames, ignore_index=True)


def label_table(d: pd.DataFrame, h: int, t: float) -> dict:
    p, y = d[f"base_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
    hi = p >= t
    return {"高の割合": hi.mean(), "高の実終了率": y[hi].mean(), "低の実終了率": y[~hi].mean(), "全体": y.mean(),
            "差pt": 100 * (y[hi].mean() - y[~hi].mean()), "高が捕まえる終了": y[hi].sum() / y.sum(),
            "平均の予測": p.mean()}


def within_set(d: pd.DataFrame, h: int, t: float, keys: list[str]) -> tuple[float, float, float, int]:
    seg = d[["snapshot", *keys]].astype(str).agg("|".join, axis=1)
    diffs, w, kind = [], [], []
    for s, g in d.assign(seg=seg).groupby("seg"):
        if len(g) < 3:
            continue
        hi = g[f"base_{h}"].to_numpy() >= t
        if hi.all() or (~hi).all():
            continue
        y = g[f"target_{h}"].to_numpy()
        diffs.append(y[hi].mean() - y[~hi].mean())
        w.append(len(g))
        kind.append(s.split("|", 1)[1])
    diffs, w, kind = np.array(diffs), np.array(w, float), np.array(kind)
    uk = np.unique(kind)
    idx = {k: np.flatnonzero(kind == k) for k in uk}
    est = 100 * np.average(diffs, weights=w)
    bs = [100 * np.average(diffs[p], weights=w[p]) for p in
          (np.concatenate([idx[uk[i]] for i in rng.integers(0, len(uk), len(uk))]) for _ in range(500))]
    lo, hi_ = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi_, len(diffs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c22 = score_2022()
    c23 = pd.read_parquet(CAND_2023)
    rows, wrows = [], []
    for h in HORIZONS:
        bound = {"a 目的から決めた境界": OBJECTIVE_T[h],
                 "b 2022年の上位30%の点数": float(np.quantile(c22[f"base_{h}"], 1 - HIGH_SHARE))}
        for name, t in bound.items():
            for year, d in [("2022(境界を決めた年)", c22), ("2023(固定して適用)", c23)]:
                rows.append({"予測期間(日)": h, "境界の決め方": name, "境界": t, "年": year, **label_table(d, h, t)})
            for sname, keys in SETS.items():
                est, lo, hi_, n = within_set(c23, h, t, keys)
                wrows.append({"予測期間(日)": h, "境界の決め方": name, "境界": t, "候補の似かた": sname,
                              "高低が分かれた集合": n, "高−低の実終了率の差pt": est, "ci_low": lo, "ci_high": hi_})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(wrows)
    t1.to_csv(OUT / "labels_by_year.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT / "within_set_2023.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== 高・低の割合と実終了率(2022で決め、2023に固定) ===")
    print(t1.round(3).to_string(index=False))
    print("\n=== 2023年の、同じ候補の中での高−低の差 ===")
    print(t2.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

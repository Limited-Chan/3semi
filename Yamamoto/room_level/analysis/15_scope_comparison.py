"""学習に使う地域の範囲を変えて、国分寺市の2023年の候補で比べる(探索的)。

範囲: S1 国分寺市だけ / S2 近隣10市+国分寺市 / S3 多摩地域全体
外部検証: S2x S3x = 国分寺市を学習から外す(他の市だけで学習して国分寺市で確かめる)
評価: 国分寺市2023年のスナップショットの「同時に掲載中の候補」の中での順位付け(09bの5通り)
手法: 部屋単位エピソード(統合ルール1日)、Model 2相当の離散時間ハザード(地域の変数つき)。
      大きい範囲では、掲載日数を179日までに切り、終了しなかった行を間引いて重みを付けて学習する。
      比較を公平にするため、S1も同じ方法で学習し直す。
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # リポジトリの直下(data/ がある場所)
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
m = importlib.import_module("analysis.09_candidate_ranking")
s9b = importlib.import_module("analysis.09b_segment_sensitivity")
from src.data import SOURCE_COLUMNS, prepare
from src.discrete_model import BASE_CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_feature_frame
from room_lib.listing_episodes import build_episodes_fast

TSV = Path("/Users/aki/Downloads/bukken_rent_v1.0.1.tsv")
TAMA_RAW = ROOT / "data" / "prepared" / "tama_raw.parquet"
OUT = HERE / "results" / "15_scope_comparison"
CAP_DAY = 179
TARGET_NEGATIVE_ROWS = 2_500_000
HORIZONS = [7, 14, 30]
TARGET_CITY = "国分寺市"
NEAR = ["国分寺市", "国立市", "小金井市", "小平市", "立川市", "府中市", "三鷹市", "武蔵野市", "西東京市", "東村山市", "調布市"]
METHODS = ["S1", "S2", "S3", "S2x", "S3x"]
SCENARIOS = s9b.SCENARIOS
rng = np.random.default_rng(20261013)


def extract_tama() -> pd.DataFrame:
    if TAMA_RAW.exists():
        return pd.read_parquet(TAMA_RAW)
    t0 = time.time()
    chunks = []
    for ch in pd.read_csv(TSV, sep="\t", usecols=SOURCE_COLUMNS, dtype=str, chunksize=300_000,
                          encoding="utf-8", keep_default_na=False):
        f = ch[(ch["addr1_1_name"] == "東京都") & (~ch["addr1_2_name"].str.endswith("区"))
               & ch["pub_start_date"].str[:4].isin(["2020", "2021", "2022", "2023"])]
        if len(f):
            chunks.append(f)
    d = pd.concat(chunks, ignore_index=True)
    d.to_parquet(TAMA_RAW, index=False)
    print(f"多摩地域を抽出: {len(d):,}件 ({time.time() - t0:.0f}秒)")
    return d


def person_period_sample(train: pd.DataFrame, multi_city: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = ["age", "rent", "area", "walk_distance", "layout", "station", "start_month"] + (["city"] if multi_city else [])
    days = train["survival_days"].astype(int).to_numpy()
    observed = np.minimum(days, CAP_DAY)
    event = (train["event"].to_numpy(int) == 1) & (days <= CAP_DAY)
    rows_per = observed + 1
    total_neg = int(rows_per.sum() - event.sum())
    p_neg = min(1.0, TARGET_NEGATIVE_ROWS / max(total_neg, 1))
    print(f"  学習エピソード {len(train):,}件, 全行 {int(rows_per.sum()):,}, 終了行 {int(event.sum()):,}, 非終了行の保持率 {p_neg:.1%}")
    frames, ys, ws = [], [], []
    for start in range(0, len(train), 20000):
        sl = slice(start, start + 20000)
        n = rows_per[sl]
        idx = np.repeat(np.arange(len(n)), n)
        t = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        y = np.zeros(len(idx), dtype=int)
        y[np.cumsum(n) - 1] = event[sl].astype(int)
        keep = (y == 1) | (rng.random(len(y)) < p_neg)
        base = train.iloc[sl][cols].reset_index(drop=True)
        part = base.iloc[idx[keep]].reset_index(drop=True)
        part["time_day"] = t[keep]
        frames.append(part)
        ys.append(y[keep])
        ws.append(np.where(y[keep] == 1, 1.0, 1.0 / p_neg))
    return pd.concat(frames, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


class ScopeHazard:
    def __init__(self, multi_city: bool):
        self.multi_city = multi_city

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = prepare_feature_frame(df, use_age_layout_combo=True)
        if self.multi_city:
            x["city"] = df["city"].astype("string").fillna("欠損").astype(str).to_numpy()
        return x

    def fit(self, train: pd.DataFrame) -> "ScopeHazard":
        rows, y, w = person_period_sample(train, self.multi_city)
        cat = BASE_CATEGORICAL_FEATURES + ["age_layout_combo"] + (["city"] if self.multi_city else [])
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        self.pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=500, solver="lbfgs", C=1.0))])
        t0 = time.time()
        self.pipe.fit(self._x(rows), y, lr__sample_weight=w)
        print(f"  学習時間 {time.time() - t0:.0f}秒")
        return self

    def risk(self, cand: pd.DataFrame, horizon: int) -> np.ndarray:
        n = len(cand)
        pos = np.repeat(np.arange(n), horizon)
        cols = ["age", "rent", "area", "walk_distance", "layout", "station", "start_month"] + (["city"] if self.multi_city else [])
        fut = cand.iloc[pos][cols].reset_index(drop=True).copy()
        day = np.repeat(cand["elapsed_day"].to_numpy(dtype=int), horizon) + np.tile(np.arange(1, horizon + 1), n)
        fut["time_day"] = np.minimum(day, CAP_DAY)
        daily = self.pipe.predict_proba(self._x(fut))[:, 1].reshape(n, horizon)
        return 1 - np.exp(np.log1p(-np.clip(daily, 0, 1 - 1e-12)).sum(axis=1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = prepare(extract_tama())
    raw["city"] = raw["addr1_2_name"]
    print(f"多摩地域の掲載 {len(raw):,}件 を読み込み ({time.time() - t0:.0f}秒)")
    episodes = build_episodes_fast(raw, max_gap_days=1)
    episodes["city"] = episodes["addr1_2_name"]
    episodes = m.add_bins(episodes)
    print(f"部屋エピソード {len(episodes):,}件 (掲載から {1 - len(episodes) / len(raw):.1%} 減)")
    print("市ごとの2020〜2022年開始エピソード数(上位):", episodes[episodes.start_year.isin([2020, 2021, 2022])].city.value_counts().head(5).to_dict())

    pool = episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])]
    scopes = {
        "S1": (pool["city"] == TARGET_CITY, False),
        "S2": (pool["city"].isin(NEAR), True),
        "S3": (pd.Series(True, index=pool.index), True),
        "S2x": (pool["city"].isin(NEAR) & (pool["city"] != TARGET_CITY), True),
        "S3x": (pool["city"] != TARGET_CITY, True),
    }
    models = {}
    for name, (mask, multi) in scopes.items():
        print(f"\n[{name}] 学習")
        train = m.make_survival_table(pool.loc[mask].copy(), m.TRAIN_CUTOFF)
        models[name] = (ScopeHazard(multi).fit(train), multi)

    kok = episodes.loc[episodes["city"] == TARGET_CITY]
    frames = []
    for snapshot in m.SNAPSHOTS:
        c = kok.loc[(kok["start_year"] == 2023) & (kok["start_date"] <= snapshot) & (kok["end_date"] > snapshot)].copy()
        c["elapsed_day"] = (snapshot - c["start_date"]).dt.days.astype(int)
        for h in HORIZONS:
            c[f"target_{h}"] = ((c["end_date"] <= snapshot + pd.Timedelta(days=h)) & (c["end_date"] < m.OBSERVATION_END)).astype(int)
        c["snapshot"] = snapshot.date().isoformat()
        c = m.add_bins(c)
        for name, (model, _) in models.items():
            for h in HORIZONS:
                c[f"{name}_{h}"] = model.risk(c, h)
        frames.append(c)
        print(f"{snapshot.date()}: 国分寺市の候補 {len(c):,}件")
    cand = pd.concat(frames, ignore_index=True)
    cand.drop(columns=[c for c in cand.columns if c.startswith("_")], errors="ignore").to_csv(
        OUT / "kokubunji_2023_candidates.csv", index=False, encoding="utf-8-sig")

    rows, diff_rows = [], []
    m.METHODS = METHODS
    m.REFERENCE = "S1"
    for h in HORIZONS:
        d = cand.copy()
        d["target"] = d[f"target_{h}"]
        for name in METHODS:
            d[name] = d[f"{name}_{h}"]
        for scen, keys in SCENARIOS.items():
            dd = d.copy()
            dd["segment"] = dd[keys].astype(str).agg(" | ".join, axis=1) if keys else "all"
            metrics, _ = m.method_metrics(dd)
            mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
            row = {"予測期間(日)": h, "候補の似かた": scen, "対の数": int(mixed.loc[mixed["method"] == "S1", "positive_negative_pairs"].sum())}
            for name in METHODS:
                g = mixed.loc[mixed["method"] == name]
                row[name] = float(np.average(g["within_group_auc"], weights=g["positive_negative_pairs"]))
            rows.append(row)
            if scen.startswith("A"):
                _, diffs = m.summarize_with_intervals(metrics)
                for _, r in diffs.iterrows():
                    diff_rows.append({"予測期間(日)": h, "範囲": r["method"], "S1との差": r["auc_difference"],
                                      "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "scope_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diff_rows).to_csv(OUT / "scope_A_differences.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 国分寺市2023年の候補内AUC(範囲別) ===")
    print(table.round(3).to_string(index=False))
    print("\n=== 最も厳しい集合(A)での、S1(国分寺市だけ)との差(グループ単位ブートストラップ95%区間) ===")
    print(pd.DataFrame(diff_rows).round(3).to_string(index=False))
    print(f"\n総時間 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()

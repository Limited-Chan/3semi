"""部屋単位の掲載終了リスクを、予測期間(7/14/30日)と属性の足し方を変えて、
似た候補の中で比べる(探索的)。

- 目的変数: 08で統合した「部屋のエピソード」が、スナップショットから h 日以内に終わるか
- 方式: 経過日数だけ(overall_km) / 築年数×間取りのKM / Model 1 / Model 2 /
        Model 2 + 現況・入居可能時期(model2x)
- 評価: 候補集合の定義5通り(09b)ごとの、候補内AUCと、上位半分−下位半分の実終了率の差
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
m = importlib.import_module("analysis.09_candidate_ranking")
s9b = importlib.import_module("analysis.09b_segment_sensitivity")
from src.discrete_model import BASE_CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_feature_frame
from src.person_period import DEFAULT_FEATURE_COLUMNS, make_person_period

TSV = Path("/Users/aki/Downloads/bukken_rent_v1.0.1.tsv")
OUT = HERE / "results" / "13_room_level_priority"
HORIZONS = [7, 14, 30]
EXTRA = ["genkyo_label", "usable_status_label"]
METHODS = ["overall_km", "age_layout_km", "model1", "model2", "model2x"]
SCENARIOS = s9b.SCENARIOS


def load_extras() -> pd.DataFrame:
    cache = OUT / "extras.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    chunks = []
    for ch in pd.read_csv(TSV, sep="\t", usecols=["id", "addr1_1_name", "addr1_2_name", *EXTRA], dtype=str,
                          chunksize=300_000, encoding="utf-8", keep_default_na=False):
        f = ch[(ch["addr1_1_name"] == "東京都") & (ch["addr1_2_name"] == "国分寺市")]
        if len(f):
            chunks.append(f[["id", *EXTRA]])
    d = pd.concat(chunks, ignore_index=True)
    d["id"] = d["id"].astype("string").str.normalize("NFKC").str.strip()
    d = d.drop_duplicates("id")
    d.to_parquet(cache, index=False)
    return d


class Hazard:
    """既存のModel 1/2と同じ作りの離散時間ハザードモデル(特徴量を足せる)。"""

    def __init__(self, combo: bool, extras: list[str]):
        self.combo, self.extras = combo, extras

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = prepare_feature_frame(df, use_age_layout_combo=self.combo)
        for c in self.extras:
            x[c] = df[c].astype("string").fillna("missing").astype(str).to_numpy()
        return x

    def fit(self, train: pd.DataFrame) -> "Hazard":
        pp = make_person_period(train, feature_columns=DEFAULT_FEATURE_COLUMNS + self.extras)
        cat = BASE_CATEGORICAL_FEATURES + (["age_layout_combo"] if self.combo else []) + self.extras
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        self.pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0))])
        self.pipe.fit(self._x(pp), pp["event_in_day"].astype(int))
        return self

    def risk(self, cand: pd.DataFrame, horizon: int) -> np.ndarray:
        n = len(cand)
        pos = np.repeat(np.arange(n), horizon)
        cols = ["age", "rent", "area", "walk_distance", "layout", "station", "start_month", *self.extras]
        fut = cand.iloc[pos][cols].reset_index(drop=True).copy()
        fut["time_day"] = np.repeat(cand["elapsed_day"].to_numpy(dtype=int), horizon) + np.tile(np.arange(1, horizon + 1), n)
        daily = self.pipe.predict_proba(self._x(fut))[:, 1].reshape(n, horizon)
        return 1 - np.exp(np.log1p(-np.clip(daily, 0, 1 - 1e-12)).sum(axis=1))


def make_candidates(episodes: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    active = episodes.loc[
        (episodes["start_year"] == 2023) & (episodes["start_date"] <= snapshot) & (episodes["end_date"] > snapshot)
    ].copy()
    active["elapsed_day"] = (snapshot - active["start_date"]).dt.days.astype(int)
    for h in HORIZONS:
        if not snapshot + pd.Timedelta(days=h) < m.OBSERVATION_END:
            raise ValueError(f"{snapshot.date()} は観測終了日に近すぎます")
        active[f"target_{h}"] = (
            (active["end_date"] <= snapshot + pd.Timedelta(days=h)) & (active["end_date"] < m.OBSERVATION_END)
        ).astype(int)
    active["snapshot"] = snapshot.date().isoformat()
    return m.add_bins(active)


def half_gap(frame: pd.DataFrame, method: str, target: str) -> float:
    """候補内で点数が上位の半分と下位の半分の、実際の終了率の差(pt)。集合の大きさで重み付け。"""
    gaps, weights = [], []
    for _, g in frame.groupby(["snapshot", "segment"]):
        if len(g) < m.MIN_CANDIDATES:
            continue
        order = np.argsort(-g[method].to_numpy(dtype=float), kind="stable")
        y = g[target].to_numpy()[order]
        k = int(np.ceil(len(y) / 2))
        gaps.append(y[:k].mean() - y[k:].mean())
        weights.append(len(y))
    return 100 * float(np.average(gaps, weights=weights)) if gaps else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = m.add_bins(pd.read_parquet(m.AUDIT_DIR / "episodes_gap1.parquet"))
    episodes["first_source_id"] = episodes["first_source_id"].astype("string").str.normalize("NFKC").str.strip()
    episodes = episodes.merge(load_extras().rename(columns={"id": "first_source_id"}), on="first_source_id", how="left")
    print("現況・入居可能時期の付与率:", f"{episodes[EXTRA].notna().all(axis=1).mean():.1%}")

    train = m.make_survival_table(episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])].copy(), m.TRAIN_CUTOFF)
    print(f"学習: {len(train):,} 部屋エピソード(終了 {int(train['event'].sum()):,})")
    models = {
        "model1": Hazard(False, []).fit(train),
        "model2": Hazard(True, []).fit(train),
        "model2x": Hazard(True, EXTRA).fit(train),
    }
    overall_km = m.fit_km(train["survival_days"], train["event"])
    subgroup = {(str(k[0]), str(k[1])): m.fit_km(g["survival_days"], g["event"])
                for k, g in train.groupby(["age_group", "layout_group"], dropna=False)}

    frames = []
    for snapshot in m.SNAPSHOTS:
        c = make_candidates(episodes, snapshot)
        if c.empty:
            continue
        for h in HORIZONS:
            ov = np.array([m.km_risk(overall_km, int(e), h) for e in c["elapsed_day"]])
            gr = []
            for row, o in zip(c.itertuples(), ov):
                km = subgroup.get((str(row.age_group), str(row.layout_group)))
                r = m.km_risk(km, int(row.elapsed_day), h) if km else np.nan
                gr.append(r if np.isfinite(r) else o)
            c[f"overall_km_{h}"], c[f"age_layout_km_{h}"] = ov, gr
            for name, model in models.items():
                c[f"{name}_{h}"] = model.risk(c, h)
        frames.append(c)
        print(f"{snapshot.date()}: {len(c):,}件")
    cand = pd.concat(frames, ignore_index=True)
    cand.to_csv(OUT / "snapshot_candidates.csv", index=False, encoding="utf-8-sig")

    rows, ci_rows, calib_rows = [], [], []
    for h in HORIZONS:
        d = cand.copy()
        d["target"] = d[f"target_{h}"]
        for method in METHODS:
            d[method] = d[f"{method}_{h}"]
        ok = d[METHODS].apply(lambda s: np.isfinite(s.to_numpy(dtype=float))).all(axis=1)
        d = d.loc[ok].copy()
        m.METHODS = METHODS
        for name, keys in SCENARIOS.items():
            dd = d.copy()
            dd["segment"] = dd[keys].astype(str).agg(" | ".join, axis=1) if keys else "all"
            metrics, _ = m.method_metrics(dd)
            mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
            row = {"予測期間(日)": h, "候補の似かた": name}
            for method in METHODS:
                g = mixed.loc[mixed["method"] == method]
                row[f"AUC_{method}"] = float(np.average(g["within_group_auc"], weights=g["positive_negative_pairs"]))
            for method in ["overall_km", "model2", "model2x"]:
                row[f"上位半分-下位半分pt_{method}"] = half_gap(dd, method, "target")
            rows.append(row)
            if name.startswith("A"):
                summary, diffs = m.summarize_with_intervals(metrics)
                for _, r in diffs.iterrows():
                    if r["method"] in ("model2", "model2x"):
                        ci_rows.append({"予測期間(日)": h, "方式": r["method"], "基準との差": r["auc_difference"],
                                        "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        # 較正: Model 2の予測確率と実際の終了率(全候補を5等分)
        q = pd.qcut(d["model2"], 5, duplicates="drop")
        cal = d.groupby(q, observed=True).agg(予測=("model2", "mean"), 実際=("target", "mean"), 件数=("target", "size"))
        cal["予測期間(日)"] = h
        calib_rows.append(cal.reset_index(drop=True))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "horizon_by_scenario.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(OUT / "scenario_A_differences.csv", index=False, encoding="utf-8-sig")
    calib = pd.concat(calib_rows, ignore_index=True)
    calib.to_csv(OUT / "calibration_model2.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 30)
    show = ["予測期間(日)", "候補の似かた", "AUC_overall_km", "AUC_age_layout_km", "AUC_model1", "AUC_model2", "AUC_model2x"]
    print("\n=== 候補内AUC ===")
    print(table[show].round(3).to_string(index=False))
    print("\n=== 上位半分と下位半分の実終了率の差(pt) ===")
    print(table[["予測期間(日)", "候補の似かた", "上位半分-下位半分pt_overall_km", "上位半分-下位半分pt_model2", "上位半分-下位半分pt_model2x"]].round(1).to_string(index=False))
    print("\n=== 最も厳しい集合(A)で、経過日数だけの方法との差(グループ単位ブートストラップ95%区間) ===")
    print(pd.DataFrame(ci_rows).round(3).to_string(index=False))
    print("\n=== Model 2の較正(予測と実際) ===")
    print(calib.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

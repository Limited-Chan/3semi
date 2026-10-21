"""高/低の出し方を、同じ土台(2022年で境界を決め、2023年に固定して適用)で比べる(探索的)。

方式1: Model 2のリスクを3分位で 低/中/高 (報告書の方式)
方式2: 掲載期間が短い(6日以下)=高、長い(60日以上)=低 で学習した属性だけの分類器の点数を
       同じ3分位で 低/中/高 (掲載期間の長短で分ける方式)
評価はどちらも「その時点まで掲載中の物件が、その後30日以内に実際に終わった割合」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
from src.data import load_prepared
from src.survival import prepare_survival_data

SHORT_DAYS, LONG_DAYS = 6, 60
NUM = ["age", "log_rent", "log_area", "walk"]
CAT = ["layout", "station", "start_month"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["age"] = pd.to_numeric(df["age"], errors="coerce")
    x["log_rent"] = np.log1p(pd.to_numeric(df["rent"], errors="coerce"))
    x["log_area"] = np.log1p(pd.to_numeric(df["area"], errors="coerce"))
    x["walk"] = pd.to_numeric(df["walk_distance"], errors="coerce")
    for c in CAT:
        x[c] = df[c].astype("string").fillna("missing").astype(str)
    return x


def duration_label(df: pd.DataFrame) -> pd.Series:
    """短い(6日以下で終了)=1、長い(60日以上)=0、それ以外=欠損(除く)。"""
    y = pd.Series(np.nan, index=df.index)
    y[(df["event"] == 1) & (df["survival_days"] <= SHORT_DAYS)] = 1
    y[df["survival_days"] >= LONG_DAYS] = 0
    return y


def fit_static(train: pd.DataFrame) -> Pipeline:
    d = train.copy()
    d["y"] = duration_label(d)
    d = d.dropna(subset=["y"])
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ])),
        ("lr", LogisticRegression(max_iter=3000, C=1.0)),
    ])
    pipe.fit(features(d), d["y"].astype(int))
    return pipe


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cw, q * cw[-1], side="left")), len(v) - 1)])


def tiers(score: np.ndarray, q1: float, q2: float) -> np.ndarray:
    return np.select([score < q1, score < q2], ["低", "中"], default="高")


def evaluate(rows: pd.DataFrame, score_col: str, q1: float, q2: float, label: str) -> pd.DataFrame:
    out = []
    for landmark, g in rows.groupby("landmark_day"):
        t = tiers(g[score_col].to_numpy(), q1, q2)
        rate = {k: float(g.loc[t == k, "target"].mean()) for k in ["低", "中", "高"]}
        share = {k: float((t == k).mean()) for k in ["低", "中", "高"]}
        out.append({
            "方式": label, "掲載日数": landmark, "AUC": float(roc_auc_score(g["target"], g[score_col])),
            "低の終了率": rate["低"], "中の終了率": rate["中"], "高の終了率": rate["高"],
            "高−低(pt)": 100 * (rate["高"] - rate["低"]),
            "低の割合": share["低"], "高の割合": share["高"],
        })
    return pd.DataFrame(out)


def main() -> None:
    surv = prepare_survival_data(load_prepared(HERE.parents[1] / "data" / "prepared" / "kokubunji_raw.parquet"))
    surv = surv.drop_duplicates("id").set_index("id")
    val = pd.read_csv(HERE / "results/06_model_selection/model2_validation_predictions.csv")
    test = pd.read_csv(HERE / "results/07_final_test/final_model_2023_predictions.csv")
    val = val[val["horizon"] == 30].copy()
    test = test[test["horizon"] == 30].copy()
    val["model2"] = val["prediction"]
    test["model2"] = test["prediction"]

    train_val = surv.loc[surv["start_year"].isin([2020, 2021])]
    train_test = surv.loc[surv["start_year"].isin([2020, 2021, 2022])]
    static_val, static_test = fit_static(train_val), fit_static(train_test)
    val["duration_static"] = static_val.predict_proba(features(surv.loc[val["id"]]))[:, 1]
    test["duration_static"] = static_test.predict_proba(features(surv.loc[test["id"]]))[:, 1]

    # 方式2自身の課題(短い/長いを当てる)での性能。2023年掲載開始のうち短い・長い物件だけ。
    t23 = surv.loc[surv["start_year"] == 2023].copy()
    t23["y"] = duration_label(t23)
    t23 = t23.dropna(subset=["y"])
    own_auc = roc_auc_score(t23["y"].astype(int), static_test.predict_proba(features(t23))[:, 1])
    print(f"方式2が本来解く問題(2023年の短い/長い物件: 高{int(t23['y'].sum())}件, 低{int((1 - t23['y']).sum())}件)でのAUC: {own_auc:.3f}")

    weights = (1 / val.groupby("id")["id"].transform("size")).to_numpy(dtype=float)
    results = []
    for col, label in [("model2", "1 Model 2のリスク(報告書)"), ("duration_static", "2 短い/長いで学習(属性のみ)")]:
        q1 = weighted_quantile(val[col].to_numpy(dtype=float), weights, 1 / 3)
        q2 = weighted_quantile(val[col].to_numpy(dtype=float), weights, 2 / 3)
        print(f"{label}: 2022年の境界 {q1:.4f} / {q2:.4f}")
        results.append(evaluate(val, col, q1, q2, label).assign(年=2022))
        results.append(evaluate(test, col, q1, q2, label).assign(年=2023))
    table = pd.concat(results, ignore_index=True)
    out = HERE / "results" / "12_high_low_compare"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for year in (2022, 2023):
        print(f"\n=== {year}年 ===")
        print(table[table["年"] == year].drop(columns="年").round(3).to_string(index=False))
    print("\n平均(掲載7・14・30日目):")
    print(table.groupby(["年", "方式"])[["AUC", "高−低(pt)"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

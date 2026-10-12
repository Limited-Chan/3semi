"""既存の予測CSVから「高/低」と従来「高/中/低」を同じ対象で比較。

2022年の重み付き分位点で境界決定→2023年に固定適用。
必ず 06_model_selection.py と 07_final_test.py 実行後に実行。

ここでの「高/低」は、モデルが出した30日以内の掲載終了リスクを中央値で2つに分けたもの。
掲載期間が実際に短い/長い物件で分ける方式(Yamamoto側の「高と低」)とは別物。
3段階の境界(0.357134 / 0.516636)は報告書の35.71% / 51.66%を再現する。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
VALID = HERE / "results" / "06_model_selection" / "model2_validation_predictions.csv"
TEST = HERE / "results" / "07_final_test" / "final_model_2023_predictions.csv"
OUTPUT = HERE / "results" / "11_two_level_priority"
HORIZON = 30


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"予測CSVがありません: {path}")
    df = pd.read_csv(path)
    required = {"id", "landmark_day", "horizon", "target", "prediction"}
    if required - set(df.columns):
        raise ValueError(f"必要な列が欠けています: {required - set(df.columns)}")
    df = df[df["horizon"] == HORIZON].copy()
    if "model" in df.columns:
        df = df[df["model"] == "model2_age_layout"].copy()
    return df


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if len(cw) == 0 or cw[-1] <= 0:
        raise ValueError("重み付き分位点を計算できません")
    position = np.searchsorted(cw, quantile * cw[-1], side="left")
    return float(v[min(position, len(v) - 1)])


def evaluate(df: pd.DataFrame, year: int, median: float, q1: float, q2: float) -> pd.DataFrame:
    d = df.copy()
    d["year"] = year
    d["priority2"] = np.where(d["prediction"] >= median, "高", "低")
    d["priority3"] = np.select(
        [d["prediction"] < q1, d["prediction"] < q2], ["低", "中"], default="高"
    )
    output = []
    for landmark, group in d.groupby("landmark_day"):
        for scheme in ["priority2", "priority3"]:
            for label, sub in group.groupby(scheme):
                output.append({
                    "year": year, "landmark_day": landmark,
                    "scheme": scheme, "priority": label,
                    "n": len(sub), "actual_end_rate": float(sub["target"].mean()),
                    "mean_predicted_risk": float(sub["prediction"].mean()),
                    "share_of_candidates": float(len(sub) / len(group)),
                })
    return pd.DataFrame(output)


def main() -> None:
    valid = load_predictions(VALID)
    test = load_predictions(TEST)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # 一物件が7/14/30日目の3回出ても、境界の決定に使う重みは合計1。
    counts = valid.groupby("id")["id"].transform("size")
    weights = (1 / counts).to_numpy(dtype=float)
    values = valid["prediction"].to_numpy(dtype=float)
    median = weighted_quantile(values, weights, .5)
    q1 = weighted_quantile(values, weights, 1/3)
    q2 = weighted_quantile(values, weights, 2/3)
    print(f"2022年の高/低境界: {median:.6f} (これ以上は高)")
    print(f"従来の3段階境界: {q1:.6f} / {q2:.6f}")
    result = pd.concat([
        evaluate(valid, 2022, median, q1, q2),
        evaluate(test, 2023, median, q1, q2),
    ], ignore_index=True)
    result.to_csv(OUTPUT / "two_vs_three_level.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"boundary2": median, "boundary3_low": q1, "boundary3_high": q2}]).to_csv(
        OUTPUT / "thresholds_from_2022.csv", index=False, encoding="utf-8-sig"
    )
    print(result.to_string(index=False))
    print("\nこの値は正解率ではなく、区分ごとの実際の掲載終了率です。")
    print("現在の生存モデルと同じ、統合前の掲載レコードに基づく比較です。")


if __name__ == "__main__":
    main()

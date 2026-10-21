"""AUCの説明用に、2023年のROC曲線の点を表にする(発表スライドの図29h用)。

入力: results/17_fixed_window_features/candidates_2023.parquet(2023年の候補、7日以内の確率 base_7 と、実際に終わったか target_7)
出力: summary_tables/30_roc_2023.csv(偽陽性率・真陽性率・その点の確率の境界、約200点に間引き)
AUCは、この曲線の下の面積。全候補で計算する(似た候補の中のAUCではない)。
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
OUT = HERE / "summary_tables" / "30_roc_2023.csv"
H = 7


def roc(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """確率の高い順に並べ、境界を1つずつ下げたときの(偽陽性率, 真陽性率, 境界)。同じ確率はまとめて動かす。"""
    order = np.argsort(-p, kind="mergesort")
    p, y = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]], np.r_[1.0, p[last]]


def main() -> None:
    d = pd.read_parquet(SRC)
    y, p = d[f"target_{H}"].to_numpy().astype(float), d[f"base_{H}"].to_numpy()
    fpr, tpr, thr = roc(y, p)
    auc = np.trapz(tpr, fpr)
    keep = np.unique(np.r_[0, np.searchsorted(fpr, np.linspace(0, 1, 201)).clip(0, len(fpr) - 1), len(fpr) - 1])
    t = pd.DataFrame({"偽陽性率": fpr[keep], "真陽性率": tpr[keep], "境界": thr[keep]})
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"候補 {len(d):,}件、AUC {auc:.4f}、点 {len(t)}")


if __name__ == "__main__":
    main()

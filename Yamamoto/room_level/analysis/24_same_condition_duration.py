"""提案機能の元になる数字: 「同じ条件の物件は、掲載から何日で終わるか」(予測ではなく、過去データの事実)。

条件の組: 間取り × 築年数帯(10年刻み) × 家賃帯(2万円刻み)。国分寺市、2020〜2023年開始の22,332件。
出力(results/24_same_condition_duration/): same_condition_duration.csv
  n(件数)、掲載期間の25%・50%・75%点(日)、すでに12日載っている物件が、あと7日以内に終わる割合
方法: Kaplan–Meier法(2024-03-31まで載っていた物件=まだ終わっていない物件として扱う)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # リポジトリの直下(data/ がある場所)
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
from src.data import load_prepared

OUT = HERE / "results" / "24_same_condition_duration"
END = pd.Timestamp("2024-03-31")
ELAPSED, AHEAD = 12, 7
MIN_N = 100
AGE_BINS = [0, 10, 20, 30, 40, np.inf]
AGE_LABELS = ["10年未満", "10〜20年", "20〜30年", "30〜40年", "40年以上"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    s = d[d["start_year"].between(2020, 2023) & d["valid_complete_duration"]].copy()
    s["event"] = (s["end_date"] < END).astype(int)
    s["age_band"] = pd.cut(s["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    s["rent_band"] = (s["rent"] // 20000 * 2).astype("Int64")  # 万円の下限(2万円刻み)
    rows = []
    for (lay, age, rent), g in s.dropna(subset=["layout", "age_band", "rent_band"]).groupby(
            ["layout", "age_band", "rent_band"], observed=True):
        if len(g) < MIN_N:
            continue
        km = KaplanMeierFitter().fit(g["duration"], g["event"])
        sf = km.survival_function_["KM_estimate"]
        at = lambda t: float(sf[sf.index <= t].iloc[-1])
        q = lambda p: float(sf[sf <= 1 - p].index.min()) if (sf <= 1 - p).any() else np.nan
        s_now, s_next = at(ELAPSED), at(ELAPSED + AHEAD)
        rows.append({"間取り": lay, "築年数": str(age), "家賃(万円)": f"{rent}〜{rent + 2}", "件数": len(g),
                     "25%が終わる日": q(0.25), "半分が終わる日": q(0.5), "75%が終わる日": q(0.75),
                     f"{ELAPSED}日載った物件が、あと{AHEAD}日以内に終わる割合": 1 - s_next / s_now if s_now > 0 else np.nan})
    t = pd.DataFrame(rows).sort_values("件数", ascending=False)
    t.to_csv(OUT / "same_condition_duration.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(t.round(3).head(12).to_string(index=False))


if __name__ == "__main__":
    main()

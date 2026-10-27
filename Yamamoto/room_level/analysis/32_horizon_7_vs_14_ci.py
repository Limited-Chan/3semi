"""7日と14日の差に、信頼区間をつける(探索的)。

「なぜ1週間(7日)か」に答えるため、同じ候補・同じ抽出し直しで、7日と14日を比べる。
  1. 全物件のAUC(14日 − 7日)
  2. 似た物件どうし(間取り・築年数・家賃が近い)のAUC(14日 − 7日)
  3. 確率の上位30%を高としたときの、高は低の何倍か(7日 − 14日)
信頼区間は、同じ部屋ごとの抽出し直し(1000回。似た物件どうしのAUCだけ300回)の2.5%〜97.5%。
同じ抽出しで7日と14日を両方計算するので、「差」の区間になる。
入力: results/25_overall_ci_baselines/candidates_2022.parquet、results/17_fixed_window_features/candidates_2023.parquet
出力: summary_tables/32_horizon_7_vs_14_ci.csv
実行: cd Yamamoto/room_level && python3 analysis/32_horizon_7_vs_14_ci.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slide_figures", HERE / "analysis" / "31_slide_figures.py")
m = importlib.util.module_from_spec(spec)
sys.modules["slide_figures"] = m
spec.loader.exec_module(m)

FILES = {"2022年": m.C22, "2023年": m.C23}
REPS, REPS_WITHIN = 1000, 300
OUT = HERE / "summary_tables" / "32_horizon_7_vs_14_ci.csv"


def stats(d: pd.DataFrame, idx: np.ndarray, within: bool) -> dict:
    s = d.iloc[idx]
    out = {}
    for h in (7, 14):
        p, y = s[f"base_{h}"].to_numpy(), s[f"target_{h}"].to_numpy()
        hi = p >= np.quantile(p, 0.7)
        out[f"auc_{h}"] = m.auc(y, p)
        out[f"ratio_{h}"] = y[hi].mean() / y[~hi].mean()
        if within:
            out[f"within_{h}"] = m.within_auc(s.reset_index(drop=True), p, y, m.SIM_A)
    return out


def main() -> None:
    rows = []
    for year, path in FILES.items():
        d = pd.read_parquet(path).reset_index(drop=True)
        mem = m.members(d)
        rng = np.random.default_rng(20261027)
        full = stats(d, np.arange(len(d)), True)
        boots = []
        for r in range(REPS):
            idx = m.boot_unit_idx(mem, rng)
            boots.append(stats(d, idx, r < REPS_WITHIN))
        b = pd.DataFrame(boots)
        for label, a14, a7, key in [("全物件のAUC", "auc_14", "auc_7", "auc"), ("似た物件どうしのAUC", "within_14", "within_7", "within")]:
            diff = (b[a14] - b[a7]).dropna()
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"年": year, "指標": label, "14日": full[a14], "7日": full[a7], "差(14日−7日)": full[a14] - full[a7], "区間 下": lo, "区間 上": hi, "0を含む": lo <= 0 <= hi})
        diff = b["ratio_7"] - b["ratio_14"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"年": year, "指標": "高は低の何倍", "14日": full["ratio_14"], "7日": full["ratio_7"], "差(14日−7日)": full["ratio_14"] - full["ratio_7"],
                     "区間 下": -hi, "区間 上": -lo, "0を含む": lo <= 0 <= hi})
        print(year, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

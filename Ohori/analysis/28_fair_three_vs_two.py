"""2グループと3グループを、同じ方法で境界を決めて比べる(公平な比較、探索的)。

27では、2グループは目的から決めた15%、3グループは別の決め方だった。ここでは、どちらも同じ方法で決める。
  V1 等分      : 2022年の確率を、2等分(2グループ)/ 3等分(3グループ)
  V2 AUC最大   : 2022年のAUCが最大になる境界を、同じ探し方で選ぶ(各グループは全体の10%以上)
どちらも境界は2022年だけで決め、2023年と、未使用の2024年1〜3月に動かさず当てる。
見るもの: 各グループの割合と終了率、AUC(全体と似た候補の中)、3グループと2グループのAUCの差の95%区間。
"""
from __future__ import annotations

import importlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

m27 = importlib.import_module("analysis.27_three_vs_two_levels")
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "28_fair_three_vs_two"
H = 7
MIN_SHARE = 0.10
rng = np.random.default_rng(20261031)


def grouped_auc(y: np.ndarray, p: np.ndarray, cuts: list[float]) -> float:
    return roc_auc_score(y, m27.level(p, cuts).astype(float))


def best_cuts(p: np.ndarray, y: np.ndarray, k: int) -> list[float]:
    """2022年のAUCが最大になる境界(各グループが全体のMIN_SHARE以上)。"""
    grid = np.quantile(p, np.arange(0.10, 0.91, 0.02))
    grid = np.unique(grid)
    best, bc = -1.0, None
    for cuts in itertools.combinations(grid, k - 1):
        lv = m27.level(p, list(cuts))
        shares = np.bincount(lv, minlength=k) / len(p)
        if shares.min() < MIN_SHARE:
            continue
        a = roc_auc_score(y, lv.astype(float))
        if a > best:
            best, bc = a, list(cuts)
    return bc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: pd.read_parquet(v).reset_index(drop=True) for k, v in m27.SRC.items()}
    c22 = data["2022(境界を決めた年)"]
    p22, y22 = c22[f"base_{H}"].to_numpy(), c22[f"target_{H}"].to_numpy()
    sch = {
        "V1 2グループ(2等分)": [float(np.quantile(p22, 0.5))],
        "V1 3グループ(3等分)": np.quantile(p22, [1 / 3, 2 / 3]).tolist(),
        "V2 2グループ(AUC最大)": best_cuts(p22, y22, 2),
        "V2 3グループ(AUC最大)": best_cuts(p22, y22, 3),
    }
    print("境界(2022年で決めた):")
    for k, v in sch.items():
        print(f"  {k}: {[round(x, 4) for x in v]}  2022年のAUC {grouped_auc(y22, p22, v):.4f}")
    pairs = [("V1 2グループ(2等分)", "V1 3グループ(3等分)"), ("V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)")]
    grp_rows, met_rows = [], []
    for dname in ["2023(固定して適用)", "2024年1〜3月(未使用)"]:
        d = data[dname]
        y, p = d[f"target_{H}"].to_numpy(), d[f"base_{H}"].to_numpy()
        codes, uniq = pd.factorize(d["unit_key"].astype(str))
        members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
        scores = {"確率そのまま": p, **{k: m27.level(p, c).astype(float) for k, c in sch.items()}}
        for k, c in sch.items():
            lv = m27.level(p, c)
            names = ["低", "高"] if len(c) == 1 else ["低", "中", "高"]
            for i, nm in enumerate(names):
                msk = lv == i
                grp_rows.append({"対象": dname, "分け方": k, "グループ": nm, "割合": msk.mean(),
                                 "実際の終了率": y[msk].mean() if msk.any() else np.nan, "件数": int(msk.sum())})
        draws = {k: [] for k in scores}
        for _ in range(400):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
            for k, s in scores.items():
                draws[k].append(roc_auc_score(y[idx], s[idx]))
        for k, s in scores.items():
            row = {"対象": dname, "分け方": k, "全体のAUC": roc_auc_score(y, s)}
            for sn, keys in m27.SETS.items():
                row[f"AUC({sn})"] = m27.within_auc(d, s, keys)
            for two, three in pairs:
                if k == three:
                    diff = np.array(draws[three]) - np.array(draws[two])
                    row["3グループ−2グループ(全体のAUC)"] = roc_auc_score(y, s) - roc_auc_score(y, scores[two])
                    row["差_ci_low"], row["差_ci_high"] = np.percentile(diff, [2.5, 97.5])
            met_rows.append(row)
    g, m = pd.DataFrame(grp_rows), pd.DataFrame(met_rows)
    g.to_csv(OUT / "group_rates.csv", index=False, encoding="utf-8-sig")
    m.to_csv(OUT / "auc_by_scheme.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== 各グループの割合と実際の終了率 ===")
    print(g.round(3).to_string(index=False))
    print("\n=== AUC ===")
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

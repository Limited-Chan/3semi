"""2段階(高・低)と同じ手順で、3段階(高・中・低)を作って比べる(探索的)。

手順(2段階と同じ):
  1. 境界を2022年の候補で決める(または目的から決める)
  2. その境界を動かさず、2023年と、未使用の2024年1〜3月に当てる
比べるもの(7日以内に終わる確率 base_7 を、境界で区切る):
  2段階  : 15%以上=高 / 未満=低(いまの方式)
  3段階A : 2022年の確率の3分位(上位1/3=高、真ん中=中、下位1/3=低)(報告書の方式)
  3段階B : 低を2つに分ける(10%未満=低、10〜15%=中、15%以上=高)
  3段階C : 高を2つに分ける(15%未満=低、15〜25%=中、25%以上=高)
見るもの:
  - 各グループの割合と、実際の終了率(段が上がるほど、終了率が上がっているか)
  - AUC(確率そのまま / 各分け方)。同点は半分当たりとして数える。全体と、似た候補の中(間取りだけ同じ、間取り・築年数・家賃まで同じ)
  - 3段階と2段階のAUCの差の95%区間(同じ部屋ごとの抽出し直し)
注意: グループを粗くするほどAUCは下がるのが普通。2段階が3段階より低くなるのは当然で、問題は下がり幅。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "27_three_vs_two_levels"
H = 7
SRC = {
    "2022(境界を決めた年)": HERE / "results" / "25_overall_ci_baselines" / "candidates_2022.parquet",
    "2023(固定して適用)": HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet",
    "2024年1〜3月(未使用)": HERE / "results" / "26_holdout_2024q1" / "candidates_2024q1.parquet",
}
SETS = {"間取りだけ同じ": ["city", "layout_group"], "間取り・築年数・家賃まで同じ": ["city", "layout_group", "age_group", "rent_group"]}
rng = np.random.default_rng(20261030)


def schemes(c22: pd.DataFrame) -> dict[str, list[float]]:
    q = c22[f"base_{H}"].quantile([1 / 3, 2 / 3]).tolist()
    return {
        "2段階(15%)": [0.15],
        "3段階A(2022の3分位)": q,
        "3段階B(10%・15%)": [0.10, 0.15],
        "3段階C(15%・25%)": [0.15, 0.25],
    }


def level(p: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.searchsorted(np.array(cuts), p, side="right")


def within_auc(d: pd.DataFrame, score: np.ndarray, keys: list[str]) -> float:
    seg = d[["snapshot", *keys]].astype(str).agg("|".join, axis=1).to_numpy()
    y = d[f"target_{H}"].to_numpy()
    num = den = 0.0
    order = np.argsort(seg, kind="stable")
    seg_s = seg[order]
    bounds = np.flatnonzero(np.r_[True, seg_s[1:] != seg_s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yy = y[idx]
        npos, nneg = yy.sum(), len(yy) - yy.sum()
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(yy, score[idx])
        den += w
    return num / den


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: pd.read_parquet(v).reset_index(drop=True) for k, v in SRC.items()}
    sch = schemes(data["2022(境界を決めた年)"])
    print("境界:", {k: [round(x, 4) for x in v] for k, v in sch.items()})
    grp_rows, met_rows = [], []
    for dname in ["2023(固定して適用)", "2024年1〜3月(未使用)"]:
        d = data[dname]
        y = d[f"target_{H}"].to_numpy()
        p = d[f"base_{H}"].to_numpy()
        codes, uniq = pd.factorize(d["unit_key"].astype(str))
        members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
        scores = {"確率そのまま": p, **{k: level(p, c).astype(float) for k, c in sch.items()}}
        # 各グループの割合と終了率
        for k, c in sch.items():
            lv = level(p, c)
            names = ["低", "高"] if len(c) == 1 else ["低", "中", "高"]
            for i, nm in enumerate(names):
                m = lv == i
                grp_rows.append({"対象": dname, "分け方": k, "グループ": nm, "割合": m.mean(), "実際の終了率": y[m].mean() if m.any() else np.nan, "件数": int(m.sum())})
        # AUC
        B = 400
        draws = {k: [] for k in scores}
        for _ in range(B):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([members[i] for i in pick])
            for k, s in scores.items():
                draws[k].append(roc_auc_score(y[idx], s[idx]))
        for k, s in scores.items():
            row = {"対象": dname, "分け方": k, "全体のAUC": roc_auc_score(y, s)}
            for sn, keys in SETS.items():
                row[f"AUC({sn})"] = within_auc(d, s, keys)
            if k not in ("確率そのまま", "2段階(15%)"):
                diff = np.array(draws[k]) - np.array(draws["2段階(15%)"])
                row["全体のAUCの差(2段階との)"] = roc_auc_score(y, s) - roc_auc_score(y, scores["2段階(15%)"])
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

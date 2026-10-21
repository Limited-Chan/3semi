"""確率の較正と、高・低(+「差はほとんどない」)の境界を確かめる(探索的)。

入力: 17番が保存した2023年の候補(学習は2021〜2022年開始だけ=時点をずらした検証)
方式: base(Model 2) と hgb_all3(HGB+履歴・建物・需給)。現況・入居可能時期は使わない。
内容:
  1. 較正: 予測確率を10等分し、実際の終了率と比べる(7日・14日)
  2. 絶対の境界: 「予測確率がT以上=高」。Tを動かして、高の割合・高と低の実終了率を見る
  3. 候補内の表示: 同じ候補の中で、高と低の両方が出る集合の実終了率の差。
     点数の差(最大−最小)が小さい集合は「差はほとんどない」と出し、その中の実際の差を確かめる
注意: 境界を2023年の結果を見て選ぶ以上、2023年の数字は楽観的になる(探索的)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
IN = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
OUT = HERE / "results" / "18_calibration_labels"
MODELS = ["base", "hgb_all3"]
HORIZONS = [7, 14]
T_GRID = {7: [0.06, 0.08, 0.10, 0.12, 0.15, 0.20], 14: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]}
SPREAD = [0.02, 0.03, 0.05, 0.08]  # 「差はほとんどない」の幅(最大−最小の確率)
SETS = {"D 間取りだけ同じ": ["city", "layout_group"],
        "A 間取り×築年数×家賃": ["city", "layout_group", "age_group", "rent_group"]}
rng = np.random.default_rng(20261015)


def calibration(d: pd.DataFrame, model: str, h: int) -> pd.DataFrame:
    p, y = d[f"{model}_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(予測=("p", "mean"), 実際=("y", "mean"), 件数=("y", "size"))
    g["モデル"], g["予測期間(日)"] = model, h
    return g.reset_index(drop=True)


def threshold_table(d: pd.DataFrame, model: str, h: int) -> pd.DataFrame:
    p, y = d[f"{model}_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
    rows = []
    for t in T_GRID[h]:
        hi = p >= t
        if hi.sum() == 0 or (~hi).sum() == 0:
            continue
        rows.append({"モデル": model, "予測期間(日)": h, "境界T": t, "高の割合": hi.mean(),
                     "高の実終了率": y[hi].mean(), "低の実終了率": y[~hi].mean(), "全体": y.mean(),
                     "差pt": 100 * (y[hi].mean() - y[~hi].mean()), "高が捕まえる終了の割合": y[hi].sum() / y.sum()})
    return pd.DataFrame(rows)


def within_set(d: pd.DataFrame, model: str, h: int, keys: list[str], t: float, spread: float) -> dict:
    """集合ごとに、点数が境界以上=高/未満=低。高低が両方いて、点数の幅が spread 以上なら「判定あり」。"""
    d = d.assign(seg=d[["snapshot", *keys]].astype(str).agg("|".join, axis=1))
    out = {"decided": [], "undecided": [], "n_sets": 0}
    for _, g in d.groupby("seg"):
        if len(g) < 3:
            continue
        out["n_sets"] += 1
        p, y = g[f"{model}_{h}"].to_numpy(), g[f"target_{h}"].to_numpy()
        hi = p >= t
        if hi.all() or (~hi).all():
            continue
        diff = y[hi].mean() - y[~hi].mean()
        (out["decided"] if p.max() - p.min() >= spread else out["undecided"]).append((diff, len(g), g["seg"].iloc[0]))
    return out


def summarize(items: list, boot: int = 1000) -> tuple[float, float, float]:
    if not items:
        return (np.nan, np.nan, np.nan)
    diff = np.array([i[0] for i in items])
    w = np.array([i[1] for i in items], dtype=float)
    seg = np.array([i[2].split("|", 1)[1] for i in items])  # 日付を除いた集合の種類=グループ単位
    keys = np.unique(seg)
    idx = {k: np.flatnonzero(seg == k) for k in keys}
    est = 100 * np.average(diff, weights=w)
    bs = []
    for _ in range(boot):
        pick = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        bs.append(100 * np.average(diff[pick], weights=w[pick]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_parquet(IN)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cal, thr, win = [], [], []
    for label, d in [("近隣10市+国分寺市", cand), ("国分寺市", cand[cand["city"] == "国分寺市"])]:
        for model in MODELS:
            for h in HORIZONS:
                c = calibration(d, model, h)
                c["評価対象"] = label
                cal.append(c)
                t = threshold_table(d, model, h)
                t["評価対象"] = label
                thr.append(t)
    cal, thr = pd.concat(cal), pd.concat(thr)
    cal.to_csv(OUT / "calibration.csv", index=False, encoding="utf-8-sig")
    thr.to_csv(OUT / "threshold_table.csv", index=False, encoding="utf-8-sig")
    print("=== 1. 較正(近隣10市+国分寺市。予測と実際の差が小さいほど良い) ===")
    for (model, h), g in cal[cal["評価対象"] == "近隣10市+国分寺市"].groupby(["モデル", "予測期間(日)"]):
        print(f"\n[{model} {h}日]")
        print(g[["予測", "実際", "件数"]].round(3).to_string(index=False))
    print("\n=== 2. 絶対の境界(近隣10市+国分寺市) ===")
    print(thr[thr["評価対象"] == "近隣10市+国分寺市"].drop(columns="評価対象").round(3).to_string(index=False))

    print("\n=== 3. 候補内の高・低。境界=下の列T、「差はほとんどない」=点数の幅が spread 未満(近隣10市+国分寺市) ===")
    rows = []
    for model in MODELS:
        for h in HORIZONS:
            for sname, keys in SETS.items():
                for t in T_GRID[h]:
                    for sp in SPREAD:
                        r = within_set(cand, model, h, keys, t, sp)
                        n_dec, n_und = len(r["decided"]), len(r["undecided"])
                        if n_dec + n_und == 0:
                            continue
                        e1, l1, h1 = summarize(r["decided"], boot=300)
                        e2, _, _ = summarize(r["undecided"], boot=1)
                        rows.append({"モデル": model, "予測期間(日)": h, "候補の似かた": sname, "境界T": t, "幅spread": sp,
                                     "高低ありの集合": n_dec + n_und, "判定する集合": n_dec,
                                     "判定ありの差pt": e1, "ci_low": l1, "ci_high": h1, "「差なし」とした集合の差pt": e2})
    win = pd.DataFrame(rows)
    win.to_csv(OUT / "within_set_labels.csv", index=False, encoding="utf-8-sig")
    show = win[(win["予測期間(日)"] == 7) & (win["モデル"] == "base") & (win["境界T"] == 0.10)]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

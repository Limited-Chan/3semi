"""高・低の差に誤差の範囲をつけ、単純な目安(1つの条件だけ)と比べる。

問い:
  1. 全体の「高−低」の差(2023年、7日、境界15%)は、偶然とは言えない大きさか。
  2. 今のモデルは、掲載日数だけ・家賃だけ…の単純な目安より、高・低をうまく分けているか。
方法:
  - 単純な目安は、2022年の候補(5つの日付)で、条件の区分ごとの7日以内の終了率を数え、それを点数にする(2023年の結果は使わない)。
  - 高の割合は、モデルの15%境界と同じ(2023年で約30%)にそろえる。高=点数の上位その割合。
  - 誤差の範囲は、同じ部屋(unit_key)をまとめて抽出し直す方法(クラスター・ブートストラップ、1000回)。
注意: 2023年は何度も見たデータ。探索的。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
s19 = importlib.import_module("analysis.19_boundary_holdout")

OUT = HERE / "results" / "25_overall_ci_baselines"
H = 7
T = 0.15
ELAPSED_EDGES = [0, 7, 14, 21, 30, 45, 60, 90, 120, 10_000]
rng = np.random.default_rng(20261020)


def bins_elapsed(d: pd.DataFrame) -> pd.Series:
    return pd.cut(d["elapsed_day"], ELAPSED_EDGES, right=False).astype(str)


def rate_score(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """2022年の区分ごとの7日以内終了率を、2023年の点数にする。区分が無ければ全体率。"""
    tr, te = train.copy(), test.copy()
    tr["_e"], te["_e"] = bins_elapsed(tr), bins_elapsed(te)
    g = tr.groupby(keys)[f"target_{H}"].mean().rename("s").reset_index()
    out = te.merge(g, on=keys, how="left")["s"].fillna(tr[f"target_{H}"].mean())
    return out.to_numpy()


def labels(score: np.ndarray, share: float) -> np.ndarray:
    """点数の上位 share を高にする(同点は乱数で順序づけ、割合をそろえる)。"""
    order = np.lexsort((rng.random(len(score)), -score))
    hi = np.zeros(len(score), bool)
    hi[order[: int(round(share * len(score)))]] = True
    return hi


def stats(y: np.ndarray, hi: np.ndarray) -> tuple[float, float, float, float]:
    return y[hi].mean(), y[~hi].mean(), 100 * (y[hi].mean() - y[~hi].mean()), y[hi].mean() / y[~hi].mean()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "candidates_2022.parquet"
    c22 = pd.read_parquet(cache) if cache.exists() else s19.score_2022()
    if not cache.exists():
        c22.to_parquet(cache)
    c23 = pd.read_parquet(s19.CAND_2023).reset_index(drop=True)
    y = c23[f"target_{H}"].to_numpy()
    model_hi = c23[f"base_{H}"].to_numpy() >= T
    share = model_hi.mean()
    print(f"2023年の候補 {len(c23):,}件、高の割合 {share:.3f}、部屋 {c23['unit_key'].nunique():,}")

    scores = {
        "モデル(今回)": c23[f"base_{H}"].to_numpy(),
        "掲載日数だけ": rate_score(c22, c23, ["_e"]),
        "家賃帯だけ": rate_score(c22, c23, ["rent_group"]),
        "間取りだけ": rate_score(c22, c23, ["layout_group"]),
        "築年数帯だけ": rate_score(c22, c23, ["age_group"]),
        "間取り×築年数×家賃": rate_score(c22, c23, ["layout_group", "age_group", "rent_group"]),
        "でたらめ(並べ替え)": rng.random(len(c23)),
    }
    his = {k: (model_hi if k.startswith("モデル") else labels(v, share)) for k, v in scores.items()}

    # クラスター・ブートストラップ: 部屋ごとに抽出し直す
    codes, uniq = pd.factorize(c23["unit_key"].astype(str))
    n_u = len(uniq)
    members = [np.flatnonzero(codes == i) for i in range(n_u)]
    B = 1000
    draws = {k: [] for k in scores}
    ratios = []
    for _ in range(B):
        pick = rng.integers(0, n_u, n_u)
        idx = np.concatenate([members[i] for i in pick])
        yy = y[idx]
        for k in scores:
            h = his[k][idx]
            if h.all() or (~h).all():
                draws[k].append(np.nan)
                continue
            draws[k].append(100 * (yy[h].mean() - yy[~h].mean()))
        h = his["モデル(今回)"][idx]
        ratios.append(yy[h].mean() / yy[~h].mean())
    rows = []
    for k, sc in scores.items():
        hr, lr, gap, ratio = stats(y, his[k])
        d = np.array(draws[k])
        diff = d - np.array(draws["モデル(今回)"])
        rows.append({
            "目安": k, "高の割合": his[k].mean(), "高の実終了率": hr, "低の実終了率": lr,
            "差pt": gap, "差pt_ci_low": np.nanpercentile(d, 2.5), "差pt_ci_high": np.nanpercentile(d, 97.5),
            "倍率(高/低)": ratio, "AUC": roc_auc_score(y, sc),
            "モデルとの差pt(目安−モデル)": np.nanmean(diff) if k != "モデル(今回)" else 0.0,
            "モデルとの差_ci_low": np.nanpercentile(diff, 2.5) if k != "モデル(今回)" else 0.0,
            "モデルとの差_ci_high": np.nanpercentile(diff, 97.5) if k != "モデル(今回)" else 0.0,
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "overall_ci_baselines.csv", index=False, encoding="utf-8-sig")
    r = np.array(ratios)
    print(f"\nモデルの倍率(高/低) {y[model_hi].mean() / y[~model_hi].mean():.2f}倍、95%区間 {np.nanpercentile(r, 2.5):.2f}〜{np.nanpercentile(r, 97.5):.2f}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

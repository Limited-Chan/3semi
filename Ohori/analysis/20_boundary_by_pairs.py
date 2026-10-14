"""高・低の境界を、「2件を比べたときの当たり外れ」から決める(探索的)。

考え方: 同じ候補集合(同じ日・同じ市・同じ間取り、など)の中の物件を2件選ぶ。
  ラベルが「高」と「低」に分かれたら、「高」の方を先に問い合わせる。
  - 当たり: 「高」の物件だけが7日以内に終わった
  - 外れ  : 「低」の物件だけが7日以内に終わった
  正味 V = (当たり − 外れ) の割合(全ペアあたり)。大きいほど、ラベルが役に立つ。
  ほかに、ラベルが分かれるペアの割合、決着のつくペアでの当たり率。
手順: 2022年の候補(2021年開始で学習したModel 2)で、Vが最大になる境界を探す
      → その境界を、2023年の候補(2021〜2022年開始で学習)に固定して確かめる。
候補集合: D 同じ日・同じ市・同じ間取り / A さらに築年数帯・家賃帯も同じ(最も厳しい)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
s19 = importlib.import_module("analysis.19_boundary_holdout")

OUT = HERE / "results" / "20_pair_boundary"
CAND_2023 = s19.CAND_2023
H = 7
GRID = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
SETS = {"D 間取りだけ同じ": ["city", "layout_group"],
        "A 間取り×築年数×家賃": ["city", "layout_group", "age_group", "rent_group"]}
rng = np.random.default_rng(20261020)


def group_stats(d: pd.DataFrame, keys: list[str], t: float) -> pd.DataFrame:
    p, y = d[f"base_{H}"].to_numpy(), d[f"target_{H}"].to_numpy()
    g = d[["snapshot", *keys]].astype(str).agg("|".join, axis=1)
    f = pd.DataFrame({"g": g.to_numpy(), "hi": p >= t, "y": y})
    f["yh"] = f["y"] * f["hi"]
    f["yl"] = f["y"] * (~f["hi"])
    a = f.groupby("g").agg(n=("y", "size"), H=("hi", "sum"), yH=("yh", "sum"), yL=("yl", "sum"))
    a["L"] = a["n"] - a["H"]
    a["pairs"] = a["n"] * (a["n"] - 1) / 2
    a["differ"] = a["H"] * a["L"]
    a["win"] = a["yH"] * (a["L"] - a["yL"])      # 高だけ終了
    a["lose"] = (a["H"] - a["yH"]) * a["yL"]     # 低だけ終了
    a["cluster"] = [x.split("|", 1)[1] for x in a.index]
    return a.loc[a["pairs"] > 0]


def metrics(a: pd.DataFrame) -> dict:
    pairs = a["pairs"].sum()
    win, lose = a["win"].sum(), a["lose"].sum()
    return {"ラベルが分かれるペア": a["differ"].sum() / pairs, "V(当たり−外れ)": (win - lose) / pairs,
            "決着のつくペアでの当たり率": win / (win + lose) if win + lose else np.nan}


def boot_V(a: pd.DataFrame, reps: int = 400) -> tuple[float, float]:
    cl = a["cluster"].to_numpy()
    uk = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in uk}
    pairs, win, lose = a["pairs"].to_numpy(), a["win"].to_numpy(), a["lose"].to_numpy()
    vs = []
    for _ in range(reps):
        pick = np.concatenate([idx[uk[i]] for i in rng.integers(0, len(uk), len(uk))])
        vs.append((win[pick].sum() - lose[pick].sum()) / pairs[pick].sum())
    return tuple(np.percentile(vs, [2.5, 97.5]))


def table(d: pd.DataFrame, year: str) -> pd.DataFrame:
    rows = []
    p = d[f"base_{H}"].to_numpy()
    for sname, keys in SETS.items():
        for t in GRID:
            a = group_stats(d, keys, t)
            m = metrics(a)
            lo, hi = boot_V(a)
            rows.append({"年": year, "候補の似かた": sname, "境界": t, "高の割合(全体)": float((p >= t).mean()),
                         **m, "V_ci_low": lo, "V_ci_high": hi})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c22 = s19.score_2022()
    c23 = pd.read_parquet(CAND_2023)
    t22, t23 = table(c22, "2022(境界を探した年)"), table(c23, "2023(固定して確認)")
    tab = pd.concat([t22, t23])
    tab.to_csv(OUT / "pair_boundary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, t in [("2022", t22), ("2023", t23)]:
        print(f"\n=== {name} ===")
        print((t.assign(**{c: (100 * t[c]).round(2) for c in ["V(当たり−外れ)", "V_ci_low", "V_ci_high", "ラベルが分かれるペア", "高の割合(全体)", "決着のつくペアでの当たり率"]}))
              .drop(columns="年").to_string(index=False))


if __name__ == "__main__":
    main()

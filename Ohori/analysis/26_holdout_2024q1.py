"""2024年1〜3月に掲載を始めた物件で、決めたとおりのルールが通用するか確かめる(未使用データでの確認)。

事前に決めて、動かさないもの(2023年までの結果で決めた):
  - モデル: 2021〜2022年開始で学習した Model 2(2023年の確認と同じもの。学習し直さない)
  - 予測期間: 7日、境界: 15%(確率が15%以上=高)
  - 単純な目安: 2022年の候補から作ったもの(25番と同じ)
評価: 2024年に掲載を始めた物件(2023年の確認では使っていない物件)のうち、
      近隣10市+国分寺市で、5つの日付(2024-01-15, 02-01, 02-15, 03-01, 03-15)に掲載中のもの。
      データの終わりは2024-03-31なので、7日先まで見える日付だけを使う。
注意: 掲載開始から日が浅い物件だけ(最大90日)。2023年の評価(最大331日)とは経過日数の分布が違う。
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
m = importlib.import_module("analysis.09_candidate_ranking")
s15 = importlib.import_module("analysis.15_scope_comparison")
s17 = importlib.import_module("analysis.17_fixed_window_features")
s19 = importlib.import_module("analysis.19_boundary_holdout")
s25 = importlib.import_module("analysis.25_overall_ci_baselines")
from src.data import SOURCE_COLUMNS, prepare
from src.listing_episodes import build_episodes_fast

OUT = HERE / "results" / "26_holdout_2024q1"
Q1_RAW = ROOT / "data" / "prepared" / "tama_2024q1_raw.parquet"
H, T = 7, 0.15
SNAPSHOTS_2024 = pd.to_datetime(["2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-15"])
rng = np.random.default_rng(20261024)


def extract_q1() -> pd.DataFrame:
    if Q1_RAW.exists():
        return pd.read_parquet(Q1_RAW)
    t0 = time.time()
    chunks = []
    for ch in pd.read_csv(s15.TSV, sep="\t", usecols=SOURCE_COLUMNS, dtype=str, chunksize=300_000,
                          encoding="utf-8", keep_default_na=False):
        f = ch[(ch["addr1_1_name"] == "東京都") & (~ch["addr1_2_name"].str.endswith("区"))
               & (ch["pub_start_date"].str[:4] == "2024")]
        if len(f):
            chunks.append(f)
    d = pd.concat(chunks, ignore_index=True)
    d.to_parquet(Q1_RAW, index=False)
    print(f"2024年開始の多摩地域を抽出: {len(d):,}件 ({time.time() - t0:.0f}秒)")
    return d


def candidates(ep: pd.DataFrame, model, year: int, snaps) -> pd.DataFrame:
    ev = ep.loc[ep["city"].isin(s15.NEAR) & (ep["start_year"] == year)]
    frames = []
    for snap in snaps:
        c = ev.loc[(ev["start_date"] <= snap) & (ev["end_date"] > snap)].copy()
        c["elapsed_day"] = (snap - c["start_date"]).dt.days.astype(int)
        c[f"target_{H}"] = ((c["end_date"] <= snap + pd.Timedelta(days=H)) & (c["end_date"] < m.OBSERVATION_END)).astype(int)
        c[f"base_{H}"] = model.risk(c, H)
        c["snapshot"] = snap.date().isoformat()
        frames.append(m.add_bins(c))
        print(f"  {snap.date()}: 候補 {len(c):,}件")
    return pd.concat(frames, ignore_index=True)


def summarize(d: pd.DataFrame, label: str, c22: pd.DataFrame) -> list[dict]:
    d = d.reset_index(drop=True)
    y = d[f"target_{H}"].to_numpy()
    hi = d[f"base_{H}"].to_numpy() >= T
    share = hi.mean()
    scores = {
        "モデル(今回)": d[f"base_{H}"].to_numpy(),
        "掲載日数だけ": s25.rate_score(c22, d, ["_e"]),
        "家賃帯だけ": s25.rate_score(c22, d, ["rent_group"]),
        "間取りだけ": s25.rate_score(c22, d, ["layout_group"]),
        "築年数帯だけ": s25.rate_score(c22, d, ["age_group"]),
        "間取り×築年数×家賃": s25.rate_score(c22, d, ["layout_group", "age_group", "rent_group"]),
        "でたらめ(並べ替え)": rng.random(len(d)),
    }
    his = {k: (hi if k.startswith("モデル") else s25.labels(v, share)) for k, v in scores.items()}
    codes, uniq = pd.factorize(d["unit_key"].astype(str))
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    draws = {k: [] for k in scores}
    ratios = []
    for _ in range(1000):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        yy = y[idx]
        for k in scores:
            h = his[k][idx]
            draws[k].append(np.nan if (h.all() or (~h).all()) else 100 * (yy[h].mean() - yy[~h].mean()))
        h = his["モデル(今回)"][idx]
        ratios.append(yy[h].mean() / yy[~h].mean() if (~h).any() and h.any() else np.nan)
    rows = []
    for k, sc in scores.items():
        hr, lr, gap, ratio = s25.stats(y, his[k])
        dd = np.array(draws[k])
        diff = dd - np.array(draws["モデル(今回)"])
        rows.append({
            "対象": label, "目安": k, "候補数": len(d), "部屋数": len(uniq), "高の割合": his[k].mean(),
            "高の実終了率": hr, "低の実終了率": lr, "全体": y.mean(), "差pt": gap,
            "差pt_ci_low": np.nanpercentile(dd, 2.5), "差pt_ci_high": np.nanpercentile(dd, 97.5),
            "倍率(高/低)": ratio,
            "倍率_ci_low": np.nanpercentile(ratios, 2.5) if k.startswith("モデル") else np.nan,
            "倍率_ci_high": np.nanpercentile(ratios, 97.5) if k.startswith("モデル") else np.nan,
            "AUC": roc_auc_score(y, sc), "平均の予測": d[f"base_{H}"].mean() if k.startswith("モデル") else np.nan,
            "モデルとの差pt(目安−モデル)": 0.0 if k.startswith("モデル") else np.nanmean(diff),
            "モデルとの差_ci_low": 0.0 if k.startswith("モデル") else np.nanpercentile(diff, 2.5),
            "モデルとの差_ci_high": 0.0 if k.startswith("モデル") else np.nanpercentile(diff, 97.5),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = pd.concat([s15.extract_tama(), extract_q1()], ignore_index=True)
    raw = prepare(raw)
    raw["city"] = raw["addr1_2_name"]
    ep = build_episodes_fast(raw, max_gap_days=1)
    ep["city"] = ep["addr1_2_name"]
    ep = m.add_bins(ep)
    print(f"部屋エピソード {len(ep):,}件、うち2024年開始 {(ep['start_year'] == 2024).sum():,}件 ({time.time() - t0:.0f}秒)")

    train = m.make_survival_table(ep.loc[ep["start_year"].isin([2021, 2022])].copy(), m.TRAIN_CUTOFF)
    print(f"学習(2021〜2022年開始): {len(train):,}件")
    model = s17.Hazard("lr", []).fit(train)

    print("確認用: 2023年(これまでと同じ結果になるはず)")
    c23 = candidates(ep, model, 2023, m.SNAPSHOTS)
    chk = s19.label_table(c23, H, T)
    print({k: round(float(v), 4) for k, v in chk.items()})
    print("2024年1〜3月")
    c24 = candidates(ep, model, 2024, SNAPSHOTS_2024)
    c24.to_parquet(OUT / "candidates_2024q1.parquet", index=False)

    c22 = pd.read_parquet(HERE / "results" / "25_overall_ci_baselines" / "candidates_2022.parquet")
    rows = summarize(c24, "近隣10市+国分寺市", c22)
    rows += summarize(c24[c24["city"] == s15.TARGET_CITY], "国分寺市のみ", c22)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "holdout_2024q1.csv", index=False, encoding="utf-8-sig")

    ws = []
    for sname, keys in s19.SETS.items():
        est, lo, hi_, n = s19.within_set(c24, H, T, keys)
        ws.append({"候補の似かた": sname, "高低が分かれた集合": n, "高−低の実終了率の差pt": est, "ci_low": lo, "ci_high": hi_})
    pd.DataFrame(ws).to_csv(OUT / "holdout_2024q1_within_set.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== 2024年1〜3月(未使用データ) ===")
    print(t.round(3).to_string(index=False))
    print("\n=== 同じ候補の中での高−低 ===")
    print(pd.DataFrame(ws).round(2).to_string(index=False))
    print(f"\n総時間 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()

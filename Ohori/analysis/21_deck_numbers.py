"""発表資料に載せた、掲載日数・属性・件数の数字を、再現できる形で出力する。

出力(results/21_deck_numbers/): duration_summary.csv, attribute_table.csv, counts.csv
入力: data/prepared/kokubunji_raw.parquet(国分寺市)、契約日の件数だけ元のTSVを読む
対象: 2020〜2023年に掲載開始の22,332件(国分寺市)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from src.data import load_prepared
from src.listing_episodes import build_episodes_fast, prepare_room_keys

TSV = Path("/Users/aki/Downloads/bukken_rent_v1.0.1.tsv")
OUT = HERE / "results" / "21_deck_numbers"
END = pd.Timestamp("2024-03-31")


def eta(df: pd.DataFrame, col: str) -> float:
    x = df[[col, "duration"]].dropna()
    g = x.groupby(col, observed=True)["duration"]
    return float(np.sqrt((g.size() * (g.mean() - x["duration"].mean()) ** 2).sum()
                         / ((x["duration"] - x["duration"].mean()) ** 2).sum()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    scope = d[d["start_year"].between(2020, 2023)]
    s = scope[scope["valid_complete_duration"]].copy()
    s["event"] = (s["end_date"] < END).astype(int)

    km = KaplanMeierFitter().fit(s["duration"], s["event"])
    rows = [("件数", len(s)), ("中央値(日)", float(s["duration"].median())), ("平均(日)", float(s["duration"].mean())),
            ("右打ち切り(件)", int((1 - s["event"]).sum()))]
    rows += [(f"{t}日後までに終了(KM)", float(1 - km.predict(t))) for t in [3, 7, 14, 22, 30, 60, 90, 180]]
    bins = [-1, 6, 13, 29, 59, 89, 179, 10**6]
    labels = ["0〜6日", "7〜13日", "14〜29日", "30〜59日", "60〜89日", "90〜179日", "180日以上"]
    share = pd.cut(s["duration"], bins=bins, labels=labels).value_counts(normalize=True, sort=False)
    rows += [(f"掲載件数の割合 {k}", float(v)) for k, v in share.items()]
    pd.DataFrame(rows, columns=["指標", "値"]).to_csv(OUT / "duration_summary.csv", index=False, encoding="utf-8-sig")

    attr = []
    for col, name in [("age", "築年数(年)"), ("rent", "家賃(円)"), ("area", "専有面積(㎡)"),
                      ("walk_distance", "駅までの距離(m)"), ("start_month", "掲載開始月")]:
        ok = s[col].notna()
        attr.append({"属性": name, "欠損率": float(scope[col].isna().mean()),
                     "中央値": float(s[col].median()), "最小": float(s[col].min()), "最大": float(s[col].max()),
                     "順位相関": float(spearmanr(s.loc[ok, col], s.loc[ok, "duration"])[0]) if col != "start_month" else np.nan,
                     "相関比": eta(s, col) if col == "start_month" else np.nan})
    for col, name in [("layout", "間取り"), ("station", "最寄駅")]:
        attr.append({"属性": name, "欠損率": float(scope[col].isna().mean()), "種類数": int(s[col].nunique()),
                     "最多": s[col].value_counts().index[0], "最多の割合": float(s[col].value_counts(normalize=True).iloc[0]),
                     "相関比": eta(s, col)})
    pd.DataFrame(attr).to_csv(OUT / "attribute_table.csv", index=False, encoding="utf-8-sig")

    med = []
    for col, bins_, name in [("age", [0, 10, 20, 30, 40, 200], "築年数"), ("rent", [0, 50000, 70000, 90000, 110000, 1e9], "家賃")]:
        for k, v in s.groupby(pd.cut(s[col], bins_, right=(col == "rent")), observed=True)["duration"].median().items():
            med.append({"区分": f"{name} {k}", "掲載日数の中央値": float(v)})
    for k, v in s.groupby("layout")["duration"].median().items():
        if k in ("1K", "1R", "1LDK", "2LDK"):
            med.append({"区分": f"間取り {k}", "掲載日数の中央値": float(v)})
    for k, v in s.groupby("start_month")["duration"].median().items():
        med.append({"区分": f"掲載開始月 {k}月", "掲載日数の中央値": float(v)})
    pd.DataFrame(med).to_csv(OUT / "group_medians.csv", index=False, encoding="utf-8-sig")

    keyed = prepare_room_keys(d)
    ep = build_episodes_fast(d, max_gap_days=1)
    counts = [("掲載(2020〜2023年開始)", len(scope)),
              ("部屋(building_id×unit_id)", int(keyed.loc[keyed["start_year"].between(2020, 2023), "unit_key"].nunique())),
              ("部屋単位エピソード(統合ルール1日)", int(ep["start_year"].between(2020, 2023).sum()))]
    if TSV.exists():
        n = 0
        for ch in pd.read_csv(TSV, sep="\t", usecols=["addr1_2_name", "pub_start_date", "keiyaku_date"], dtype=str,
                              chunksize=500_000, keep_default_na=False):
            f = ch[(ch["addr1_2_name"] == "国分寺市") & ch["pub_start_date"].str[:4].isin(["2020", "2021", "2022", "2023"])]
            n += int((f["keiyaku_date"].str.strip() != "").sum())
        counts.append(("契約日(keiyaku_date)が入っている掲載", n))
    pd.DataFrame(counts, columns=["指標", "値"]).to_csv(OUT / "counts.csv", index=False, encoding="utf-8-sig")
    for f in ["duration_summary", "attribute_table", "group_medians", "counts"]:
        print(f"\n=== {f} ===")
        print(pd.read_csv(OUT / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アプリの「問い合わせ優先度」向け: 築年数 × 間取り から早期終了率を求め、
新着物件を 高/中/低 の3段階にランク分けするための参照テーブルを作る。

「早く掲載が終わった物件 ≒ 需要が高かった物件」という代理指標の考え方に基づき、
過去実績(2020〜2023年掲載開始、国分寺市)から
- 7日/14日/30日以内に掲載終了した割合
- 中央値・よくある範囲(Q1〜Q3)
を築年数帯×間取りごとに求め、30日以内終了率の実績件数を三等分するしきい値で
優先度ランクを割り当てる。

実行方法:
    python3 priority_score.py                       # prepared/ のキャッシュを使う
    python3 priority_score.py --input /path/to.tsv   # キャッシュが無い場合
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import (
    AGE_DISPLAY_MAX,
    OUT_DIR,
    TABLE_DIR,
    TOP_LAYOUTS,
    build_age_bins,
    load_data,
)


def build_priority_table(d: pd.DataFrame, edges: np.ndarray, labels: list[str]) -> pd.DataFrame:
    sub = d.dropna(subset=["age", "layout"]).copy()
    sub["age_band"] = pd.cut(sub["age"], bins=edges, labels=labels, right=False, include_lowest=True)

    rows = []
    for age_band in labels:
        for layout in TOP_LAYOUTS:
            g = sub[(sub["age_band"] == age_band) & (sub["layout"] == layout)]["duration"]
            n = len(g)
            if n == 0:
                rows.append({"age_band": age_band, "layout": layout, "n": 0})
                continue
            rows.append({
                "age_band": age_band, "layout": layout, "n": n,
                "median": float(g.median()), "q1": float(g.quantile(0.25)), "q3": float(g.quantile(0.75)),
                "pct_7d": float((g <= 7).mean()), "pct_14d": float((g <= 14).mean()), "pct_30d": float((g <= 30).mean()),
            })
    table = pd.DataFrame(rows)
    table["age_band"] = pd.Categorical(table["age_band"], categories=labels, ordered=True)
    return table.sort_values(["age_band", "layout"]).reset_index(drop=True)


def assign_priority_rank(table: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """30日以内終了率をもとに、件数(実件数ベース)で三等分して 高/中/低 を割り当てる。
    n が少ないセル(min_n未満)は判定不能として '参考値(n不足)' にする。"""
    t = table.copy()
    t["priority_rank"] = "参考値(n不足)"
    reliable = t[t["n"] >= min_n].sort_values("pct_30d", ascending=False).copy()
    reliable["cum_n"] = reliable["n"].cumsum()
    total_n = reliable["n"].sum()
    reliable["priority_rank"] = np.where(
        reliable["cum_n"] <= total_n / 3, "高",
        np.where(reliable["cum_n"] <= total_n * 2 / 3, "中", "低"),
    )
    t.loc[reliable.index, "priority_rank"] = reliable["priority_rank"]
    return t


def plot_heatmap(table: pd.DataFrame, edges: np.ndarray, labels: list[str]) -> None:
    display_labels = [lab for lab, lo in zip(labels, edges[:-1]) if lo < AGE_DISPLAY_MAX]
    pivot = table[table["age_band"].isin(display_labels)].pivot(index="layout", columns="age_band", values="pct_30d")
    pivot = pivot.reindex(index=TOP_LAYOUTS, columns=display_labels)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(TOP_LAYOUTS)))
    ax.set_yticklabels(TOP_LAYOUTS)
    for i, layout in enumerate(TOP_LAYOUTS):
        for j, band in enumerate(display_labels):
            row = table[(table["layout"] == layout) & (table["age_band"] == band)]
            if row.empty or pd.isna(row.iloc[0].get("pct_30d")):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="gray")
                continue
            r = row.iloc[0]
            label = f"{r['pct_30d']:.0%}\nn={int(r['n'])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="black" if 0.3 < r["pct_30d"] < 0.8 else "white")
    ax.set_title("築年数 × 間取り: 30日以内に掲載終了した割合(緑=優先度高いと推定, 赤=低い)")
    fig.colorbar(im, ax=ax, label="30日以内終了率", shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_priority_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame) -> None:
    reliable = table[table["priority_rank"].isin(["高", "中", "低"])]
    lines = ["# 問い合わせ優先度スコア(参考実装)", ""]
    lines.append(
        "「掲載期間が短い ≒ 需要が高かった」という代理指標に基づき、"
        "築年数帯 × 間取り(主要8種)ごとの**30日以内終了率**を過去実績(2020〜2023年、n>=30のセルのみ)から求め、"
        "実件数を三等分するしきい値で 高/中/低 の3段階にランク分けした。新着物件が来たら、"
        "その築年数・間取りに対応するランクを引くだけで使える簡易な参照テーブル。"
    )
    lines.append("")
    lines.append("## ランク別のしきい値(実測)")
    for rank in ["高", "中", "低"]:
        rr = reliable[reliable["priority_rank"] == rank]
        if len(rr):
            lines.append(f"- **{rank}**: 30日以内終了率 {rr['pct_30d'].min():.0%}〜{rr['pct_30d'].max():.0%}"
                         f"(該当セル{len(rr)}件、合計n={int(rr['n'].sum()):,})")
    lines.append("")
    lines.append("## 優先度「高」の組み合わせ(30日以内終了率が高い順、上位8)")
    top = reliable[reliable["priority_rank"] == "高"].sort_values("pct_30d", ascending=False).head(8)
    for _, r in top.iterrows():
        lines.append(f"- 築年数{r['age_band']} × {r['layout']}: 30日以内終了率{r['pct_30d']:.0%}"
                     f"(n={int(r['n']):,}, 中央値{r['median']:.0f}日)")
    lines.append("")
    lines.append("## 優先度「低」の組み合わせ(30日以内終了率が低い順、上位8)")
    bottom = reliable[reliable["priority_rank"] == "低"].sort_values("pct_30d").head(8)
    for _, r in bottom.iterrows():
        lines.append(f"- 築年数{r['age_band']} × {r['layout']}: 30日以内終了率{r['pct_30d']:.0%}"
                     f"(n={int(r['n']):,}, 中央値{r['median']:.0f}日)")
    lines.append("")
    lines.append("## 使い方のイメージ")
    lines.append(
        "新着物件の「築年数」「間取り」から `tables/05_priority_score.csv` を検索し、"
        "priority_rank をそのままアプリの優先度バッジ(高/中/低)として表示する。"
        "n不足のセル(参考値)は表示を避けるか、上位カテゴリ(例: 築年数帯を1つ広げる)にフォールバックする。"
    )
    lines.append("")
    lines.append("## 限界・注意点")
    lines.append(
        "- あくまで「掲載期間の短さ」を需要の代理指標にしているだけで、実際の問い合わせ数データではない。\n"
        "- 築年数×間取りだけでの説明力はR^2で7%程度(`report.md`参照)と低く、同じセル内でもばらつきは大きい。\n"
        "- 家賃・面積などを組み合わせればランクの精度は上げられるが、その分セルが細分化されnが不足しやすくなる。\n"
        "- 「掲載終了」は成約とは限らない(掲載主都合の取り下げ等を含む)ため、優先度スコアは絶対視せず"
        "他の指標と併用することを推奨。"
    )
    lines.append("")
    (OUT_DIR.parent / "priority_score_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"レポートを保存: {OUT_DIR.parent / 'priority_score_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    d = load_data(args.input)
    edges, labels = build_age_bins(d)
    table = build_priority_table(d, edges, labels)
    table = assign_priority_rank(table)
    table.to_csv(TABLE_DIR / "05_priority_score.csv", index=False, encoding="utf-8-sig")
    print(f"優先度テーブルを保存: {TABLE_DIR / '05_priority_score.csv'}")

    reliable = table[table["priority_rank"].isin(["高", "中", "低"])]
    print(reliable.groupby("priority_rank", observed=True)["n"].agg(["count", "sum"]))

    plot_heatmap(table, edges, labels)
    write_report(table)


if __name__ == "__main__":
    main()

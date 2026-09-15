#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
築年数・間取りと掲載期間の関係を分析する。

リポジトリには物件1件ごとの生データは含まれておらず(容量が大きいため
analysis_kokubunji/ 以下の集計済みCSVのみが管理されている)、本スクリプトは
その集計値(件数・平均・中央値・25/75パーセンタイル)を入力として使う。
そのため一部の指標は listing 単位の生データがあれば計算できるものの近似値
であり、各グラフ・数値にその旨を明記する。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parent.parent / "analysis_kokubunji"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

AGE_ORDER = ["築5年以下", "築6〜10年", "築11〜20年", "築21〜30年", "築31〜40年", "築41年以上"]
AGE_MIDPOINT = {
    "築5年以下": 2.5,
    "築6〜10年": 8,
    "築11〜20年": 15.5,
    "築21〜30年": 25.5,
    "築31〜40年": 35.5,
    "築41年以上": 45,  # 上限なしの帯のため仮の代表値(下限+4年)を採用
}


def load_data() -> dict[str, pd.DataFrame]:
    age = pd.read_csv(DATA_DIR / "06_building_age.csv", encoding="utf-8-sig")
    layout = pd.read_csv(DATA_DIR / "03_layout.csv", encoding="utf-8-sig")
    quality = pd.read_csv(DATA_DIR / "91_quality_report.csv", encoding="utf-8-sig")
    metadata = json.loads((DATA_DIR / "92_analysis_metadata.json").read_text(encoding="utf-8"))
    return {"age": age, "layout": layout, "quality": quality, "metadata": metadata}


def step1_overview(data: dict) -> None:
    print("=" * 60)
    print("手順1: 事前確認")
    print("=" * 60)
    for name in ["age", "layout"]:
        df = data[name]
        print(f"\n[{name}] shape={df.shape}")
        print(df.dtypes)

    q = data["quality"].set_index("項目")["値"]
    print("\n--- 欠損・異常値(quality report より) ---")
    print(f"国分寺市有効行数: {int(q['国分寺市有効行数'])}")
    print(f"築年数欠損件数: {int(q['築年数欠損件数'])} ({q['築年数欠損率']:.2%})")
    print(f"間取り欠損件数: {int(q['間取り欠損件数'])} ({q['間取り欠損率']:.2%})")
    print(f"掲載期間が負数: {int(q['掲載期間が負数'])}")
    print(f"掲載期間が3650日超: {int(q['掲載期間が3650日超'])}")
    print(f"重複id件数(未削除): {int(q['重複id件数_未削除'])}")
    print("\n※ 欠損・異常値・重複はいずれも集計元スクリプト側で既に除外/処理済み。")


def step2_age_distribution(age: pd.DataFrame) -> None:
    df = age.set_index("築年数帯").loc[AGE_ORDER]
    counts = df["件数"]

    weighted_mean = np.average([AGE_MIDPOINT[a] for a in AGE_ORDER], weights=counts)
    mode_band = counts.idxmax()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(AGE_ORDER, counts, color="#4C72B0")
    ax.set_xlabel("築年数帯")
    ax.set_ylabel("件数")
    ax.set_title("築年数の分布(帯別件数)")
    ax.axhline(0, color="black", linewidth=0.8)
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.text(
        0.01, -0.02,
        f"帯の代表値による加重平均年数(目安): 約{weighted_mean:.1f}年 / 最頻値の帯: {mode_band}\n"
        "※ 生データが無いため、既存の帯区分(不均等幅)をそのまま採用。1年/5年刻みのヒストグラムは作成不可。",
        fontsize=8, color="dimgray",
    )
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_age_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[手順2-1] 築年数分布")
    print(f"  最頻値の帯: {mode_band} ({counts.max():,}件)")
    print(f"  帯代表値による加重平均年数(目安): {weighted_mean:.1f}年")


def step2_layout_distribution(layout: pd.DataFrame) -> None:
    df = layout.sort_values("件数", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["間取り"], df["件数"], color="#DD8452")
    ax.set_xlabel("間取り")
    ax.set_ylabel("件数")
    ax.set_title("間取り別の件数")
    for i, v in enumerate(df["件数"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_layout_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[手順2-2] 間取り分布")
    print(f"  最頻値(最多件数の間取り): {df.iloc[0]['間取り']} ({df.iloc[0]['件数']:,}件)")
    small = df[df["件数"] < 100]
    if len(small):
        print(f"  件数が少なく「その他」候補: {', '.join(small['間取り'])}")


def _quartile_box(ax, labels, q1, med, q3, color):
    stats_list = [
        {"med": m, "q1": a, "q3": b, "whislo": a, "whishi": b, "fliers": []}
        for a, m, b in zip(q1, med, q3)
    ]
    bxp_artists = ax.bxp(
        stats_list, showfliers=False, patch_artist=True,
        positions=range(len(labels)),
    )
    for box in bxp_artists["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)


def step3_age_vs_duration(age: pd.DataFrame) -> None:
    df = age.set_index("築年数帯").loc[AGE_ORDER].reset_index()

    # 帯単位の加重相関(参考値。listing単位の相関係数とは異なる)
    x = [AGE_MIDPOINT[a] for a in df["築年数帯"]]
    y_mean = df["掲載期間平均日数"]
    r_pearson, _ = stats.pearsonr(x, y_mean)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x, df["掲載期間平均日数"], "o-", label="平均日数")
    axes[0].plot(x, df["掲載期間中央値"], "s--", label="中央値日数")
    axes[0].set_xlabel("築年数帯の代表値(年)")
    axes[0].set_ylabel("掲載期間(日)")
    axes[0].set_title(f"築年数 × 掲載期間(帯単位, 参考r={r_pearson:.2f})")
    axes[0].legend()

    _quartile_box(
        axes[1], df["築年数帯"],
        df["掲載期間25パーセンタイル"], df["掲載期間中央値"], df["掲載期間75パーセンタイル"],
        color="#4C72B0",
    )
    axes[1].set_ylabel("掲載期間(日) [Q1-中央値-Q3]")
    axes[1].set_title("築年数帯別 掲載期間(簡易箱ひげ図)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_age_vs_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[手順3-1] 築年数 × 掲載期間")
    print(f"  帯単位の参考ピアソン相関係数(代表年 vs 平均掲載期間): {r_pearson:.3f}")
    print("  ※ listing単位の生データが無いため、真の相関係数(全件ベース)とは異なる近似値。")


def step3_layout_vs_duration(layout: pd.DataFrame) -> None:
    df = layout.sort_values("掲載期間中央値")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(df["間取り"], df["掲載期間中央値"], color="#55A868")
    axes[0].set_xlabel("間取り")
    axes[0].set_ylabel("掲載期間中央値(日)")
    axes[0].set_title("間取り別 掲載期間中央値")

    _quartile_box(
        axes[1], df["間取り"],
        df["掲載期間25パーセンタイル"], df["掲載期間中央値"], df["掲載期間75パーセンタイル"],
        color="#55A868",
    )
    axes[1].set_ylabel("掲載期間(日) [Q1-中央値-Q3]")
    axes[1].set_title("間取り別 掲載期間(簡易箱ひげ図)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_layout_vs_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    table = df[["間取り", "件数", "掲載期間平均日数", "掲載期間中央値"]].sort_values("件数", ascending=False)
    print("\n[手順3-2] 間取り × 掲載期間")
    print(table.to_string(index=False))


def step4_representative_values(age: pd.DataFrame, layout: pd.DataFrame) -> str:
    lines = ["# 築年数・間取り分析レポート", ""]
    lines.append(
        "本レポートは `analysis_kokubunji/` 配下の集計済みCSV(帯別・カテゴリ別の件数と"
        "掲載期間統計量)を用いて作成した。物件1件ごとの生データ(bukken_rent TSV)は"
        "容量の都合でリポジトリに含まれていないため、ヒストグラムのビン幅は既存の"
        "帯区分に従い、散布図・相関係数は帯/カテゴリ単位の近似値である。"
    )
    lines.append("")

    lines.append("## 築年数")
    for _, row in age.sort_values("件数", ascending=False).iterrows():
        skew = "平均 > 中央値(右に歪み)" if row["掲載期間平均日数"] > row["掲載期間中央値"] else "平均 ≈ 中央値"
        lines.append(
            f"- {row['築年数帯']}: 件数{int(row['件数']):,} / "
            f"平均{row['掲載期間平均日数']:.1f}日 / 中央値{row['掲載期間中央値']:.1f}日 ({skew})"
        )
    lines.append("")
    lines.append(
        "築年数帯ごとの掲載期間は、どの帯も平均が中央値を上回っており、右に裾を引く"
        "分布(長期掲載の少数の物件に平均が引っ張られる形)になっていると考えられる。"
        "そのため代表値としては外れ値の影響を受けにくい**中央値**が妥当。"
        "築年数自体の代表値は、最も件数が多い帯(最頻値)で捉えるのが実務上分かりやすい。"
    )
    lines.append("")

    lines.append("## 間取り")
    for _, row in layout.sort_values("件数", ascending=False).iterrows():
        lines.append(
            f"- {row['間取り']}: 件数{int(row['件数']):,} / "
            f"平均{row['掲載期間平均日数']:.1f}日 / 中央値{row['掲載期間中央値']:.1f}日"
        )
    lines.append("")
    lines.append(
        "間取りはカテゴリ変数であるため、間取りそのものの代表値は平均・中央値ではなく"
        "**最頻値(最も件数の多いカテゴリ)**で示すべき。掲載期間については、間取りごとに"
        "平均が中央値より高い傾向が共通しており、こちらも中央値を代表値として使うのが妥当。"
    )
    lines.append("")

    lines.append("## 生データがない場合の限界")
    lines.append(
        "- ヒストグラムのビン幅を1年/5年刻みで検討する、という計画書の手順は"
        "listing単位の生データが必要であり、集計済みデータのみでは実行できない。"
        "既存の帯区分(不均等幅)をそのまま用いた。\n"
        "- 相関係数は帯/カテゴリの代表値(平均・中央値)同士から計算した参考値であり、"
        "listing単位のピアソン/スピアマン相関とは異なる(自由度がbin数しかないため"
        "解釈には注意が必要)。\n"
        "- 生データ(bukken_rent_v1.0.1.tsv)を用意できる場合は、"
        "listing単位の再分析を推奨する。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_data()
    step1_overview(data)
    step2_age_distribution(data["age"])
    step2_layout_distribution(data["layout"])
    step3_age_vs_duration(data["age"])
    step3_layout_vs_duration(data["layout"])
    report = step4_representative_values(data["age"], data["layout"])
    report_path = Path(__file__).resolve().parent / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nレポートを書き出しました: {report_path}")
    print(f"グラフを書き出しました: {OUT_DIR}")


if __name__ == "__main__":
    main()

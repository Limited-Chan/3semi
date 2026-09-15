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
# 各帯の下限・上限(年)。「築41年以上」は上限なしの帯のため、直前の帯幅(10年)を
# 目安に上限60年と仮定する(※この仮定は結果に注記する)。
AGE_BOUNDS = {
    "築5年以下": (0, 5),
    "築6〜10年": (5, 10),
    "築11〜20年": (10, 20),
    "築21〜30年": (20, 30),
    "築31〜40年": (30, 40),
    "築41年以上": (40, 60),
}
AGE_MIDPOINT = {band: (lo + hi) / 2 for band, (lo, hi) in AGE_BOUNDS.items()}
AGE_WIDTH = {band: hi - lo for band, (lo, hi) in AGE_BOUNDS.items()}


def load_data() -> dict[str, pd.DataFrame]:
    overview = pd.read_csv(DATA_DIR / "00_overview.csv", encoding="utf-8-sig")
    age = pd.read_csv(DATA_DIR / "06_building_age.csv", encoding="utf-8-sig")
    layout = pd.read_csv(DATA_DIR / "03_layout.csv", encoding="utf-8-sig")
    quality = pd.read_csv(DATA_DIR / "91_quality_report.csv", encoding="utf-8-sig")
    metadata = json.loads((DATA_DIR / "92_analysis_metadata.json").read_text(encoding="utf-8"))
    return {"overview": overview, "age": age, "layout": layout, "quality": quality, "metadata": metadata}


def step0_duration_overview(overview: pd.DataFrame) -> None:
    """計画書 0.前提「掲載期間そのものの分布確認」に対応。
    国分寺市全体(1行)の集計値のみ利用可能なため、平均・中央値・四分位範囲・
    歪度の目安(平均と中央値の差)を確認する。"""
    row = overview.iloc[0]
    mean, median = row["掲載期間平均日数"], row["掲載期間中央値"]
    q1, q3 = row["掲載期間25パーセンタイル"], row["掲載期間75パーセンタイル"]
    iqr = q3 - q1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    stats_list = [{"med": median, "q1": q1, "q3": q3, "whislo": q1, "whishi": q3, "fliers": []}]
    bxp_artists = ax.bxp(stats_list, vert=False, showfliers=False, patch_artist=True, positions=[0])
    bxp_artists["boxes"][0].set_facecolor("#8172B2")
    bxp_artists["boxes"][0].set_alpha(0.6)
    ax.axvline(mean, color="crimson", linewidth=1.5, label=f"平均 {mean:.1f}日")
    ax.set_yticks([])
    ax.set_xlabel("掲載期間(日)")
    ax.set_title(f"掲載期間の分布(国分寺市全体, n={int(row['件数']):,})")
    ax.legend(fontsize=9)
    fig.text(
        0.01, -0.05,
        f"中央値{median:.1f}日 / Q1={q1:.1f}日 / Q3={q3:.1f}日(IQR={iqr:.1f}日) / "
        f"7日以内終了率{row['7日以内終了率']:.1%} / 14日以内{row['14日以内終了率']:.1%} / 30日以内{row['30日以内終了率']:.1%}\n"
        "※ 都市全体1行の集計値のみのため、上記の箱(Q1-中央値-Q3)と平均線から分布形状を概観する。",
        fontsize=8, color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "00_duration_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=" * 60)
    print("手順0(前提): 掲載期間そのものの分布確認")
    print("=" * 60)
    print(f"  件数: {int(row['件数']):,}")
    print(f"  平均: {mean:.1f}日 / 中央値: {median:.1f}日 / Q1: {q1:.1f}日 / Q3: {q3:.1f}日 / IQR: {iqr:.1f}日")
    print(f"  平均が中央値の約{mean / median:.1f}倍 → 右に強く歪んだ分布(少数の長期掲載が平均を押し上げている)")
    print(f"  7日以内終了率: {row['7日以内終了率']:.1%} / 14日以内: {row['14日以内終了率']:.1%} / 30日以内: {row['30日以内終了率']:.1%}")


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


def grouped_age_stats(counts: pd.Series) -> dict[str, float]:
    """帯(不均等幅)ごとの件数から、区間内一様分布を仮定したグループ化データの
    平均・標準偏差・中央値・最頻値を求める(生データが無いための近似値)。"""
    bands = list(counts.index)
    n = counts.sum()

    mean = sum(counts[b] * AGE_MIDPOINT[b] for b in bands) / n
    variance = sum(
        counts[b] * ((AGE_MIDPOINT[b] - mean) ** 2 + AGE_WIDTH[b] ** 2 / 12)
        for b in bands
    ) / n
    std = variance ** 0.5

    cum = 0
    median = None
    for b in bands:
        lo, hi = AGE_BOUNDS[b]
        if cum + counts[b] >= n / 2:
            median = lo + (n / 2 - cum) / counts[b] * (hi - lo)
            break
        cum += counts[b]

    density = {b: counts[b] / AGE_WIDTH[b] for b in bands}
    modal_band = max(density, key=density.get)
    i = bands.index(modal_band)
    lo, hi = AGE_BOUNDS[modal_band]
    f1 = density[modal_band]
    f0 = density[bands[i - 1]] if i > 0 else 0
    f2 = density[bands[i + 1]] if i < len(bands) - 1 else 0
    denom = (f1 - f0) + (f1 - f2)
    mode = lo + (f1 - f0) / denom * (hi - lo) if denom else (lo + hi) / 2

    return {"mean": mean, "std": std, "median": median, "mode": mode, "modal_band": modal_band}


def step2_age_distribution(age: pd.DataFrame) -> dict[str, float]:
    df = age.set_index("築年数帯").loc[AGE_ORDER]
    counts = df["件数"]
    s = grouped_age_stats(counts)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    lefts = [AGE_BOUNDS[b][0] for b in AGE_ORDER]
    widths = [AGE_WIDTH[b] for b in AGE_ORDER]
    heights = [counts[b] / AGE_WIDTH[b] for b in AGE_ORDER]
    bars = ax.bar(lefts, heights, width=widths, align="edge", color="#4C72B0", edgecolor="white")
    for b, bar in zip(AGE_ORDER, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{counts[b]:,}件", ha="center", va="bottom", fontsize=8,
        )

    ax.axvspan(s["mean"] - s["std"], s["mean"] + s["std"], color="gray", alpha=0.15, label="平均±標準偏差")
    ax.axvline(s["mean"], color="crimson", linewidth=1.5, label=f"平均 {s['mean']:.1f}年")
    ax.axvline(s["median"], color="darkorange", linestyle="--", linewidth=1.5, label=f"中央値 {s['median']:.1f}年")
    ax.axvline(s["mode"], color="purple", linestyle=":", linewidth=1.5, label=f"最頻値(推定) {s['mode']:.1f}年")

    ax.set_xlabel("築年数(年)")
    ax.set_ylabel("度数密度(件/年)")
    ax.set_title("築年数の分布(帯の幅で正規化したヒストグラム)")
    ax.set_xticks([0, 5, 10, 20, 30, 40, 60])
    ax.legend(fontsize=9)
    fig.text(
        0.01, -0.04,
        f"標準偏差: {s['std']:.1f}年(帯内は一様分布と仮定した近似値)\n"
        "※ 生データが無いため、帯別件数から算出したグループ化統計量。既存の帯区分(不均等幅)をそのまま採用し、"
        "高さは件数/帯幅(度数密度)で正規化。「築41年以上」は上限を60年と仮定。",
        fontsize=8, color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_age_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[手順2-1] 築年数分布")
    print(f"  平均: {s['mean']:.1f}年 / 中央値: {s['median']:.1f}年 / "
          f"最頻値(推定): {s['mode']:.1f}年(最多密度の帯: {s['modal_band']}) / 標準偏差: {s['std']:.1f}年")
    return s


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


def step4_representative_values(age: pd.DataFrame, layout: pd.DataFrame, age_stats: dict[str, float]) -> str:
    lines = ["# 築年数・間取り分析レポート", ""]
    lines.append(
        "本レポートは `analysis_kokubunji/` 配下の集計済みCSV(帯別・カテゴリ別の件数と"
        "掲載期間統計量)を用いて作成した。物件1件ごとの生データ(bukken_rent TSV)は"
        "容量の都合でリポジトリに含まれていないため、ヒストグラムのビン幅は既存の"
        "帯区分に従い、散布図・相関係数は帯/カテゴリ単位の近似値である。"
    )
    lines.append("")

    lines.append("## 築年数")
    lines.append(
        f"グループ化統計量(近似): 平均{age_stats['mean']:.1f}年 / 中央値{age_stats['median']:.1f}年 / "
        f"最頻値(推定){age_stats['mode']:.1f}年 / 標準偏差{age_stats['std']:.1f}年"
    )
    lines.append("")
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
        "そのため代表値としては外れ値の影響を受けにくい**中央値**が妥当。\n\n"
        f"築年数自体の分布も、平均({age_stats['mean']:.1f}年)が中央値({age_stats['median']:.1f}年)"
        f"よりやや大きく、右に緩やかに裾を引く形になっている。単純な件数最多の帯は「築11〜20年」だが、"
        f"帯の幅(10年)を考慮した度数密度でみると最も密度が高いのは「築5年以下」の帯であり、"
        f"最頻値は約{age_stats['mode']:.1f}年と推定される。したがって築年数の代表値も、"
        "件数最多の帯だけで判断せず中央値・最頻値をあわせて確認するのが妥当。"
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
    step0_duration_overview(data["overview"])
    step1_overview(data)
    age_stats = step2_age_distribution(data["age"])
    step2_layout_distribution(data["layout"])
    step3_age_vs_duration(data["age"])
    step3_layout_vs_duration(data["layout"])
    report = step4_representative_values(data["age"], data["layout"], age_stats)
    report_path = Path(__file__).resolve().parent / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nレポートを書き出しました: {report_path}")
    print(f"グラフを書き出しました: {OUT_DIR}")


if __name__ == "__main__":
    main()

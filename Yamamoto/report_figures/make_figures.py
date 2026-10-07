#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「国分寺市の賃貸物件における掲載終了リスクと問い合わせ優先度の分析」報告書用の図を作る。

図と報告書の節の対応:
    図1  掲載期間の分布 ............................ 3節
    図2  属性と掲載期間の相関 ...................... 4〜5節
    図3  築年数帯別の掲載期間中央値 ................ 4節
    図4  属性2つの組み合わせのR^2 .................. 6節
    図5  Kaplan-Meier(全体) ........................ 9節
    図6  Kaplan-Meier(築年数帯別・間取り別) ........ 9節
    図7  学習・検証・最終評価の時間分割 ............ 13節
    図8  Brier scoreによるモデル比較 ............... 15〜16節
    図9  問い合わせ優先度に使う予測期間の比較 ....... 17.1節
    図10 優先度別の実際の掲載終了率 ................ 17.2〜17.3節

データの出どころ:
    図1,3,5,6 ... 生データ(国分寺市, 2020〜2023年, 22,332件)から Ohori/src の
                   前処理・生存分析用データ作成関数を使って計算する。
    図2,4,8,9,10 ... 報告書の表の数値を REPORT に転記して描画する
                   (モデル学習を伴うため、この図スクリプトでは再計算しない)。
    図7 ......... 概念図(データなし)。

実行:
    python3 make_figures.py [--input path/to/kokubunji_raw.parquet]
    (既定の入力は リポジトリ直下の data/prepared/kokubunji_raw.parquet)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parents[1]
OHORI_DIR = REPO_DIR / "Ohori"
sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared, select_eda_scope  # noqa: E402
from src.survival import prepare_survival_data  # noqa: E402

_JP_FONTS = ["Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "MS Gothic"]
_available = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in _JP_FONTS if f in _available), "sans-serif")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e3e3e3"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["figure.dpi"] = 100

TEAL = "#1b7f79"
GRAY = "#b4b9bf"
DARK = "#3b4252"
RED = "#d1495b"
TIER_COLORS = {"低": "#4c78a8", "中": "#f2b134", "高": "#d1495b"}

# ---------------------------------------------------------------------------
# 報告書の表から転記した数値
# ---------------------------------------------------------------------------
REPORT = {
    "n": 22332,
    "duration_stats": {"mean": 50.17, "median": 22, "mode": 0, "q1": 6, "q3": 60},
    "events": 22157,
    "censored": 175,
    # 4節
    "corr": {  # 属性: (Spearman, Pearson)
        "築年数": (0.311, 0.241),
        "家賃": (-0.228, -0.087),
        "専有面積": (-0.123, -0.086),
        "駅までの距離": (-0.016, 0.010),
    },
    "layout_eta": 0.162,
    "age_band_median": {"0～5年": 13, "5～10年": 13, "10～15年": 8, "15～20年": 17, "20～25年": 24, "25～30年": 41},
    # 6節
    "pair_r2": [
        ("築年数 + 間取り", 0.071341),
        ("築年数 + 専有面積", 0.063840),
        ("築年数 + 家賃", 0.059608),
        ("家賃 + 間取り", 0.049602),
        ("専有面積 + 間取り", 0.038055),
        ("家賃 + 専有面積", 0.008113),
    ],
    # 9節 Kaplan-Meier: 経過日数 -> まだ掲載されている割合(%)
    "km_checkpoints": {7: 70.3, 14: 58.5, 30: 42.1, 60: 24.7, 90: 15.0, 180: 5.4},
    "km_age_median": {"0～10年": 13, "10～20年": 13, "20～30年": 33, "30～40年": 42},
    "km_layout_median": {"1LDK": 14, "1K": 19, "1R": 40},
    # 15節 平均Brier score(小さいほど良い)
    "brier_2022": [
        ("Model 2\n(最終モデル)", 0.183133),
        ("Model 1", 0.183956),
        ("築年数帯＋間取り", 0.185960),
        ("全物件共通", 0.192060),
    ],
    # 16節
    "brier_2023": [
        ("Model 2\n(最終モデル)", 0.181847),
        ("築年数帯＋間取り", 0.185959),
        ("全物件共通", 0.189372),
    ],
    # 17.1節 予測期間の比較(2022年)
    "horizon": {
        7: {"auc": 0.617768, "diff": 0.119870},
        14: {"auc": 0.629613, "diff": 0.205066},
        30: {"auc": 0.652643, "diff": 0.315707},
    },
    # 17.2節・17.3節 優先度別の実際の30日以内終了率(%) 評価時点(掲載日目) -> (低, 中, 高)
    "tier_rates": {
        2022: {7: (33.6, 49.9, 63.2), 14: (29.5, 48.7, 61.7), 30: (29.2, 45.3, 62.1)},
        2023: {7: (34.7, 49.5, 61.9), 14: (34.8, 49.1, 59.3), 30: (33.2, 47.1, 56.3)},
    },
    "auc_2023": {7: 0.652, 14: 0.642, 30: 0.633},
    "thresholds": (35.71, 51.66),
}

verification_rows: list[tuple[str, str, str, bool]] = []


def check(label: str, reported, computed, tol: float = 0.0) -> None:
    ok = abs(float(reported) - float(computed)) <= tol
    verification_rows.append((label, f"{reported}", f"{computed:.4g}" if isinstance(computed, float) else f"{computed}", ok))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  保存: {name}")


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """日単位の生存時間と右打ち切りを使ったKaplan-Meier推定。戻り値は (日, まだ掲載されている割合)。"""
    t = np.asarray(durations, dtype=int)
    e = np.asarray(events, dtype=int)
    n_days = int(t.max()) + 1
    event_count = np.bincount(t[e == 1], minlength=n_days)
    leave_count = np.bincount(t, minlength=n_days)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(leave_count)[:-1]])
    hazard = np.divide(event_count, at_risk, out=np.zeros(n_days), where=at_risk > 0)
    return np.arange(n_days), np.cumprod(1 - hazard)


def km_at(days: np.ndarray, surv: np.ndarray, day: int) -> float:
    return float(surv[min(day, len(surv) - 1)])


def km_median(days: np.ndarray, surv: np.ndarray) -> float:
    below = np.where(surv <= 0.5)[0]
    return float(days[below[0]]) if len(below) else float("nan")


# ---------------------------------------------------------------------------
# 図1 掲載期間の分布(3節)
# ---------------------------------------------------------------------------
def fig01_duration_distribution(eda: pd.DataFrame) -> None:
    x = eda["duration"].to_numpy(dtype=float)
    stats = {"mean": x.mean(), "median": float(np.median(x)), "mode": float(pd.Series(x).mode().iloc[0]),
             "q1": float(np.quantile(x, 0.25)), "q3": float(np.quantile(x, 0.75))}
    check("件数", REPORT["n"], len(x))
    check("平均(日)", REPORT["duration_stats"]["mean"], round(stats["mean"], 2), 0.005)
    for key, label in [("median", "中央値"), ("mode", "最頻値"), ("q1", "下位25%"), ("q3", "上位25%")]:
        check(f"{label}(日)", REPORT["duration_stats"][key], stats[key])

    display_max = 360
    shown = x[x <= display_max]
    beyond = int((x > display_max).sum())

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.hist(shown, bins=np.arange(0, display_max + 10, 10), color="#8aa9c9", edgecolor="white", linewidth=0.6)
    ax.axvspan(stats["q1"], stats["q3"], color=TEAL, alpha=0.10, label=f"中央の50%の範囲 ({stats['q1']:.0f}〜{stats['q3']:.0f}日)")
    ax.axvline(stats["median"], color=TEAL, lw=2.2, label=f"中央値 {stats['median']:.0f}日")
    ax.axvline(stats["mean"], color=RED, lw=2.2, ls="--", label=f"平均 {stats['mean']:.2f}日")
    ax.axvline(stats["mode"], color=DARK, lw=1.6, ls=":", label=f"最頻値 {stats['mode']:.0f}日")
    ax.set_xlabel("掲載期間(日, 10日刻み)")
    ax.set_ylabel("物件数")
    ax.set_xlim(0, display_max)
    ax.set_title("図1  掲載期間は短期に集中し、一部の長期掲載が平均を押し上げている", loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.annotate(f"{display_max}日超の{beyond:,}件({beyond / len(x):.1%})は表示範囲外", xy=(display_max, 0), xytext=(-8, 70),
                textcoords="offset points", ha="right", fontsize=9, color="#555")
    fig.text(0.01, -0.03, f"対象: 東京都国分寺市, 2020〜2023年掲載開始 {len(x):,}件。掲載期間 = 掲載終了日 − 掲載開始日。", fontsize=9, color="#666")
    save(fig, "fig01_duration_distribution.png")


# ---------------------------------------------------------------------------
# 図2 属性と掲載期間の相関(4〜5節)
# ---------------------------------------------------------------------------
def fig02_attribute_correlation() -> None:
    names = list(REPORT["corr"].keys())
    spearman = [REPORT["corr"][k][0] for k in names]
    pearson = [REPORT["corr"][k][1] for k in names]
    y = np.arange(len(names))[::-1]
    h = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    b1 = ax.barh(y + h / 2, spearman, height=h, color=TEAL, label="Spearman順位相関")
    b2 = ax.barh(y - h / 2, pearson, height=h, color="#9aa5b1", label="Pearson相関")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_width()
            ax.text(v + (0.008 if v >= 0 else -0.008), bar.get_y() + bar.get_height() / 2, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(-0.35, 0.40)
    ax.set_xlabel("掲載期間との相関係数(+: 値が大きいほど掲載期間が長い)")
    ax.set_title("図2  築年数が最も強く、駅までの距離はほぼ無関係", loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.text(0.01, -0.04, f"間取り(カテゴリ)は相関係数を使えないため、別指標の相関比で確認: η = {REPORT['layout_eta']:.3f}(0〜1, 大きいほど間取りによる違いが大きい)。",
             fontsize=9, color="#666")
    save(fig, "fig02_attribute_correlation.png")


# ---------------------------------------------------------------------------
# 図3 築年数帯別の掲載期間中央値(4節)
# ---------------------------------------------------------------------------
def fig03_age_band_median(eda: pd.DataFrame) -> None:
    d = eda[eda["age"].notna()].copy()
    edges = np.arange(0, 65, 5)
    labels = [f"{a}～{b}年" for a, b in zip(edges[:-1], edges[1:])]
    d["band"] = pd.cut(d["age"], bins=edges, labels=labels, right=False, include_lowest=True)
    g = d.groupby("band", observed=False)["duration"].agg(["median", "count"]).reindex(labels)
    for band, reported in REPORT["age_band_median"].items():
        check(f"築年数{band} 中央値(日)", reported, float(g.loc[band, "median"]))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = [TEAL if c >= 30 else GRAY for c in g["count"]]
    bars = ax.bar(range(len(g)), g["median"], color=colors, width=0.72)
    for i, (m, c) in enumerate(zip(g["median"], g["count"])):
        ax.text(i, m + 0.8, f"{m:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{l}\nn={int(c):,}" for l, c in zip(labels, g["count"])], fontsize=8.5)
    ax.set_ylabel("掲載期間の中央値(日)")
    ax.set_title("図3  築20年を超えるあたりから掲載期間の中央値が伸び、築25年以降は40日前後で推移する", loc="left", fontsize=13, fontweight="bold")
    ax.legend(handles=[Patch(color=TEAL, label="n 30以上"), Patch(color=GRAY, label="n 30未満(参考値)")], frameon=False, loc="upper left")
    fig.text(0.01, -0.04, "築年数 = 掲載開始月 − 建築年月(年換算)。築年数が不明な物件(約4%)は除く。", fontsize=9, color="#666")
    save(fig, "fig03_age_band_median.png")


# ---------------------------------------------------------------------------
# 図4 属性2つの組み合わせのR^2(6節)
# ---------------------------------------------------------------------------
def fig04_pair_r2() -> None:
    pairs = REPORT["pair_r2"][::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = [TEAL if i == len(pairs) - 1 else GRAY for i in range(len(pairs))]
    bars = ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=colors, height=0.62)
    for bar, (_, v) in zip(bars, pairs):
        ax.text(v + 0.0012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 0.09)
    ax.set_xlabel("R²(掲載期間のばらつきのうち、2つの属性で説明できる割合)")
    ax.set_title("図4  最も説明力の高い築年数＋間取りでも、R² は約7%にとどまる", loc="left", fontsize=13, fontweight="bold")
    save(fig, "fig04_pair_r2.png")


# ---------------------------------------------------------------------------
# 図5 Kaplan-Meier 全体(9節)
# ---------------------------------------------------------------------------
def fig05_km_overall(surv_df: pd.DataFrame) -> None:
    days, surv = kaplan_meier(surv_df["survival_days"].to_numpy(), surv_df["event"].to_numpy())
    check("掲載終了イベント数", REPORT["events"], int(surv_df["event"].sum()))
    check("右打ち切り数", REPORT["censored"], int((surv_df["event"] == 0).sum()))
    for day, pct in REPORT["km_checkpoints"].items():
        check(f"{day}日後に掲載継続(%)", pct, round(km_at(days, surv, day) * 100, 1), 0.05)

    shown = days <= 180
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.step(days[shown], surv[shown] * 100, where="post", color=TEAL, lw=2.4)
    ax.fill_between(days[shown], 0, surv[shown] * 100, step="post", color=TEAL, alpha=0.08)
    offsets = {7: (10, 4), 14: (10, 6), 30: (10, 6), 60: (10, 8), 90: (10, 8), 180: (-12, 14)}
    for day in REPORT["km_checkpoints"]:
        v = km_at(days, surv, day) * 100
        ax.plot([day], [v], "o", color=RED, ms=7, zorder=5)
        ax.annotate(f"{day}日後: {v:.1f}%", (day, v), textcoords="offset points", xytext=offsets[day],
                    fontsize=10, fontweight="bold", color=DARK, ha="left" if offsets[day][0] > 0 else "right")
    ax.axhline(50, color=GRAY, lw=1, ls="--")
    med = km_median(days, surv)
    ax.text(178, 52, f"50%線(中央値 {med:.0f}日)", ha="right", fontsize=9, color="#666")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 102)
    ax.set_xlabel("掲載開始からの経過日数")
    ax.set_ylabel("まだ掲載されている物件の割合(%)")
    ax.set_title("図5  掲載開始30日で約58%、90日で約85%の物件が掲載を終了する", loc="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.04, f"Kaplan–Meier法。右打ち切り{REPORT['censored']}件(2024-03-31時点で掲載継続)を含む{len(surv_df):,}件。表示は180日まで(計算は全期間)。",
             fontsize=9, color="#666")
    save(fig, "fig05_km_overall.png")


# ---------------------------------------------------------------------------
# 図6 Kaplan-Meier 築年数帯別・間取り別(9節)
# ---------------------------------------------------------------------------
def fig06_km_by_group(surv_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

    # 築年数帯別(順序あり → 明→暗の連続色)
    d = surv_df[surv_df["age"].notna()].copy()
    age_labels = ["0～10年", "10～20年", "20～30年", "30～40年", "40年以上"]
    d["g"] = pd.cut(d["age"], bins=[0, 10, 20, 30, 40, np.inf], labels=age_labels, right=False, include_lowest=True)
    cmap = plt.get_cmap("YlGnBu")
    ax = axes[0]
    for i, lab in enumerate(age_labels):
        sub = d[d["g"] == lab]
        days, surv = kaplan_meier(sub["survival_days"].to_numpy(), sub["event"].to_numpy())
        m = km_median(days, surv)
        if lab in REPORT["km_age_median"]:
            check(f"築年数{lab} 掲載継続の中央値(日)", REPORT["km_age_median"][lab], m)
        shown = days <= 180
        ax.step(days[shown], surv[shown] * 100, where="post", lw=2.2, color=cmap(0.3 + 0.65 * i / 4),
                label=f"{lab}  (n={len(sub):,}, 中央値{m:.0f}日)")
    ax.axhline(50, color=GRAY, lw=1, ls="--")
    ax.set_title("築年数帯別(濃いほど築古)", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("まだ掲載されている物件の割合(%)")

    # 間取り別(順序なし → 区別しやすい色)
    d2 = surv_df[surv_df["layout"].notna()].copy()
    top = d2["layout"].value_counts().head(6).index.tolist()
    ax2 = axes[1]
    qual = plt.get_cmap("tab10")
    for i, lab in enumerate(top):
        sub = d2[d2["layout"] == lab]
        days, surv = kaplan_meier(sub["survival_days"].to_numpy(), sub["event"].to_numpy())
        m = km_median(days, surv)
        if lab in REPORT["km_layout_median"]:
            check(f"間取り{lab} 掲載継続の中央値(日)", REPORT["km_layout_median"][lab], m)
        shown = days <= 180
        ax2.step(days[shown], surv[shown] * 100, where="post", lw=2.2, color=qual(i),
                 label=f"{lab}  (n={len(sub):,}, 中央値{m:.0f}日)")
    ax2.axhline(50, color=GRAY, lw=1, ls="--")
    ax2.set_title("間取り別(件数上位6種)", loc="left", fontsize=12, fontweight="bold")

    for a in axes:
        a.set_xlim(0, 180)
        a.set_ylim(0, 102)
        a.set_xlabel("掲載開始からの経過日数")
        a.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("図6  築年数が古いほど、また1Rのような間取りほど、掲載が長く続く", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig06_km_by_age_layout.png")


# ---------------------------------------------------------------------------
# 図7 時間分割(13節)
# ---------------------------------------------------------------------------
def fig07_time_split() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(2017.8, 2024.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.grid(False)

    def block(x0, x1, y, text, face, edge="white", hatch=None, color="white", fs=10):
        ax.add_patch(Rectangle((x0, y), x1 - x0, 0.8, facecolor=face, edgecolor=edge, hatch=hatch, lw=1.2))
        ax.text((x0 + x1) / 2, y + 0.4, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold")

    grey = "#d9dce1"
    block(2018, 2020, 3.0, "対象外\n(収録期間の開始側)", grey, hatch="///", color="#555", fs=9)
    block(2020, 2022, 3.0, "学習  2020〜2021年", "#7aa6cf")
    block(2022, 2023, 3.0, "検証  2022年", "#3f78b0")
    block(2023, 2024, 3.0, "最終確認  2023年", "#1f4e79")
    block(2024, 2024.25, 3.0, "対象外", grey, hatch="///", color="#555", fs=8)

    ax.text(2021, 2.65, "① モデル学習", ha="center", va="top", fontsize=10, color=DARK)
    ax.text(2022.5, 2.65, "② モデル選択と\n予測期間・高/中/低の\n境界を決める", ha="center", va="top", fontsize=9.5, color=DARK)
    ax.text(2023.5, 2.65, "④ 決めたルールを\n固定して適用", ha="center", va="top", fontsize=9.5, color=DARK)

    ax.annotate("", xy=(2023.0, 1.55), xytext=(2020.0, 1.55), arrowprops=dict(arrowstyle="<->", color=DARK))
    ax.text(2021.5, 1.7, "③ 2020〜2022年(16,030件)で再学習", ha="center", fontsize=10, color=DARK)
    ax.annotate("", xy=(2024.0, 1.55), xytext=(2023.0, 1.55), arrowprops=dict(arrowstyle="->", color=DARK))
    ax.text(2023.5, 1.12, "2023年 6,302件で評価", ha="center", fontsize=10, color=DARK)

    ax.text(2018, 0.45, "2024年は3月31日までしかデータがなく、年間を通した観測ではないため主分析から除外。\n2018〜2019年はデータ収集期間の開始側で、掲載期間中央値が2018年879日・2019年267日と大きく異なるため除外。",
            fontsize=9, color="#555", va="top")
    ax.set_title("図7  将来のデータを見ずにルールを決めるため、年単位で時間順に分割した", loc="left", fontsize=13, fontweight="bold")
    save(fig, "fig07_time_split.png")


# ---------------------------------------------------------------------------
# 図8 Brier score(15〜16節)
# ---------------------------------------------------------------------------
def fig08_brier() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, key, title in [(axes[0], "brier_2022", "2022年(検証データ)"), (axes[1], "brier_2023", "2023年(最終確認)")]:
        rows = REPORT[key][::-1]
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        y = np.arange(len(rows))
        best = int(np.argmin(vals))
        ax.hlines(y, 0.178, vals, color="#d5d9de", lw=2)
        ax.scatter(vals, y, s=110, color=[TEAL if i == best else GRAY for i in range(len(vals))], zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.0003, yi + 0.18, f"{v:.4f}", fontsize=10, ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("平均Brier score(小さいほど確率予測の誤差が小さい)")
        ax.set_xlim(0.179, 0.196)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    fig.suptitle("図8  Model 2 が検証でも最終確認でも最も誤差が小さい(ただし差は小幅)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, -0.04, "横軸は0から始めていない(差を見やすくするため)。評価は9通りの条件の平均。2023年はModel 2が9条件すべてで比較対象より小さい誤差。", fontsize=9, color="#666")
    save(fig, "fig08_brier_scores.png")


# ---------------------------------------------------------------------------
# 図9 予測期間の比較(17.1節)
# ---------------------------------------------------------------------------
def fig09_horizon() -> None:
    horizons = [7, 14, 30]
    labels = ["7日以内", "14日以内", "30日以内"]
    auc = [REPORT["horizon"][h]["auc"] for h in horizons]
    diff = [REPORT["horizon"][h]["diff"] * 100 for h in horizons]
    colors = [GRAY, GRAY, TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    ax.bar(labels, auc, color=colors, width=0.6)
    for i, v in enumerate(auc):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 0.68)
    ax.axhline(0.5, color=DARK, lw=1)
    ax.set_ylabel("平均AUC")
    ax.set_title("順位付け性能(AUC)", loc="left", fontsize=12, fontweight="bold")

    ax = axes[1]
    ax.bar(labels, diff, color=colors, width=0.6)
    for i, v in enumerate(diff):
        ax.text(i, v + 0.6, f"{v:.1f}pt", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.set_ylabel("高群−低群の差(ポイント)")
    ax.set_title("高群と低群の実際の掲載終了率の差", loc="left", fontsize=12, fontweight="bold")

    fig.suptitle("図9  予測期間は30日以内が最も順位付け性能が高く、群間差も最大", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, -0.06,
             "2022年の検証データでの比較。AUCは1に近いほど「終了しやすい物件を上位に並べられる」ことを表し、0.5は順位付けがランダムと同程度(左図は0.5を基準線とした)。\n"
             "いずれの期間も、3つの評価時点すべてで 低 < 中 < 高 の順に実際の終了率が上がったため、AUCと群間差で30日以内を選択。",
             fontsize=9, color="#666")
    save(fig, "fig09_horizon_selection.png")


# ---------------------------------------------------------------------------
# 図10 優先度別の実際の掲載終了率(17.2〜17.3節)
# ---------------------------------------------------------------------------
def fig10_tier_rates() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    tiers = ["低", "中", "高"]
    eval_days = [7, 14, 30]
    width = 0.26
    for ax, year, title in [(axes[0], 2022, "2022年(境界を決めたデータ)"), (axes[1], 2023, "2023年(境界を固定して適用)")]:
        for j, tier in enumerate(tiers):
            vals = [REPORT["tier_rates"][year][d][j] for d in eval_days]
            xs = np.arange(len(eval_days)) + (j - 1) * width
            bars = ax.bar(xs, vals, width=width * 0.94, color=TIER_COLORS[tier], label=f"優先度「{tier}」")
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.9, f"{v:.1f}", ha="center", fontsize=9.5)
        for i, d in enumerate(eval_days):
            low, _, high = REPORT["tier_rates"][year][d]
            note = f"高−低 +{high - low:.1f}pt"
            if year == 2023:
                note += f"\nAUC {REPORT['auc_2023'][d]:.3f}"
            ax.text(i, 98, note, ha="center", va="top", fontsize=10, fontweight="bold", color=DARK, linespacing=1.4)
        ax.set_xticks(range(len(eval_days)))
        ax.set_xticklabels([f"掲載{d}日目" for d in eval_days], fontsize=10.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("リスクを評価した時点")
    axes[0].set_ylabel("その後30日以内に実際に掲載終了した割合(%)")
    axes[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=10, bbox_to_anchor=(0.0, 0.82))
    lo, hi = REPORT["thresholds"]
    fig.suptitle("図10  高と判定した物件ほど実際に掲載終了しやすい(2023年でも順序が再現)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, -0.05,
             f"優先度は、モデルが出す今後30日以内の掲載終了リスクで分類: 低 {lo}%未満 / 中 {lo}%以上{hi}%未満 / 高 {hi}%以上(境界は2022年のリスク分布の3分位点で決め、2023年へ固定適用)。\n"
             "優先度は掲載終了を確実に予言するものではなく、候補物件同士の相対的なリスクを比べるための目安。掲載終了は契約成立を意味しない。",
             fontsize=9, color="#666")
    save(fig, "fig10_priority_tier_end_rates.png")


def print_verification() -> None:
    print("\n=== 報告書の数値との照合(生データから再計算できる項目) ===")
    width = max(len(r[0]) for r in verification_rows)
    for label, rep, comp, ok in verification_rows:
        print(f"  [{'OK' if ok else '不一致'}] {label:<{width}}  報告書={rep:<8} 再計算={comp}")
    bad = [r for r in verification_rows if not r[3]]
    print(f"  → {len(verification_rows) - len(bad)}/{len(verification_rows)} 項目が一致" + (f"(不一致 {len(bad)} 項目)" if bad else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_DIR / "data" / "prepared" / "kokubunji_raw.parquet"))
    args = parser.parse_args()

    d = load_prepared(args.input)
    eda = select_eda_scope(d, years=[2020, 2021, 2022, 2023], keep_duplicates=True)
    surv_df = prepare_survival_data(d)
    print(f"EDA対象 {len(eda):,}件 / 生存分析対象 {len(surv_df):,}件")

    fig01_duration_distribution(eda)
    fig02_attribute_correlation()
    fig03_age_band_median(eda)
    fig04_pair_r2()
    fig05_km_overall(surv_df)
    fig06_km_by_group(surv_df)
    fig07_time_split()
    fig08_brier()
    fig09_horizon()
    fig10_tier_rates()
    print_verification()


if __name__ == "__main__":
    main()

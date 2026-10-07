from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# ============================================================
# パス設定
# ============================================================

OHORI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = OHORI_DIR.parent

if str(OHORI_DIR) not in sys.path:
    sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared
from src.survival import prepare_survival_data


INPUT_PATH = (
    REPO_DIR
    / "data"
    / "prepared"
    / "kokubunji_raw.parquet"
)

OUTPUT_DIR = (
    OHORI_DIR
    / "results"
    / "03_kaplan_meier"
)


# ============================================================
# 表示設定
# ============================================================

plt.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "DejaVu Sans",
]

plt.rcParams["axes.unicode_minus"] = False


# Kaplan-Meier図では、
# 長期掲載のごく少数の物件によって横軸が極端に伸びないよう、
# 発表用表示は180日までとする。
# 計算自体は全期間を使用する。
DISPLAY_MAX_DAYS = 180


# 比較対象とする主要間取り数
TOP_LAYOUT_COUNT = 6


# ============================================================
# 補助関数
# ============================================================

def save_csv(
    df: pd.DataFrame,
    filename: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def save_figure(
    fig,
    filename: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 1. 全物件のKaplan-Meier曲線
# ============================================================

def overall_kaplan_meier(
    df: pd.DataFrame,
) -> None:

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=df["survival_days"],
        event_observed=df["event"],
        label="全物件",
    )

    # ----------------------------------------
    # 生存曲線の数値をCSV保存
    # ----------------------------------------

    survival = (
        kmf.survival_function_
        .reset_index()
    )

    survival.columns = [
        "掲載開始からの日数",
        "掲載継続確率",
    ]

    save_csv(
        survival,
        "overall_survival_curve.csv",
    )

    # ----------------------------------------
    # 主要日数での掲載継続確率
    # ----------------------------------------

    checkpoints = [
        7,
        14,
        30,
        60,
        90,
        180,
    ]

    rows = []

    for day in checkpoints:

        survival_probability = float(
            kmf.predict(day)
        )

        rows.append(
            {
                "経過日数": day,
                "掲載継続確率": survival_probability,
                "掲載終了済み割合": 1 - survival_probability,
            }
        )

    checkpoint_table = pd.DataFrame(
        rows
    )

    save_csv(
        checkpoint_table,
        "overall_checkpoints.csv",
    )

    print()
    print("【全物件 Kaplan-Meier】")

    print(
        checkpoint_table.to_string(
            index=False
        )
    )

    print()

    print(
        f"Kaplan-Meier中央値: {kmf.median_survival_time_:.1f} 日"
    )

    # ----------------------------------------
    # グラフ
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
    )

    ax.set_xlim(
        0,
        DISPLAY_MAX_DAYS,
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_xlabel(
        "掲載開始からの日数"
    )

    ax.set_ylabel(
        "掲載が継続している確率"
    )

    ax.set_title(
        "掲載期間のKaplan–Meier曲線"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "overall_kaplan_meier.png",
    )


# ============================================================
# 2. 築年数帯ごとの比較
# ============================================================

def age_group_kaplan_meier(
    df: pd.DataFrame,
) -> None:

    sub = (
        df[
            df["age"].notna()
        ]
        .copy()
    )

    bins = [
        0,
        10,
        20,
        30,
        40,
        np.inf,
    ]

    labels = [
        "0～10年",
        "10～20年",
        "20～30年",
        "30～40年",
        "40年以上",
    ]

    sub["age_group"] = pd.cut(
        sub["age"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    summary_rows = []

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for group in labels:

        group_df = sub[
            sub["age_group"]
            == group
        ]

        if group_df.empty:
            continue

        kmf = KaplanMeierFitter()

        kmf.fit(
            durations=group_df["survival_days"],
            event_observed=group_df["event"],
            label=group,
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
        )

        summary_rows.append(
            {
                "築年数帯": group,
                "件数": len(group_df),
                "イベント数": int(
                    group_df["event"].sum()
                ),
                "右打ち切り数": int(
                    (group_df["event"] == 0).sum()
                ),
                "Kaplan-Meier中央値": float(
                    kmf.median_survival_time_
                ),
                "14日後掲載継続確率": float(
                    kmf.predict(14)
                ),
                "30日後掲載継続確率": float(
                    kmf.predict(30)
                ),
                "60日後掲載継続確率": float(
                    kmf.predict(60)
                ),
            }
        )

    ax.set_xlim(
        0,
        DISPLAY_MAX_DAYS,
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_xlabel(
        "掲載開始からの日数"
    )

    ax.set_ylabel(
        "掲載が継続している確率"
    )

    ax.set_title(
        "築年数帯別のKaplan–Meier曲線"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="築年数帯"
    )

    save_figure(
        fig,
        "age_group_kaplan_meier.png",
    )

    summary = pd.DataFrame(
        summary_rows
    )

    save_csv(
        summary,
        "age_group_summary.csv",
    )

    print()
    print("【築年数帯別 Kaplan-Meier】")

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# 3. 間取りごとの比較
# ============================================================

def layout_kaplan_meier(
    df: pd.DataFrame,
) -> None:

    sub = (
        df[
            df["layout"].notna()
        ]
        .copy()
    )

    top_layouts = (
        sub["layout"]
        .value_counts()
        .head(TOP_LAYOUT_COUNT)
        .index
        .tolist()
    )

    summary_rows = []

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for layout in top_layouts:

        group_df = sub[
            sub["layout"]
            == layout
        ]

        kmf = KaplanMeierFitter()

        kmf.fit(
            durations=group_df["survival_days"],
            event_observed=group_df["event"],
            label=layout,
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
        )

        summary_rows.append(
            {
                "間取り": layout,
                "件数": len(group_df),
                "イベント数": int(
                    group_df["event"].sum()
                ),
                "右打ち切り数": int(
                    (group_df["event"] == 0).sum()
                ),
                "Kaplan-Meier中央値": float(
                    kmf.median_survival_time_
                ),
                "14日後掲載継続確率": float(
                    kmf.predict(14)
                ),
                "30日後掲載継続確率": float(
                    kmf.predict(30)
                ),
                "60日後掲載継続確率": float(
                    kmf.predict(60)
                ),
            }
        )

    ax.set_xlim(
        0,
        DISPLAY_MAX_DAYS,
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_xlabel(
        "掲載開始からの日数"
    )

    ax.set_ylabel(
        "掲載が継続している確率"
    )

    ax.set_title(
        "間取り別のKaplan–Meier曲線"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="間取り"
    )

    save_figure(
        fig,
        "layout_kaplan_meier.png",
    )

    summary = pd.DataFrame(
        summary_rows
    )

    save_csv(
        summary,
        "layout_summary.csv",
    )

    print()
    print("【間取り別 Kaplan-Meier】")

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# main
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "国分寺市 賃貸掲載データ Kaplan-Meier分析"
    )

    print(
        "=" * 70
    )

    df = load_prepared(
        INPUT_PATH
    )

    survival_df = prepare_survival_data(
        df
    )

    print(
        f"対象件数       : {len(survival_df):,}"
    )

    print(
        f"イベント       : {survival_df['event'].sum():,}"
    )

    print(
        f"右打ち切り     : {(survival_df['event'] == 0).sum():,}"
    )

    overall_kaplan_meier(
        survival_df
    )

    age_group_kaplan_meier(
        survival_df
    )

    layout_kaplan_meier(
        survival_df
    )

    print()
    print(
        f"出力先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
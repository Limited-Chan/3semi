from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# パス設定
# ============================================================

OHORI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = OHORI_DIR.parent

if str(OHORI_DIR) not in sys.path:
    sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared, select_eda_scope


DEFAULT_INPUT = (
    REPO_DIR
    / "data"
    / "prepared"
    / "kokubunji_raw.parquet"
)

DEFAULT_OUTPUT = (
    OHORI_DIR
    / "results"
    / "01_eda"
)


# ============================================================
# 分析条件
# ============================================================

YEARS = [
    2020,
    2021,
    2022,
    2023,
]

DURATION_STEP = 30
AGE_STEP = 5
AREA_STEP = 10
RENT_STEP = 20000

# PNGで表示する範囲。
# CSVには全範囲を保存する。
AGE_DISPLAY_MAX = 50
AREA_DISPLAY_MAX = 100
RENT_DISPLAY_MAX = 200000

# 築年数 × 間取りの図で表示する最低件数
MIN_CELL_N = 5

# 間取りの図で表示する上位カテゴリ数
TOP_LAYOUT_COUNT = 8


# ============================================================
# matplotlib 日本語設定
# ============================================================

plt.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "DejaVu Sans",
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 補助関数
# ============================================================

def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def round_up_axis(
    value: float,
    unit: int = 20,
) -> float:
    """
    縦軸上限を見やすい値へ切り上げる。
    """

    if pd.isna(value) or value <= 0:
        return float(unit)

    return float(
        math.ceil(value / unit)
        * unit
    )


def make_numeric_bands(
    series: pd.Series,
    step: float,
    suffix: str,
) -> tuple[np.ndarray, list[str]]:
    """
    0から最大値まで同じ幅の区間を作る。
    """

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    x = x[x >= 0]

    if x.empty:
        edges = np.array(
            [0, step],
            dtype=float,
        )

    else:
        maximum = float(x.max())

        upper = (
            math.floor(maximum / step)
            + 1
        ) * step

        edges = np.arange(
            0,
            upper + step,
            step,
            dtype=float,
        )

    labels = []

    for left, right in zip(
        edges[:-1],
        edges[1:],
    ):

        if float(left).is_integer():
            left_text = str(int(left))
        else:
            left_text = f"{left:g}"

        if float(right).is_integer():
            right_text = str(int(right))
        else:
            right_text = f"{right:g}"

        labels.append(
            f"{left_text}–{right_text}{suffix}"
        )

    return edges, labels


def grouped_duration_stats(
    df: pd.DataFrame,
    value_column: str,
    step: float,
    suffix: str,
) -> pd.DataFrame:
    """
    数値属性を等間隔に区切り、
    各区間の掲載期間統計を計算する。
    """

    edges, labels = make_numeric_bands(
        df[value_column],
        step,
        suffix,
    )

    band = pd.cut(
        df[value_column],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    temp = pd.DataFrame(
        {
            "band": band,
            "duration": df["duration"],
        }
    )

    rows = []

    for i, label in enumerate(labels):

        values = (
            temp.loc[
                temp["band"] == label,
                "duration",
            ]
            .dropna()
        )

        left = float(edges[i])
        right = float(edges[i + 1])

        if len(values) == 0:

            rows.append(
                {
                    "band": label,
                    "band_left": left,
                    "band_right": right,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                }
            )

            continue

        rows.append(
            {
                "band": label,
                "band_left": left,
                "band_right": right,
                "n": int(len(values)),
                "mean": float(
                    values.mean()
                ),
                "median": float(
                    values.median()
                ),
                "q1": float(
                    values.quantile(0.25)
                ),
                "q3": float(
                    values.quantile(0.75)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 掲載期間全体の統計
# ============================================================

def duration_summary(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    values = (
        df["duration"]
        .dropna()
    )

    modes = (
        values
        .mode()
        .sort_values()
    )

    mode_value = (
        float(modes.iloc[0])
        if len(modes) > 0
        else np.nan
    )

    mode_frequency = (
        int(
            (values == mode_value).sum()
        )
        if len(modes) > 0
        else 0
    )

    summary = pd.DataFrame(
        [
            {
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "mode": mode_value,
                "mode_frequency": mode_frequency,
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        ]
    )

    save_csv(
        summary,
        out_dir
        / "00_summary"
        / "duration_summary.csv",
    )

    print(
        "\n【掲載期間】"
    )

    print(
        f"件数       : {len(values):,}"
    )

    print(
        f"平均値     : {values.mean():.2f} 日"
    )

    print(
        f"中央値     : {values.median():.2f} 日"
    )

    print(
        f"最頻値     : {mode_value:.2f} 日"
    )

    print(
        f"第1四分位  : {values.quantile(0.25):.2f} 日"
    )

    print(
        f"第3四分位  : {values.quantile(0.75):.2f} 日"
    )

    # 平均・中央値・最頻値の比較図
    labels = [
        "平均値",
        "中央値",
        "最頻値",
    ]

    graph_values = [
        values.mean(),
        values.median(),
        mode_value,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    bars = ax.bar(
        labels,
        graph_values,
    )

    ax.set_ylabel(
        "掲載期間（日）"
    )

    ax.set_title(
        "掲載期間の代表値"
    )

    for bar, value in zip(
        bars,
        graph_values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output = (
        out_dir
        / "00_summary"
        / "duration_representative_values.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 掲載期間分布
# ============================================================

def duration_distribution(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    table = grouped_duration_stats(
        df=df,
        value_column="duration",
        step=DURATION_STEP,
        suffix="日",
    )

    # duration自身をgroupbyすると
    # duration統計になってしまうので件数のみ使う
    edges, labels = make_numeric_bands(
        df["duration"],
        DURATION_STEP,
        "日",
    )

    bands = pd.cut(
        df["duration"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    counts = (
        bands
        .value_counts(sort=False)
        .rename_axis("掲載期間帯")
        .reset_index(name="件数")
    )

    save_csv(
        counts,
        out_dir
        / "01_distribution"
        / "duration_distribution.csv",
    )

    # 発表用には0～360日を表示
    display = counts.iloc[:12].copy()

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    ax.bar(
        display["掲載期間帯"].astype(str),
        display["件数"],
    )

    ax.set_xlabel(
        "掲載期間"
    )

    ax.set_ylabel(
        "物件数"
    )

    ax.set_title(
        "掲載期間の分布"
    )

    ax.tick_params(
        axis="x",
        rotation=40,
    )

    fig.tight_layout()

    path = (
        out_dir
        / "01_distribution"
        / "duration_distribution_0_360days.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 数値属性ごとの中央値
# ============================================================

def numeric_attribute_analysis(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    configs = {
        "age": {
            "step": AGE_STEP,
            "suffix": "年",
            "display_max": AGE_DISPLAY_MAX,
            "xlabel": "築年数帯",
            "title": "築年数帯と掲載期間中央値",
            "filename": "age",
        },
        "area": {
            "step": AREA_STEP,
            "suffix": "㎡",
            "display_max": AREA_DISPLAY_MAX,
            "xlabel": "面積帯",
            "title": "面積帯と掲載期間中央値",
            "filename": "area",
        },
        "rent": {
            "step": RENT_STEP,
            "suffix": "円",
            "display_max": RENT_DISPLAY_MAX,
            "xlabel": "家賃帯",
            "title": "家賃帯と掲載期間中央値",
            "filename": "rent",
        },
    }

    tables: dict[str, pd.DataFrame] = {}

    for column, config in configs.items():

        table = grouped_duration_stats(
            df=df,
            value_column=column,
            step=config["step"],
            suffix=config["suffix"],
        )

        tables[column] = table

        save_csv(
            table,
            out_dir
            / "02_single_attribute"
            / f"{config['filename']}_duration_stats.csv",
        )

    # ----------------------------------------
    # 共通の縦軸を決める
    # ----------------------------------------

    displayed_tables = {}

    displayed_medians = []

    for column, config in configs.items():

        display = tables[column][
            tables[column]["band_left"]
            < config["display_max"]
        ].copy()

        displayed_tables[column] = display

        displayed_medians.extend(
            display["median"]
            .dropna()
            .tolist()
        )

    common_ymax = round_up_axis(
        max(displayed_medians),
        unit=20,
    )

    print(
        f"\n数値属性グラフ共通Y軸: 0 ～ {common_ymax:.0f} 日"
    )

    # ----------------------------------------
    # 個別グラフ
    # ----------------------------------------

    for column, config in configs.items():

        display = displayed_tables[column]

        fig, ax = plt.subplots(
            figsize=(10, 5.2)
        )

        ax.bar(
            display["band"],
            display["median"],
        )

        ax.set_xlabel(
            config["xlabel"]
        )

        ax.set_ylabel(
            "掲載期間中央値（日）"
        )

        ax.set_title(
            config["title"]
        )

        # 重要：
        # 築年数・面積・家賃で同じ縦軸にする
        ax.set_ylim(
            0,
            common_ymax,
        )

        ax.tick_params(
            axis="x",
            rotation=40,
        )

        fig.tight_layout()

        path = (
            out_dir
            / "02_single_attribute"
            / f"{config['filename']}_duration_median.png"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# 間取り
# ============================================================

def layout_analysis(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    sub = (
        df[
            ["layout", "duration"]
        ]
        .dropna()
        .copy()
    )

    table = (
        sub.groupby(
            "layout",
            observed=True,
        )["duration"]
        .agg(
            n="size",
            mean="mean",
            median="median",
        )
        .reset_index()
    )

    q = (
        sub.groupby(
            "layout",
            observed=True,
        )["duration"]
        .quantile(
            [0.25, 0.75]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.25: "q1",
                0.75: "q3",
            }
        )
    )

    table = table.merge(
        q,
        on="layout",
        how="left",
    )

    table = table.sort_values(
        "n",
        ascending=False,
    )

    save_csv(
        table,
        out_dir
        / "02_single_attribute"
        / "layout_duration_stats.csv",
    )

    display = (
        table
        .head(TOP_LAYOUT_COUNT)
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        display["layout"],
        display["median"],
    )

    ax.set_xlabel(
        "間取り"
    )

    ax.set_ylabel(
        "掲載期間中央値（日）"
    )

    ax.set_title(
        "間取りと掲載期間中央値"
    )

    fig.tight_layout()

    path = (
        out_dir
        / "02_single_attribute"
        / "layout_duration_median.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 単変数の関連
# ============================================================

def eta_squared_sqrt(
    groups: pd.Series,
    values: pd.Series,
) -> float:

    temp = pd.DataFrame(
        {
            "group": groups,
            "value": values,
        }
    ).dropna()

    if temp.empty:
        return np.nan

    grand_mean = (
        temp["value"].mean()
    )

    ss_between = 0.0

    for _, group in temp.groupby(
        "group"
    ):

        ss_between += (
            len(group)
            * (
                group["value"].mean()
                - grand_mean
            )
            ** 2
        )

    ss_total = (
        (
            temp["value"]
            - grand_mean
        )
        ** 2
    ).sum()

    if ss_total == 0:
        return np.nan

    return float(
        np.sqrt(
            ss_between
            / ss_total
        )
    )


def single_variable_strength(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    rows = []

    for column in [
        "age",
        "rent",
        "area",
        "walk_distance",
    ]:

        sub = (
            df[
                [column, "duration"]
            ]
            .dropna()
        )

        pearson = (
            sub[column]
            .corr(
                sub["duration"],
                method="pearson",
            )
        )

        # scipyを追加しなくてもよいように、
        # 順位へ変換してSpearman相関を計算する
        spearman = (
            sub[column]
            .rank()
            .corr(
                sub["duration"].rank(),
                method="pearson",
            )
        )

        rows.append(
            {
                "変数": column,
                "型": "数値",
                "n": len(sub),
                "Pearson": pearson,
                "Spearman": spearman,
                "相関比": np.nan,
            }
        )

    layout_sub = (
        df[
            ["layout", "duration"]
        ]
        .dropna()
    )

    rows.append(
        {
            "変数": "layout",
            "型": "カテゴリ",
            "n": len(layout_sub),
            "Pearson": np.nan,
            "Spearman": np.nan,
            "相関比": eta_squared_sqrt(
                layout_sub["layout"],
                layout_sub["duration"],
            ),
        }
    )

    table = pd.DataFrame(rows)

    save_csv(
        table,
        out_dir
        / "03_relationship"
        / "single_variable_strength.csv",
    )


# ============================================================
# 2属性R²
# ============================================================

def pair_r2(
    df: pd.DataFrame,
    a: str,
    b: str,
) -> tuple[int, float]:

    sub = (
        df[
            [a, b, "duration"]
        ]
        .dropna()
        .copy()
    )

    if sub.empty:
        return 0, np.nan

    parts = []

    for column in [a, b]:

        if (
            pd.api.types.is_string_dtype(
                sub[column]
            )
            or isinstance(
                sub[column].dtype,
                pd.CategoricalDtype,
            )
        ):

            dummy = pd.get_dummies(
                sub[column],
                drop_first=True,
                dtype=float,
            )

            parts.append(
                dummy.to_numpy()
            )

        else:

            parts.append(
                sub[[column]]
                .astype(float)
                .to_numpy()
            )

    x = np.hstack(
        [
            np.ones(
                (len(sub), 1)
            ),
            *parts,
        ]
    )

    y = (
        sub["duration"]
        .astype(float)
        .to_numpy()
    )

    beta, *_ = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )

    prediction = (
        x @ beta
    )

    ss_res = (
        (
            y
            - prediction
        )
        ** 2
    ).sum()

    ss_total = (
        (
            y
            - y.mean()
        )
        ** 2
    ).sum()

    if ss_total == 0:
        return len(sub), np.nan

    r2 = (
        1
        - ss_res
        / ss_total
    )

    return (
        len(sub),
        float(r2),
    )


def pair_relationships(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    candidate_pairs = [
        ("age", "rent"),
        ("age", "layout"),
        ("rent", "layout"),
        ("age", "area"),
        ("rent", "area"),
        ("area", "layout"),
    ]

    rows = []

    for a, b in candidate_pairs:

        n, r2 = pair_r2(
            df,
            a,
            b,
        )

        rows.append(
            {
                "属性A": a,
                "属性B": b,
                "n": n,
                "R2": r2,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(
            "R2",
            ascending=False,
        )
    )

    save_csv(
        table,
        out_dir
        / "03_relationship"
        / "pair_r2.csv",
    )

    print(
        "\n【2属性のR²：探索用】"
    )

    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# 築年数 × 間取り
# ============================================================

def age_layout_analysis(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    edges, labels = make_numeric_bands(
        df["age"],
        AGE_STEP,
        "年",
    )

    sub = (
        df[
            [
                "age",
                "layout",
                "duration",
            ]
        ]
        .dropna()
        .copy()
    )

    sub["age_band"] = pd.cut(
        sub["age"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    rows = []

    layouts = sorted(
        sub["layout"]
        .dropna()
        .unique()
    )

    for age_band in labels:

        for layout in layouts:

            values = sub.loc[
                (
                    sub["age_band"]
                    == age_band
                )
                & (
                    sub["layout"]
                    == layout
                ),
                "duration",
            ]

            if len(values) == 0:

                rows.append(
                    {
                        "age_band": age_band,
                        "layout": layout,
                        "n": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                    }
                )

                continue

            rows.append(
                {
                    "age_band": age_band,
                    "layout": layout,
                    "n": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "median": float(
                        values.median()
                    ),
                    "q1": float(
                        values.quantile(0.25)
                    ),
                    "q3": float(
                        values.quantile(0.75)
                    ),
                }
            )

    table = pd.DataFrame(rows)

    save_csv(
        table,
        out_dir
        / "04_age_layout"
        / "age_layout_duration_stats.csv",
    )

    # ----------------------------
    # 表示する間取りを決める
    # ----------------------------

    top_layouts = (
        sub["layout"]
        .value_counts()
        .head(TOP_LAYOUT_COUNT)
        .index
        .tolist()
    )

    display_age_labels = [
        label
        for label, left
        in zip(
            labels,
            edges[:-1],
        )
        if left < AGE_DISPLAY_MAX
    ]

    view = table[
        table["age_band"].isin(
            display_age_labels
        )
        & table["layout"].isin(
            top_layouts
        )
        & table["n"].ge(
            MIN_CELL_N
        )
    ].copy()

    # ----------------------------
    # 間取りごとの折れ線
    # ----------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x_positions = np.arange(
        len(display_age_labels)
    )

    for layout in top_layouts:

        part = view[
            view["layout"]
            == layout
        ].copy()

        medians = []

        for age_label in display_age_labels:

            match = part[
                part["age_band"]
                == age_label
            ]

            if match.empty:
                medians.append(
                    np.nan
                )
            else:
                medians.append(
                    float(
                        match.iloc[0][
                            "median"
                        ]
                    )
                )

        ax.plot(
            x_positions,
            medians,
            marker="o",
            label=layout,
        )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        display_age_labels,
        rotation=40,
        ha="right",
    )

    ax.set_xlabel(
        "築年数帯"
    )

    ax.set_ylabel(
        "掲載期間中央値（日）"
    )

    ax.set_title(
        "築年数帯 × 間取りと掲載期間中央値"
    )

    ax.legend(
        title="間取り",
        ncol=2,
    )

    fig.tight_layout()

    path = (
        out_dir
        / "04_age_layout"
        / "age_layout_median.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 中央値 + Q1～Q3
# ============================================================

def age_range_analysis(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    table = grouped_duration_stats(
        df=df,
        value_column="age",
        step=AGE_STEP,
        suffix="年",
    )

    display = table[
        table["band_left"]
        < AGE_DISPLAY_MAX
    ].copy()

    x = np.arange(
        len(display)
    )

    median = (
        display["median"]
        .to_numpy(dtype=float)
    )

    q1 = (
        display["q1"]
        .to_numpy(dtype=float)
    )

    q3 = (
        display["q3"]
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.plot(
        x,
        median,
        marker="o",
        label="中央値",
    )

    ax.fill_between(
        x,
        q1,
        q3,
        alpha=0.25,
        label="第1四分位数～第3四分位数",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        display["band"],
        rotation=40,
        ha="right",
    )

    ax.set_xlabel(
        "築年数帯"
    )

    ax.set_ylabel(
        "掲載期間（日）"
    )

    ax.set_title(
        "築年数帯ごとの掲載期間中央値と中央50％の範囲"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        out_dir
        / "05_range"
        / "age_median_q1_q3.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 欠損状況
# ============================================================

def missing_summary(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:

    columns = [
        "duration",
        "rent",
        "area",
        "age",
        "layout",
        "walk_distance",
        "station",
        "railway",
    ]

    rows = []

    for column in columns:

        valid = int(
            df[column]
            .notna()
            .sum()
        )

        missing = (
            len(df)
            - valid
        )

        rows.append(
            {
                "変数": column,
                "全件数": len(df),
                "有効件数": valid,
                "欠損件数": missing,
                "欠損率": (
                    missing
                    / len(df)
                ),
            }
        )

    save_csv(
        pd.DataFrame(rows),
        out_dir
        / "00_summary"
        / "missing_summary.csv",
    )


# ============================================================
# main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "国分寺市の賃貸掲載データに対する"
            "探索的データ分析"
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
        help=(
            "国分寺市抽出済みParquetまたはCSV"
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_OUTPUT
        ),
        help="出力先",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    out_dir = Path(
        args.out
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "国分寺市 賃貸掲載データ EDA"
    )

    print(
        "=" * 70
    )

    print(
        f"入力: {input_path}"
    )

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    raw = load_prepared(
        input_path
    )

    df = select_eda_scope(
        raw,
        years=YEARS,
        keep_duplicates=True,
    )

    print(
        f"抽出済み全件数 : {len(raw):,}"
    )

    print(
        f"EDA対象件数    : {len(df):,}"
    )

    print(
        "対象年         : 2020～2023年"
    )

    # --------------------------------------------------------
    # 各分析
    # --------------------------------------------------------

    missing_summary(
        df,
        out_dir,
    )

    duration_summary(
        df,
        out_dir,
    )

    duration_distribution(
        df,
        out_dir,
    )

    numeric_attribute_analysis(
        df,
        out_dir,
    )

    layout_analysis(
        df,
        out_dir,
    )

    single_variable_strength(
        df,
        out_dir,
    )

    pair_relationships(
        df,
        out_dir,
    )

    age_layout_analysis(
        df,
        out_dir,
    )

    age_range_analysis(
        df,
        out_dir,
    )

    print(
        "\n完了しました。"
    )

    print(
        f"出力先: {out_dir}"
    )


if __name__ == "__main__":
    main()
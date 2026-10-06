from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# パス設定
# ============================================================

OHORI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = OHORI_DIR.parent

if str(OHORI_DIR) not in sys.path:
    sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared


INPUT_PATH = (
    REPO_DIR
    / "data"
    / "prepared"
    / "kokubunji_raw.parquet"
)

OUTPUT_DIR = (
    OHORI_DIR
    / "results"
    / "02_endpoint_audit"
)


# ============================================================
# CSV保存
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


# ============================================================
# メイン処理
# ============================================================

def main() -> None:

    print("=" * 70)
    print("掲載開始日・掲載終了日の監査")
    print("=" * 70)

    df = load_prepared(
        INPUT_PATH
    )

    print(
        f"全件数                 : {len(df):,}"
    )

    print(
        f"開始日あり             : {df['start_date'].notna().sum():,}"
    )

    print(
        f"終了日あり             : {df['end_date'].notna().sum():,}"
    )

    print(
        f"終了日なし             : {df['end_date'].isna().sum():,}"
    )

    negative_duration = (
        df["duration"].notna()
        & df["duration"].lt(0)
    )

    print(
        f"掲載期間が負           : {negative_duration.sum():,}"
    )

    print()

    print(
        f"最古の掲載開始日       : {df['start_date'].min()}"
    )

    print(
        f"最新の掲載開始日       : {df['start_date'].max()}"
    )

    print(
        f"最古の掲載終了日       : {df['end_date'].min()}"
    )

    print(
        f"最新の掲載終了日       : {df['end_date'].max()}"
    )

    # ========================================================
    # 掲載開始年ごとの状況
    # ========================================================

    year_rows = []

    start_years = sorted(
        df["start_year"]
        .dropna()
        .astype(int)
        .unique()
    )

    for year in start_years:

        sub = df[
            df["start_year"] == year
        ]

        valid_duration = (
            sub["duration"].notna()
            & sub["duration"].ge(0)
        )

        year_rows.append(
            {
                "掲載開始年": year,
                "件数": len(sub),
                "終了日あり": int(
                    sub["end_date"]
                    .notna()
                    .sum()
                ),
                "終了日なし": int(
                    sub["end_date"]
                    .isna()
                    .sum()
                ),
                "有効な掲載期間": int(
                    valid_duration.sum()
                ),
                "掲載期間中央値": (
                    sub.loc[
                        valid_duration,
                        "duration",
                    ]
                    .median()
                ),
                "掲載期間平均": (
                    sub.loc[
                        valid_duration,
                        "duration",
                    ]
                    .mean()
                ),
                "開始日最小": (
                    sub["start_date"]
                    .min()
                ),
                "開始日最大": (
                    sub["start_date"]
                    .max()
                ),
                "終了日最大": (
                    sub["end_date"]
                    .max()
                ),
            }
        )

    year_table = pd.DataFrame(
        year_rows
    )

    save_csv(
        year_table,
        "start_year_summary.csv",
    )

    print()
    print("【掲載開始年ごとの件数】")
    print(
        year_table.to_string(
            index=False
        )
    )

    # ========================================================
    # 終了日の頻度
    # ========================================================

    end_counts = (
        df["end_date"]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis("掲載終了日")
        .reset_index(name="件数")
    )

    save_csv(
        end_counts,
        "end_date_counts.csv",
    )

    print()
    print("【最後の20掲載終了日】")

    print(
        end_counts
        .tail(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 最新終了日への集中
    # ========================================================

    latest_end = (
        df["end_date"]
        .max()
    )

    if pd.notna(latest_end):

        latest_end_count = int(
            (
                df["end_date"]
                == latest_end
            )
            .sum()
        )

        last_7_days = (
            latest_end
            - pd.Timedelta(
                days=6
            )
        )

        last_30_days = (
            latest_end
            - pd.Timedelta(
                days=29
            )
        )

        end_last_7 = int(
            (
                df["end_date"]
                .between(
                    last_7_days,
                    latest_end,
                )
            )
            .sum()
        )

        end_last_30 = int(
            (
                df["end_date"]
                .between(
                    last_30_days,
                    latest_end,
                )
            )
            .sum()
        )

        print()

        print("【データ末尾付近への終了日の集中】")

        print(
            f"最新終了日             : {latest_end.date()}"
        )

        print(
            f"最新終了日と同日の件数 : {latest_end_count:,}"
        )

        print(
            f"最後の7日間の終了件数  : {end_last_7:,}"
        )

        print(
            f"最後の30日間の終了件数 : {end_last_30:,}"
        )

    # ========================================================
    # 2020～2023年のみの確認
    # ========================================================

    target = df[
        df["start_year"]
        .isin(
            [2020, 2021, 2022, 2023]
        )
    ].copy()

    target_summary = pd.DataFrame(
        [
            {
                "件数": len(target),
                "終了日あり": int(
                    target["end_date"]
                    .notna()
                    .sum()
                ),
                "終了日なし": int(
                    target["end_date"]
                    .isna()
                    .sum()
                ),
                "負の掲載期間": int(
                    (
                        target["duration"]
                        .notna()
                        & target["duration"].lt(0)
                    )
                    .sum()
                ),
                "有効掲載期間": int(
                    (
                        target["duration"]
                        .notna()
                        & target["duration"].ge(0)
                    )
                    .sum()
                ),
                "開始日最小": (
                    target["start_date"]
                    .min()
                ),
                "開始日最大": (
                    target["start_date"]
                    .max()
                ),
                "終了日最大": (
                    target["end_date"]
                    .max()
                ),
            }
        ]
    )

    save_csv(
        target_summary,
        "target_2020_2023_summary.csv",
    )

    print()
    print("【2020～2023年掲載開始】")
    print(
        target_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"CSV出力先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
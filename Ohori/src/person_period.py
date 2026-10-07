from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# デフォルトで引き継ぐ物件情報
# ============================================================

DEFAULT_FEATURE_COLUMNS = [
    "id",
    "start_year",
    "start_month",
    "age",
    "rent",
    "area",
    "walk_distance",
    "layout",
    "station",
    "railway",
]


# ============================================================
# Person-period形式への変換
# ============================================================

def make_person_period(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    生存分析用の1物件1行データを、
    離散時間生存モデル用のperson-period形式へ変換する。

    例：
        掲載期間3日で3日目に掲載終了した物件

        time_day    event_in_day
        0           0
        1           0
        2           0
        3           1

    掲載期間0日の場合：

        time_day    event_in_day
        0           1

    右打ち切りの場合は、
    最終観測日までevent_in_day=0となる。
    """

    required_columns = {
        "survival_days",
        "event",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"必要な列がありません: {sorted(missing)}"
        )

    if feature_columns is None:
        feature_columns = (
            DEFAULT_FEATURE_COLUMNS
        )

    feature_columns = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    source = (
        df.reset_index(drop=True)
        .copy()
    )

    # survival_daysは今回すべて日単位の整数。
    # 念のため整数へ変換する。
    duration_days = (
        source["survival_days"]
        .astype(int)
        .to_numpy()
    )

    if (
        duration_days < 0
    ).any():
        raise ValueError(
            "survival_days に負の値があります。"
        )

    # 0日掲載でも1行必要なので +1
    row_counts = (
        duration_days
        + 1
    )

    total_rows = int(
        row_counts.sum()
    )

    # ========================================================
    # 各person-period行が、
    # 元のどの物件に属するか
    # ========================================================

    listing_position = np.repeat(
        np.arange(
            len(source)
        ),
        row_counts,
    )

    # 各物件のperson-period開始位置
    starts = (
        np.cumsum(row_counts)
        - row_counts
    )

    repeated_starts = np.repeat(
        starts,
        row_counts,
    )

    # 0,1,2,... と掲載経過日数を作る
    time_day = (
        np.arange(total_rows)
        - repeated_starts
    )

    # ========================================================
    # イベント列
    # ========================================================

    event_in_day = np.zeros(
        total_rows,
        dtype=np.int8,
    )

    # 各物件の最後のperson-period行
    last_positions = (
        np.cumsum(row_counts)
        - 1
    )

    event_mask = (
        source["event"]
        .astype(int)
        .to_numpy()
        == 1
    )

    # 掲載終了を観測した物件だけ、
    # 最終日にイベント=1
    event_in_day[
        last_positions[event_mask]
    ] = 1

    # ========================================================
    # 元物件情報を繰り返す
    # ========================================================

    repeated_features = (
        source.iloc[
            listing_position
        ][feature_columns]
        .reset_index(drop=True)
    )

    person_period = (
        repeated_features.copy()
    )

    # 元データ上の何番目の物件か
    person_period[
        "listing_index"
    ] = listing_position

    # 掲載開始から何日目か
    person_period[
        "time_day"
    ] = time_day.astype(int)

    # その日に掲載終了したか
    person_period[
        "event_in_day"
    ] = event_in_day

    return person_period


# ============================================================
# 変換結果の検証
# ============================================================

def validate_person_period(
    source: pd.DataFrame,
    person_period: pd.DataFrame,
) -> dict:
    """
    person-period変換が正しいか検査する。
    """

    source = (
        source.reset_index(
            drop=True
        )
    )

    # ----------------------------------------
    # 元のイベント数と一致するか
    # ----------------------------------------

    source_event_count = int(
        source["event"]
        .sum()
    )

    pp_event_count = int(
        person_period[
            "event_in_day"
        ]
        .sum()
    )

    # ----------------------------------------
    # 1物件にイベントが複数ないか
    # ----------------------------------------

    event_per_listing = (
        person_period
        .groupby(
            "listing_index"
        )["event_in_day"]
        .sum()
    )

    multiple_event_count = int(
        (
            event_per_listing
            > 1
        )
        .sum()
    )

    # ----------------------------------------
    # 元の掲載日数と最終time_dayが一致するか
    # ----------------------------------------

    last_time = (
        person_period
        .groupby(
            "listing_index"
        )["time_day"]
        .max()
    )

    expected_time = (
        source[
            "survival_days"
        ]
        .astype(int)
    )

    duration_mismatch_count = int(
        (
            last_time.to_numpy()
            != expected_time.to_numpy()
        )
        .sum()
    )

    # ----------------------------------------
    # event=1の物件で、
    # 最終日にイベントが立っているか
    # ----------------------------------------

    last_rows = (
        person_period
        .sort_values(
            [
                "listing_index",
                "time_day",
            ]
        )
        .groupby(
            "listing_index",
            as_index=False,
        )
        .tail(1)
        .sort_values(
            "listing_index"
        )
    )

    expected_event = (
        source["event"]
        .astype(int)
        .to_numpy()
    )

    actual_last_event = (
        last_rows[
            "event_in_day"
        ]
        .astype(int)
        .to_numpy()
    )

    last_event_mismatch_count = int(
        (
            expected_event
            != actual_last_event
        )
        .sum()
    )

    result = {
        "listing_count": int(
            len(source)
        ),
        "person_period_rows": int(
            len(person_period)
        ),
        "source_event_count": (
            source_event_count
        ),
        "person_period_event_count": (
            pp_event_count
        ),
        "multiple_event_listings": (
            multiple_event_count
        ),
        "duration_mismatch": (
            duration_mismatch_count
        ),
        "last_event_mismatch": (
            last_event_mismatch_count
        ),
    }

    return result
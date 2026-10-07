from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


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
    / "04_baseline"
)


# ============================================================
# 評価条件
# ============================================================

TRAIN_YEARS = [
    2020,
    2021,
]

VALID_YEAR = 2022

# 「現在掲載○日目」を想定する評価地点
LANDMARK_DAYS = [
    7,
    14,
    30,
]

# そこから何日以内に終了するか
HORIZONS = [
    7,
    14,
    30,
]

# 築年数帯
AGE_BINS = [
    0,
    10,
    20,
    30,
    40,
    np.inf,
]

AGE_LABELS = [
    "0～10年",
    "10～20年",
    "20～30年",
    "30～40年",
    "40年以上",
]


# ============================================================
# 保存
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
# 築年数帯
# ============================================================

def add_age_group(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["age_group"] = pd.cut(
        result["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )

    return result


# ============================================================
# Kaplan-Meier
# ============================================================

def fit_km(
    df: pd.DataFrame,
) -> KaplanMeierFitter:

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=df["survival_days"],
        event_observed=df["event"],
    )

    return kmf


def conditional_end_probability(
    kmf: KaplanMeierFitter,
    current_day: int,
    horizon: int,
) -> float:
    """
    current_dayまで掲載が続いているという条件のもとで、
    次のhorizon日以内に掲載終了する確率。

    P(T <= e+h | T > e)
      = 1 - S(e+h) / S(e)
    """

    survival_now = float(
        kmf.predict(
            current_day
        )
    )

    survival_future = float(
        kmf.predict(
            current_day + horizon
        )
    )

    if survival_now <= 0:
        return np.nan

    probability = (
        1
        - survival_future
        / survival_now
    )

    return float(
        np.clip(
            probability,
            0,
            1,
        )
    )


# ============================================================
# Baseline 1
# 全物件Kaplan-Meier
# ============================================================

def make_overall_baseline(
    train: pd.DataFrame,
) -> dict[tuple[int, int], float]:

    kmf = fit_km(
        train
    )

    result = {}

    rows = []

    for landmark in LANDMARK_DAYS:

        for horizon in HORIZONS:

            probability = (
                conditional_end_probability(
                    kmf,
                    current_day=landmark,
                    horizon=horizon,
                )
            )

            result[
                (
                    landmark,
                    horizon,
                )
            ] = probability

            rows.append(
                {
                    "現在掲載日数": landmark,
                    "予測期間": horizon,
                    "終了確率": probability,
                }
            )

    save_csv(
        pd.DataFrame(rows),
        "baseline_overall_probabilities.csv",
    )

    return result


# ============================================================
# Baseline 2
# 築年数帯 × 間取り Kaplan-Meier
# ============================================================

def make_group_baseline(
    train: pd.DataFrame,
) -> dict:

    models = {}

    rows = []

    valid = train[
        train["age_group"].notna()
        & train["layout"].notna()
    ].copy()

    grouped = valid.groupby(
        [
            "age_group",
            "layout",
        ],
        observed=True,
    )

    for (
        age_group,
        layout,
    ), group_df in grouped:

        if len(group_df) == 0:
            continue

        kmf = fit_km(
            group_df
        )

        key = (
            str(age_group),
            str(layout),
        )

        models[key] = kmf

        rows.append(
            {
                "築年数帯": str(
                    age_group
                ),
                "間取り": str(
                    layout
                ),
                "学習件数": len(
                    group_df
                ),
                "イベント数": int(
                    group_df["event"]
                    .sum()
                ),
                "右打ち切り数": int(
                    (
                        group_df["event"]
                        == 0
                    )
                    .sum()
                ),
            }
        )

    save_csv(
        pd.DataFrame(rows),
        "baseline_group_counts.csv",
    )

    return models


# ============================================================
# 検証データをランドマーク形式に変換
# ============================================================

def make_validation_rows(
    validation: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:
    """
    landmark日より後まで掲載が続いていた物件を対象にする。

    target = 1:
        landmarkより後、
        landmark+horizon以内に掲載終了

    target = 0:
        landmark+horizonより後まで掲載継続を確認

    horizon終了前に右打ち切りになった場合は、
    正解が分からないため評価から除外する。
    """

    # landmark時点でまだ掲載継続していた物件
    at_risk = validation[
        validation["survival_days"]
        > landmark
    ].copy()

    horizon_end = (
        landmark
        + horizon
    )

    # horizon内に掲載終了
    positive = (
        (at_risk["event"] == 1)
        & (
            at_risk["survival_days"]
            <= horizon_end
        )
    )

    # horizon終了より後まで観測できた
    negative = (
        at_risk["survival_days"]
        > horizon_end
    )

    known = (
        positive
        | negative
    )

    result = (
        at_risk.loc[known]
        .copy()
    )

    result["target"] = (
        positive.loc[known]
        .astype(int)
    )

    result["landmark_day"] = landmark
    result["horizon"] = horizon

    return result


# ============================================================
# 評価指標
# ============================================================

def brier_score(
    y: np.ndarray,
    p: np.ndarray,
) -> float:

    return float(
        np.mean(
            (
                y
                - p
            )
            ** 2
        )
    )


def calibration_summary(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:

    return (
        float(
            np.mean(p)
        ),
        float(
            np.mean(y)
        ),
    )


# ============================================================
# 2022年で評価
# ============================================================

def evaluate_baselines(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:

    overall_predictions = (
        make_overall_baseline(
            train
        )
    )

    group_models = (
        make_group_baseline(
            train
        )
    )

    evaluation_rows = []

    prediction_rows = []

    for landmark in LANDMARK_DAYS:

        for horizon in HORIZONS:

            test = make_validation_rows(
                validation,
                landmark=landmark,
                horizon=horizon,
            )

            if test.empty:
                continue

            y = (
                test["target"]
                .to_numpy(
                    dtype=float
                )
            )

            # --------------------------------
            # Baseline 1
            # 全体KM
            # --------------------------------

            p_overall_value = (
                overall_predictions[
                    (
                        landmark,
                        horizon,
                    )
                ]
            )

            p_overall = np.full(
                len(test),
                p_overall_value,
                dtype=float,
            )

            mean_pred, actual = (
                calibration_summary(
                    y,
                    p_overall,
                )
            )

            evaluation_rows.append(
                {
                    "baseline": "overall_km",
                    "landmark_day": landmark,
                    "horizon": horizon,
                    "n": len(test),
                    "predicted_rate": mean_pred,
                    "actual_rate": actual,
                    "brier_score": brier_score(
                        y,
                        p_overall,
                    ),
                    "fallback_count": 0,
                }
            )

            # --------------------------------
            # Baseline 2
            # 築年数帯 × 間取り
            # --------------------------------

            group_predictions = []

            fallback_count = 0

            for _, row in test.iterrows():

                key = (
                    str(
                        row["age_group"]
                    ),
                    str(
                        row["layout"]
                    ),
                )

                kmf = group_models.get(
                    key
                )

                if (
                    kmf is None
                    or pd.isna(
                        row["age_group"]
                    )
                    or pd.isna(
                        row["layout"]
                    )
                ):

                    probability = (
                        p_overall_value
                    )

                    fallback_count += 1

                else:

                    probability = (
                        conditional_end_probability(
                            kmf,
                            current_day=landmark,
                            horizon=horizon,
                        )
                    )

                    if pd.isna(
                        probability
                    ):

                        probability = (
                            p_overall_value
                        )

                        fallback_count += 1

                group_predictions.append(
                    probability
                )

            p_group = np.asarray(
                group_predictions,
                dtype=float,
            )

            mean_pred, actual = (
                calibration_summary(
                    y,
                    p_group,
                )
            )

            evaluation_rows.append(
                {
                    "baseline": "age_layout_km",
                    "landmark_day": landmark,
                    "horizon": horizon,
                    "n": len(test),
                    "predicted_rate": mean_pred,
                    "actual_rate": actual,
                    "brier_score": brier_score(
                        y,
                        p_group,
                    ),
                    "fallback_count": fallback_count,
                }
            )

            # --------------------------------
            # 個別予測も保存
            # --------------------------------

            output = test[
                [
                    "id",
                    "age",
                    "age_group",
                    "layout",
                    "survival_days",
                    "event",
                    "target",
                ]
            ].copy()

            output[
                "landmark_day"
            ] = landmark

            output[
                "horizon"
            ] = horizon

            output[
                "prediction_overall_km"
            ] = p_overall

            output[
                "prediction_age_layout_km"
            ] = p_group

            prediction_rows.append(
                output
            )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    save_csv(
        evaluation,
        "validation_2022_metrics.csv",
    )

    if prediction_rows:

        predictions = pd.concat(
            prediction_rows,
            ignore_index=True,
        )

        save_csv(
            predictions,
            "validation_2022_predictions.csv",
        )

    print()
    print("【2022年 ベースライン評価】")

    print(
        evaluation.to_string(
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
        "国分寺市 掲載終了確率 ベースライン評価"
    )

    print(
        "=" * 70
    )

    raw = load_prepared(
        INPUT_PATH
    )

    survival = (
        prepare_survival_data(
            raw
        )
    )

    survival = add_age_group(
        survival
    )

    train = (
        survival[
            survival["start_year"]
            .isin(TRAIN_YEARS)
        ]
        .copy()
    )

    validation = (
        survival[
            survival["start_year"]
            == VALID_YEAR
        ]
        .copy()
    )

    print(
        f"学習 2020～2021 : {len(train):,}件"
    )

    print(
        f"検証 2022       : {len(validation):,}件"
    )

    print(
        "最終テスト2023年はこの段階では使用しません。"
    )

    evaluate_baselines(
        train,
        validation,
    )

    print()
    print(
        f"出力先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
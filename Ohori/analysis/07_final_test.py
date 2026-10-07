from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


# ============================================================
# パス
# ============================================================

OHORI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = OHORI_DIR.parent

if str(OHORI_DIR) not in sys.path:
    sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared
from src.survival import prepare_survival_data
from src.discrete_model import (
    fit_model,
    evaluate_model,
)


INPUT_PATH = (
    REPO_DIR
    / "data"
    / "prepared"
    / "kokubunji_raw.parquet"
)

OUTPUT_DIR = (
    OHORI_DIR
    / "results"
    / "07_final_test"
)


# ============================================================
# 最終学習・テスト期間
# ============================================================

TRAIN_YEARS = [
    2020,
    2021,
    2022,
]

TEST_YEAR = 2023


# ============================================================
# 評価条件
# ============================================================

LANDMARK_DAYS = [
    7,
    14,
    30,
]

HORIZONS = [
    7,
    14,
    30,
]


# ============================================================
# 築年数帯
# ============================================================

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
# テスト対象
# ============================================================

def make_test_rows(
    test: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:

    at_risk = test[
        test["survival_days"]
        > landmark
    ].copy()

    horizon_end = (
        landmark
        + horizon
    )

    positive = (
        (at_risk["event"] == 1)
        & (
            at_risk["survival_days"]
            <= horizon_end
        )
    )

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

    return result


# ============================================================
# Overall KM baseline
# ============================================================

def evaluate_overall_km(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    kmf = fit_km(
        train
    )

    rows = []

    for landmark in LANDMARK_DAYS:

        for horizon in HORIZONS:

            test_rows = make_test_rows(
                test,
                landmark,
                horizon,
            )

            if test_rows.empty:
                continue

            y = (
                test_rows["target"]
                .to_numpy(
                    dtype=int
                )
            )

            probability = (
                conditional_end_probability(
                    kmf,
                    current_day=landmark,
                    horizon=horizon,
                )
            )

            p = np.full(
                len(test_rows),
                probability,
                dtype=float,
            )

            rows.append(
                {
                    "model": "overall_km",
                    "landmark_day": landmark,
                    "horizon": horizon,
                    "n": len(test_rows),
                    "predicted_rate": float(
                        p.mean()
                    ),
                    "actual_rate": float(
                        y.mean()
                    ),
                    "brier_score": float(
                        brier_score_loss(
                            y,
                            p,
                        )
                    ),
                    "roc_auc": np.nan,
                    "fallback_count": 0,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 築年数帯 × 間取り KM baseline
# ============================================================

def fit_age_layout_km(
    train: pd.DataFrame,
) -> dict:

    models = {}

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

        if group_df.empty:
            continue

        key = (
            str(age_group),
            str(layout),
        )

        models[key] = fit_km(
            group_df
        )

    return models


def evaluate_age_layout_km(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    overall_km = fit_km(
        train
    )

    group_models = (
        fit_age_layout_km(
            train
        )
    )

    rows = []

    for landmark in LANDMARK_DAYS:

        for horizon in HORIZONS:

            test_rows = make_test_rows(
                test,
                landmark,
                horizon,
            )

            if test_rows.empty:
                continue

            y = (
                test_rows["target"]
                .to_numpy(
                    dtype=int
                )
            )

            predictions = []
            fallback_count = 0

            overall_probability = (
                conditional_end_probability(
                    overall_km,
                    current_day=landmark,
                    horizon=horizon,
                )
            )

            for _, row in test_rows.iterrows():

                if (
                    pd.isna(
                        row["age_group"]
                    )
                    or pd.isna(
                        row["layout"]
                    )
                ):

                    probability = (
                        overall_probability
                    )

                    fallback_count += 1

                else:

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

                    if kmf is None:

                        probability = (
                            overall_probability
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
                                overall_probability
                            )

                            fallback_count += 1

                predictions.append(
                    probability
                )

            p = np.asarray(
                predictions,
                dtype=float,
            )

            rows.append(
                {
                    "model": "age_layout_km",
                    "landmark_day": landmark,
                    "horizon": horizon,
                    "n": len(test_rows),
                    "predicted_rate": float(
                        p.mean()
                    ),
                    "actual_rate": float(
                        y.mean()
                    ),
                    "brier_score": float(
                        brier_score_loss(
                            y,
                            p,
                        )
                    ),
                    "roc_auc": (
                        float(
                            roc_auc_score(
                                y,
                                p,
                            )
                        )
                        if len(
                            np.unique(y)
                        ) == 2
                        else np.nan
                    ),
                    "fallback_count": (
                        fallback_count
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# サマリー
# ============================================================

def make_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    d = metrics.copy()

    d["calibration_error"] = (
        d["predicted_rate"]
        - d["actual_rate"]
    ).abs()

    rows = []

    for model_name, group in d.groupby(
        "model"
    ):

        auc = (
            group["roc_auc"]
            .dropna()
        )

        rows.append(
            {
                "model": model_name,
                "評価条件数": len(
                    group
                ),
                "平均Brier": float(
                    group[
                        "brier_score"
                    ].mean()
                ),
                "Brier中央値": float(
                    group[
                        "brier_score"
                    ].median()
                ),
                "平均Calibration誤差": float(
                    group[
                        "calibration_error"
                    ].mean()
                ),
                "平均AUC": (
                    float(
                        auc.mean()
                    )
                    if len(auc) > 0
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "平均Brier",
            ascending=True,
        )
    )


# ============================================================
# 条件ごとの勝者
# ============================================================

def make_winners(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = metrics.groupby(
        [
            "landmark_day",
            "horizon",
        ]
    )

    for (
        landmark,
        horizon,
    ), group in grouped:

        winner = (
            group.sort_values(
                "brier_score",
                ascending=True,
            )
            .iloc[0]
        )

        rows.append(
            {
                "現在掲載日数": landmark,
                "予測期間": horizon,
                "最良モデル": winner[
                    "model"
                ],
                "最良Brier": winner[
                    "brier_score"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# main
# ============================================================

def main() -> None:

    print(
        "=" * 72
    )

    print(
        "国分寺市 掲載終了確率 最終テスト"
    )

    print(
        "=" * 72
    )

    print(
        "採用モデル：Model 2 "
        "（基本特徴量 + 築年数帯×間取り）"
    )

    print(
        "2023年はモデル選択には使用していません。"
    )

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------

    raw = load_prepared(
        INPUT_PATH
    )

    survival = prepare_survival_data(
        raw
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

    test = (
        survival[
            survival["start_year"]
            == TEST_YEAR
        ]
        .copy()
    )

    print()
    print(
        f"最終学習 2020～2022 : {len(train):,}件"
    )

    print(
        f"最終テスト 2023     : {len(test):,}件"
    )

    # ========================================================
    # Model 2
    # ========================================================

    print()
    print(
        "最終Model 2を学習しています..."
    )

    final_model, pp_rows = fit_model(
        train,
        use_age_layout_combo=True,
    )

    print(
        f"学習完了 "
        f"(person-period {pp_rows:,}行)"
    )

    model_metrics, model_predictions = (
        evaluate_model(
            model=final_model,
            validation=test,
            model_name="model2_age_layout",
            landmarks=LANDMARK_DAYS,
            horizons=HORIZONS,
            use_age_layout_combo=True,
        )
    )

    # ========================================================
    # Baseline
    # ========================================================

    print()
    print(
        "Baselineを2020～2022年で再学習しています..."
    )

    overall_metrics = (
        evaluate_overall_km(
            train,
            test,
        )
    )

    group_metrics = (
        evaluate_age_layout_km(
            train,
            test,
        )
    )

    # ========================================================
    # 全結果結合
    # ========================================================

    all_metrics = pd.concat(
        [
            model_metrics,
            group_metrics,
            overall_metrics,
        ],
        ignore_index=True,
    )

    all_metrics = all_metrics.sort_values(
        [
            "landmark_day",
            "horizon",
            "brier_score",
        ]
    )

    summary = make_summary(
        all_metrics
    )

    winners = make_winners(
        all_metrics
    )

    # ========================================================
    # 保存
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        OUTPUT_DIR
        / "final_model2_age_layout.joblib",
    )

    save_csv(
        model_predictions,
        "final_model_2023_predictions.csv",
    )

    save_csv(
        all_metrics,
        "final_2023_metrics.csv",
    )

    save_csv(
        summary,
        "final_2023_summary.csv",
    )

    save_csv(
        winners,
        "final_2023_condition_winners.csv",
    )

    # ========================================================
    # 表示
    # ========================================================

    print()
    print(
        "【2023年 条件別最終評価】"
    )

    print(
        all_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "【2023年 最終サマリー】"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "【各条件で最良だったモデル】"
    )

    print(
        winners.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "2023年最終テスト完了"
    )

    print(
        "=" * 72
    )

    print()
    print(
        f"出力先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
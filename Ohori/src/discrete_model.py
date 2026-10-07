from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.person_period import make_person_period


# ============================================================
# 基本特徴量
# ============================================================

NUMERIC_FEATURES = [
    "time_day",
    "age",
    "rent",
    "area",
    "walk_distance",
]

BASE_CATEGORICAL_FEATURES = [
    "time_band",
    "layout",
    "station",
    "start_month",
]


# ============================================================
# 掲載経過時間
# ============================================================

TIME_BINS = [
    -1,
    6,
    13,
    29,
    59,
    89,
    179,
    364,
    np.inf,
]

TIME_LABELS = [
    "0～6日",
    "7～13日",
    "14～29日",
    "30～59日",
    "60～89日",
    "90～179日",
    "180～364日",
    "365日以上",
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
# 特徴量名
# ============================================================

def categorical_features(
    use_age_layout_combo: bool,
) -> list[str]:

    features = list(
        BASE_CATEGORICAL_FEATURES
    )

    if use_age_layout_combo:

        features.append(
            "age_layout_combo"
        )

    return features


def model_features(
    use_age_layout_combo: bool,
) -> list[str]:

    return (
        NUMERIC_FEATURES
        + categorical_features(
            use_age_layout_combo
        )
    )


# ============================================================
# モデル用特徴量作成
# ============================================================

def prepare_feature_frame(
    df: pd.DataFrame,
    use_age_layout_combo: bool = False,
) -> pd.DataFrame:
    """
    scikit-learnへ渡す特徴量を作成する。

    Model 1:
        基本特徴量のみ

    Model 2:
        基本特徴量
        +
        築年数帯 × 間取り
    """

    d = df.copy()

    # --------------------------------------------------------
    # 掲載経過時間帯
    # --------------------------------------------------------

    d["time_band"] = pd.cut(
        d["time_day"],
        bins=TIME_BINS,
        labels=TIME_LABELS,
        right=True,
        include_lowest=True,
    )

    # --------------------------------------------------------
    # 築年数帯 × 間取り
    # --------------------------------------------------------

    if use_age_layout_combo:

        age_group = pd.cut(
            pd.to_numeric(
                d["age"],
                errors="coerce",
            ),
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
            include_lowest=True,
        )

        age_text = (
            age_group
            .astype("string")
            .fillna("欠損")
        )

        layout_text = (
            d["layout"]
            .astype("string")
            .fillna("欠損")
        )

        d["age_layout_combo"] = (
            age_text
            + " × "
            + layout_text
        )

    # --------------------------------------------------------
    # 必要列だけ取得
    # --------------------------------------------------------

    features = model_features(
        use_age_layout_combo
    )

    x = d[
        features
    ].copy()

    # --------------------------------------------------------
    # 数値
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        ).astype(float)

    # --------------------------------------------------------
    # カテゴリ
    # --------------------------------------------------------

    for column in categorical_features(
        use_age_layout_combo
    ):

        x[column] = (
            x[column]
            .astype("string")
            .fillna("欠損")
            .astype(str)
        )

    return x


# ============================================================
# モデル構築
# ============================================================

def build_model(
    use_age_layout_combo: bool = False,
) -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features(
                    use_age_layout_combo
                ),
            ),
        ]
    )

    logistic = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                logistic,
            ),
        ]
    )


# ============================================================
# 学習
# ============================================================

def fit_model(
    train: pd.DataFrame,
    use_age_layout_combo: bool = False,
) -> tuple[Pipeline, int]:

    person_period = make_person_period(
        train
    )

    x = prepare_feature_frame(
        person_period,
        use_age_layout_combo=(
            use_age_layout_combo
        ),
    )

    y = (
        person_period[
            "event_in_day"
        ]
        .astype(int)
    )

    model = build_model(
        use_age_layout_combo=(
            use_age_layout_combo
        )
    )

    model.fit(
        x,
        y,
    )

    return (
        model,
        len(person_period),
    )


# ============================================================
# 評価対象作成
# ============================================================

def make_validation_rows(
    validation: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:

    at_risk = validation[
        validation["survival_days"]
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
# 将来の日単位データ作成
# ============================================================

def make_future_rows(
    listings: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:

    days = np.arange(
        landmark + 1,
        landmark + horizon + 1,
        dtype=int,
    )

    listing_positions = np.repeat(
        np.arange(
            len(listings)
        ),
        len(days),
    )

    future_days = np.tile(
        days,
        len(listings),
    )

    columns = [
        "age",
        "rent",
        "area",
        "walk_distance",
        "layout",
        "station",
        "start_month",
    ]

    future = (
        listings.iloc[
            listing_positions
        ][columns]
        .reset_index(drop=True)
    )

    future["prediction_index"] = (
        listing_positions
    )

    future["time_day"] = (
        future_days
    )

    return future


# ============================================================
# horizon以内終了確率
# ============================================================

def predict_horizon_probability(
    model: Pipeline,
    listings: pd.DataFrame,
    landmark: int,
    horizon: int,
    use_age_layout_combo: bool = False,
) -> np.ndarray:

    future = make_future_rows(
        listings=listings,
        landmark=landmark,
        horizon=horizon,
    )

    x = prepare_feature_frame(
        future,
        use_age_layout_combo=(
            use_age_layout_combo
        ),
    )

    daily_hazard = (
        model.predict_proba(
            x
        )[:, 1]
    )

    future[
        "log_survival"
    ] = np.log1p(
        -np.clip(
            daily_hazard,
            0,
            1 - 1e-12,
        )
    )

    conditional_survival = (
        future
        .groupby(
            "prediction_index"
        )["log_survival"]
        .sum()
        .pipe(np.exp)
    )

    probability = (
        1
        - conditional_survival
        .to_numpy()
    )

    return np.clip(
        probability,
        0,
        1,
    )


# ============================================================
# 評価
# ============================================================

def evaluate_model(
    model: Pipeline,
    validation: pd.DataFrame,
    model_name: str,
    landmarks: list[int],
    horizons: list[int],
    use_age_layout_combo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    metric_rows = []
    prediction_rows = []

    for landmark in landmarks:

        for horizon in horizons:

            test = make_validation_rows(
                validation=validation,
                landmark=landmark,
                horizon=horizon,
            )

            if test.empty:
                continue

            y = (
                test["target"]
                .to_numpy(
                    dtype=int
                )
            )

            p = predict_horizon_probability(
                model=model,
                listings=test,
                landmark=landmark,
                horizon=horizon,
                use_age_layout_combo=(
                    use_age_layout_combo
                ),
            )

            brier = brier_score_loss(
                y,
                p,
            )

            if len(
                np.unique(y)
            ) == 2:

                auc = roc_auc_score(
                    y,
                    p,
                )

            else:

                auc = np.nan

            metric_rows.append(
                {
                    "model": model_name,
                    "landmark_day": landmark,
                    "horizon": horizon,
                    "n": len(test),
                    "predicted_rate": float(
                        p.mean()
                    ),
                    "actual_rate": float(
                        y.mean()
                    ),
                    "brier_score": float(
                        brier
                    ),
                    "roc_auc": float(
                        auc
                    ),
                }
            )

            output = test[
                [
                    "id",
                    "target",
                ]
            ].copy()

            output[
                "model"
            ] = model_name

            output[
                "landmark_day"
            ] = landmark

            output[
                "horizon"
            ] = horizon

            output[
                "prediction"
            ] = p

            prediction_rows.append(
                output
            )

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    return (
        metrics,
        predictions,
    )
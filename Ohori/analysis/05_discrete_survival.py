from __future__ import annotations

import sys
from pathlib import Path

import joblib
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


# ============================================================
# パス
# ============================================================

OHORI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = OHORI_DIR.parent

if str(OHORI_DIR) not in sys.path:
    sys.path.insert(0, str(OHORI_DIR))

from src.data import load_prepared
from src.survival import prepare_survival_data
from src.person_period import make_person_period


INPUT_PATH = (
    REPO_DIR
    / "data"
    / "prepared"
    / "kokubunji_raw.parquet"
)

OUTPUT_DIR = (
    OHORI_DIR
    / "results"
    / "05_discrete_survival"
)

BASELINE_METRICS_PATH = (
    OHORI_DIR
    / "results"
    / "04_baseline"
    / "validation_2022_metrics.csv"
)


# ============================================================
# 学習・検証年
# ============================================================

TRAIN_YEARS = [
    2020,
    2021,
]

VALID_YEAR = 2022


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
# モデルで使う変数
# ============================================================

NUMERIC_FEATURES = [
    "time_day",
    "age",
    "rent",
    "area",
    "walk_distance",
]

CATEGORICAL_FEATURES = [
    "time_band",
    "layout",
    "station",
    "start_month",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# 掲載経過時間のカテゴリ
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
# 時間特徴量
# ============================================================

def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["time_band"] = pd.cut(
        result["time_day"],
        bins=TIME_BINS,
        labels=TIME_LABELS,
        right=True,
        include_lowest=True,
    )

    return result


# ============================================================
# モデル入力を安全な型へ変換
# ============================================================

def prepare_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    scikit-learnへ渡す前に型を統一する。

    数値：
        数値へ変換できないものはnp.nan。
        モデル内で中央値補完。

    カテゴリ：
        pd.NAを残さず「欠損」という明示的カテゴリにする。
    """

    x = (
        df[MODEL_FEATURES]
        .copy()
    )

    # ----------------------------------------
    # 数値
    # ----------------------------------------

    for column in NUMERIC_FEATURES:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        ).astype(float)

    # ----------------------------------------
    # カテゴリ
    # ----------------------------------------

    for column in CATEGORICAL_FEATURES:

        x[column] = (
            x[column]
            .astype("string")
            .fillna("欠損")
            .astype(str)
        )

    return x


# ============================================================
# モデル
# ============================================================

def build_model() -> Pipeline:
    """
    離散時間ハザードモデル。

    各person-period行について、

        その日に掲載終了するか

    をロジスティック回帰で推定する。
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
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
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logistic = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )

    model = Pipeline(
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

    return model


# ============================================================
# 検証用データ
# ============================================================

def make_validation_rows(
    validation: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:
    """
    landmark日まで掲載が続いている物件について、

    target = 1
        その後horizon日以内に掲載終了

    target = 0
        horizon終了より後まで掲載継続

    horizon終了前に右打ち切りになった場合は、
    正解不明なので評価対象から除外する。
    """

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
# 将来の日ごとの予測行
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

    future = add_time_features(
        future
    )

    return future


# ============================================================
# 今後horizon日以内の終了確率
# ============================================================

def predict_horizon_probability(
    model: Pipeline,
    listings: pd.DataFrame,
    landmark: int,
    horizon: int,
) -> np.ndarray:
    """
    各日のハザード h(t) を使い、

        1 - Π(1 - h(t))

    として、今後horizon日以内に
    掲載終了する確率を求める。
    """

    future = make_future_rows(
        listings=listings,
        landmark=landmark,
        horizon=horizon,
    )

    x_future = prepare_model_features(
        future
    )

    daily_hazard = (
        model.predict_proba(
            x_future
        )[:, 1]
    )

    future[
        "daily_hazard"
    ] = daily_hazard

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
# モデル評価
# ============================================================

def evaluate_model(
    model: Pipeline,
    validation: pd.DataFrame,
) -> pd.DataFrame:

    metric_rows = []
    prediction_rows = []

    for landmark in LANDMARK_DAYS:

        for horizon in HORIZONS:

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
                    "model": (
                        "discrete_time_logistic"
                    ),
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
                    "age",
                    "rent",
                    "area",
                    "walk_distance",
                    "layout",
                    "station",
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
                "prediction"
            ] = p

            prediction_rows.append(
                output
            )

    metrics = pd.DataFrame(
        metric_rows
    )

    save_csv(
        metrics,
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

    return metrics


# ============================================================
# Baselineと比較
# ============================================================

def compare_with_baseline(
    model_metrics: pd.DataFrame,
) -> None:

    if not BASELINE_METRICS_PATH.exists():

        print()
        print(
            "Baselineの評価結果が見つかりません。"
        )

        return

    baseline = pd.read_csv(
        BASELINE_METRICS_PATH,
        encoding="utf-8-sig",
    )

    baseline = baseline[
        [
            "baseline",
            "landmark_day",
            "horizon",
            "n",
            "predicted_rate",
            "actual_rate",
            "brier_score",
        ]
    ].copy()

    baseline = baseline.rename(
        columns={
            "baseline": "model",
        }
    )

    model_for_compare = (
        model_metrics[
            [
                "model",
                "landmark_day",
                "horizon",
                "n",
                "predicted_rate",
                "actual_rate",
                "brier_score",
            ]
        ]
        .copy()
    )

    comparison = pd.concat(
        [
            baseline,
            model_for_compare,
        ],
        ignore_index=True,
    )

    comparison = comparison.sort_values(
        [
            "landmark_day",
            "horizon",
            "brier_score",
        ]
    )

    save_csv(
        comparison,
        "comparison_with_baselines.csv",
    )

    print()
    print(
        "【2022年 Baselineとの比較】"
    )

    print(
        comparison.to_string(
            index=False
        )
    )


# ============================================================
# モデル入力の欠損状況
# ============================================================

def print_feature_summary(
    person_period: pd.DataFrame,
) -> None:

    print()
    print(
        "【学習特徴量の欠損状況】"
    )

    for column in MODEL_FEATURES:

        missing = int(
            person_period[column]
            .isna()
            .sum()
        )

        print(
            f"{column:15s}: "
            f"{missing:,}"
        )


# ============================================================
# main
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "国分寺市 離散時間生存モデル"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    raw = load_prepared(
        INPUT_PATH
    )

    survival = prepare_survival_data(
        raw
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
        f"学習物件数 2020～2021 : {len(train):,}"
    )

    print(
        f"検証物件数 2022       : {len(validation):,}"
    )

    print(
        "2023年は最終テスト用として使用しません。"
    )

    # --------------------------------------------------------
    # Person-period
    # --------------------------------------------------------

    print()
    print(
        "person-periodデータを作成しています..."
    )

    person_period = make_person_period(
        train
    )

    person_period = add_time_features(
        person_period
    )

    print(
        f"person-period行数 : {len(person_period):,}"
    )

    print(
        f"イベント行数      : "
        f"{person_period['event_in_day'].sum():,}"
    )

    print_feature_summary(
        person_period
    )

    # --------------------------------------------------------
    # scikit-learnへ渡す型へ変換
    # --------------------------------------------------------

    x_train = prepare_model_features(
        person_period
    )

    y_train = (
        person_period[
            "event_in_day"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # モデル
    # --------------------------------------------------------

    model = build_model()

    print()
    print(
        "離散時間生存モデルを学習しています..."
    )

    model.fit(
        x_train,
        y_train,
    )

    print(
        "学習完了"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        OUTPUT_DIR
        / "discrete_time_logistic.joblib",
    )

    # --------------------------------------------------------
    # 2022年評価
    # --------------------------------------------------------

    print()
    print(
        "2022年データで評価しています..."
    )

    metrics = evaluate_model(
        model=model,
        validation=validation,
    )

    print()
    print(
        "【2022年 離散時間生存モデル】"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    compare_with_baseline(
        metrics
    )

    print()
    print(
        f"出力先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
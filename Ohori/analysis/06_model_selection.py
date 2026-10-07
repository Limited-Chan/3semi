from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

BASELINE_METRICS_PATH = (
    OHORI_DIR
    / "results"
    / "04_baseline"
    / "validation_2022_metrics.csv"
)

OUTPUT_DIR = (
    OHORI_DIR
    / "results"
    / "06_model_selection"
)


# ============================================================
# 年
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
# Baseline読み込み
# ============================================================

def load_baselines() -> pd.DataFrame:

    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Baseline評価結果がありません。"
            "\n先に analysis/04_baseline.py を実行してください。"
        )

    baseline = pd.read_csv(
        BASELINE_METRICS_PATH,
        encoding="utf-8-sig",
    )

    baseline = baseline.rename(
        columns={
            "baseline": "model",
        }
    )

    baseline["roc_auc"] = np.nan

    return baseline[
        [
            "model",
            "landmark_day",
            "horizon",
            "n",
            "predicted_rate",
            "actual_rate",
            "brier_score",
            "roc_auc",
        ]
    ].copy()


# ============================================================
# モデル比較サマリー
# ============================================================

def make_model_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:

    d = comparison.copy()

    d["calibration_error"] = (
        d["predicted_rate"]
        - d["actual_rate"]
    ).abs()

    rows = []

    for model_name, group in d.groupby(
        "model"
    ):

        auc_values = (
            group["roc_auc"]
            .dropna()
        )

        rows.append(
            {
                "model": model_name,
                "評価条件数": len(group),
                "平均Brier": float(
                    group["brier_score"]
                    .mean()
                ),
                "Brier中央値": float(
                    group["brier_score"]
                    .median()
                ),
                "平均Calibration誤差": float(
                    group["calibration_error"]
                    .mean()
                ),
                "平均AUC": (
                    float(
                        auc_values.mean()
                    )
                    if len(auc_values) > 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return summary.sort_values(
        "平均Brier",
        ascending=True,
    )


# ============================================================
# Model 1 vs Model 2
# ============================================================

def select_discrete_model(
    summary: pd.DataFrame,
) -> str:

    candidates = summary[
        summary["model"].isin(
            [
                "model1_basic",
                "model2_age_layout",
            ]
        )
    ].copy()

    if len(candidates) != 2:
        raise ValueError(
            "Model 1 / Model 2 の評価結果が揃っていません。"
        )

    winner = (
        candidates
        .sort_values(
            "平均Brier",
            ascending=True,
        )
        .iloc[0]
    )

    return str(
        winner["model"]
    )


# ============================================================
# 各条件で誰が最良か
# ============================================================

def make_condition_winners(
    comparison: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = comparison.groupby(
        [
            "landmark_day",
            "horizon",
        ]
    )

    for (
        landmark,
        horizon,
    ), group in grouped:

        best = (
            group
            .sort_values(
                "brier_score",
                ascending=True,
            )
            .iloc[0]
        )

        rows.append(
            {
                "現在掲載日数": landmark,
                "予測期間": horizon,
                "最良モデル": best["model"],
                "最良Brier": best["brier_score"],
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
        "国分寺市 離散時間生存モデル選択"
    )

    print(
        "=" * 72
    )

    print(
        "選択基準："
        "2022年の9評価条件における平均Brier score"
    )

    print(
        "2023年は最終テスト用のため、この処理では使用しません。"
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

    print()
    print(
        f"学習 2020～2021 : {len(train):,}件"
    )

    print(
        f"検証 2022       : {len(validation):,}件"
    )

    # ========================================================
    # Model 1
    # ========================================================

    print()
    print(
        "Model 1：基本モデルを学習しています..."
    )

    model1, model1_pp_rows = fit_model(
        train,
        use_age_layout_combo=False,
    )

    print(
        f"Model 1 学習完了 "
        f"(person-period {model1_pp_rows:,}行)"
    )

    model1_metrics, model1_predictions = (
        evaluate_model(
            model=model1,
            validation=validation,
            model_name="model1_basic",
            landmarks=LANDMARK_DAYS,
            horizons=HORIZONS,
            use_age_layout_combo=False,
        )
    )

    # ========================================================
    # Model 2
    # ========================================================

    print()
    print(
        "Model 2：築年数帯×間取りを追加して学習しています..."
    )

    model2, model2_pp_rows = fit_model(
        train,
        use_age_layout_combo=True,
    )

    print(
        f"Model 2 学習完了 "
        f"(person-period {model2_pp_rows:,}行)"
    )

    model2_metrics, model2_predictions = (
        evaluate_model(
            model=model2,
            validation=validation,
            model_name="model2_age_layout",
            landmarks=LANDMARK_DAYS,
            horizons=HORIZONS,
            use_age_layout_combo=True,
        )
    )

    # ========================================================
    # 保存
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model1,
        OUTPUT_DIR
        / "model1_basic.joblib",
    )

    joblib.dump(
        model2,
        OUTPUT_DIR
        / "model2_age_layout.joblib",
    )

    save_csv(
        model1_metrics,
        "model1_validation_metrics.csv",
    )

    save_csv(
        model2_metrics,
        "model2_validation_metrics.csv",
    )

    save_csv(
        model1_predictions,
        "model1_validation_predictions.csv",
    )

    save_csv(
        model2_predictions,
        "model2_validation_predictions.csv",
    )

    # ========================================================
    # Baselineも含めて比較
    # ========================================================

    baseline = load_baselines()

    comparison = pd.concat(
        [
            baseline,
            model1_metrics,
            model2_metrics,
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
        "all_models_by_condition.csv",
    )

    # ========================================================
    # モデル単位の平均
    # ========================================================

    summary = make_model_summary(
        comparison
    )

    save_csv(
        summary,
        "model_summary.csv",
    )

    # ========================================================
    # Model 1 / Model 2 の採用判断
    # ========================================================

    selected_model = (
        select_discrete_model(
            summary
        )
    )

    # ========================================================
    # 条件ごとの勝者
    # ========================================================

    winners = (
        make_condition_winners(
            comparison
        )
    )

    save_csv(
        winners,
        "condition_winners.csv",
    )

    # ========================================================
    # 採用結果保存
    # ========================================================

    selection = pd.DataFrame(
        [
            {
                "selection_metric": (
                    "mean_brier_score_2022"
                ),
                "selected_model": (
                    selected_model
                ),
                "train_years": (
                    "2020,2021"
                ),
                "validation_year": (
                    2022
                ),
                "test_year": (
                    2023
                ),
                "test_used_for_selection": (
                    False
                ),
            }
        ]
    )

    save_csv(
        selection,
        "selected_model.csv",
    )

    # ========================================================
    # 表示
    # ========================================================

    print()
    print(
        "【モデル別サマリー】"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "【各条件で最もBrier scoreが小さいモデル】"
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
        f"採用する離散時間生存モデル：{selected_model}"
    )

    print(
        "選択には2023年データを使用していません。"
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
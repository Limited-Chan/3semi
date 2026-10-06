from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 共通設定
# ============================================================

# 今後の分析で共通して使う元データ列
SOURCE_COLUMNS = [
    "id",
    "building_id",
    "unit_id",
    "pub_start_date",
    "pub_end_date",
    "addr1_1_name",
    "addr1_2_name",
    "bukken_type",

    # 築年数
    "kenchiku_date",

    # 間取り
    "madori_number_all",
    "madori_kind_all_label",

    # 家賃・面積
    "money_room",
    "house_area",

    # 駅関連
    "walk_distance1",
    "eki1_name",
    "rosen1_name",
]


VALID_LAYOUT_KINDS = [
    "R",
    "K",
    "DK",
    "LK",
    "LDK",
    "SK",
    "SDK",
    "SLDK",
]


# ============================================================
# 基本変換
# ============================================================

def normalize_text(series: pd.Series) -> pd.Series:
    """
    文字列を共通形式にそろえる。
    """

    return (
        series.astype("string")
        .fillna("")
        .str.normalize("NFKC")
        .str.strip()
    )


def numeric(series: pd.Series) -> pd.Series:
    """
    数値へ変換する。
    数値として扱えない値は欠損値 NaN にする。
    """

    return pd.to_numeric(
        normalize_text(series).str.replace(",", "", regex=False),
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================
# 共通前処理
# ============================================================

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """
    EDA、Kaplan-Meier、生存モデルで共通して使う前処理。

    この関数では、
    ・中央値を使う
    ・Kaplan-Meierを使う
    ・特定の属性をモデルから除外する

    といった分析手法固有の判断は行わない。
    """

    d = raw.copy()

    # --------------------------------------------------------
    # 必要列の確認
    # --------------------------------------------------------

    # 元データに存在しない任意列があっても
    # 前処理自体は実行できるように空列を作る
    for column in SOURCE_COLUMNS:
        if column not in d.columns:
            d[column] = ""

    # 文字列を正規化
    for column in SOURCE_COLUMNS:
        d[column] = normalize_text(d[column])

    # --------------------------------------------------------
    # 掲載開始日・掲載終了日
    # --------------------------------------------------------

    d["start_date"] = pd.to_datetime(
        d["pub_start_date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    d["end_date"] = pd.to_datetime(
        d["pub_end_date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # 記録された掲載期間
    d["duration"] = (
        d["end_date"] - d["start_date"]
    ).dt.days.astype(float)

    d["start_year"] = d["start_date"].dt.year
    d["start_month"] = d["start_date"].dt.month

    d["valid_start_date"] = d["start_date"].notna()
    d["valid_end_date"] = d["end_date"].notna()

    # EDAで従来使用していた
    # 「開始日・終了日があり、掲載期間が0日以上」
    d["valid_complete_duration"] = (
        d["valid_start_date"]
        & d["valid_end_date"]
        & d["duration"].ge(0)
    )

    # --------------------------------------------------------
    # 家賃
    # --------------------------------------------------------

    rent = numeric(d["money_room"])

    d["rent"] = rent.where(
        rent.gt(0)
    )

    d["rent_missing"] = rent.isna()

    # --------------------------------------------------------
    # 面積
    # --------------------------------------------------------

    area = numeric(d["house_area"])

    d["area"] = area.where(
        area.gt(0)
    )

    d["area_missing"] = area.isna()

    # --------------------------------------------------------
    # 駅までの距離
    # --------------------------------------------------------

    walk_distance = numeric(
        d["walk_distance1"]
    )

    d["walk_distance"] = walk_distance.where(
        walk_distance.ge(0)
    )

    d["walk_distance_missing"] = (
        walk_distance.isna()
    )

    # --------------------------------------------------------
    # 駅・路線
    # --------------------------------------------------------

    d["station"] = (
        normalize_text(d["eki1_name"])
        .replace("", pd.NA)
    )

    d["railway"] = (
        normalize_text(d["rosen1_name"])
        .replace("", pd.NA)
    )

    # --------------------------------------------------------
    # 築年数
    # --------------------------------------------------------

    build_text = (
        d["kenchiku_date"]
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    valid_build_format = (
        build_text
        .str.fullmatch(r"\d{6}")
        .fillna(False)
    )

    build_date = pd.to_datetime(
        build_text.where(
            valid_build_format
        ),
        format="%Y%m",
        errors="coerce",
    )

    age_months = (
        (
            d["start_date"].dt.year
            - build_date.dt.year
        )
        * 12
        + d["start_date"].dt.month
        - build_date.dt.month
    )

    # 掲載開始時点での築年数
    d["age"] = (
        age_months / 12
    ).where(
        age_months.ge(0)
    )

    d["age_missing"] = (
        build_date.isna()
    )

    d["age_build_after_listing"] = (
        age_months.lt(0)
    )

    # --------------------------------------------------------
    # 間取り
    # --------------------------------------------------------

    rooms = numeric(
        d["madori_number_all"]
    )

    kinds = (
        normalize_text(
            d["madori_kind_all_label"]
        )
        .str.upper()
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )

    valid_rooms = (
        rooms.notna()
        & rooms.ge(1)
        & rooms.mod(1).eq(0)
    )

    valid_kind = (
        kinds.isin(
            VALID_LAYOUT_KINDS
        )
    )

    valid_layout = (
        valid_rooms
        & valid_kind
    )

    d["layout"] = pd.Series(
        pd.NA,
        index=d.index,
        dtype="string",
    )

    d.loc[
        valid_layout,
        "layout",
    ] = (
        rooms.loc[valid_layout]
        .astype(int)
        .astype(str)
        + kinds.loc[valid_layout]
    )

    d["layout_missing"] = (
        ~valid_layout
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    d["missing_id"] = (
        d["id"].eq("")
    )

    d["repeated_id"] = (
        ~d["missing_id"]
        & d["id"].duplicated(
            keep=False
        )
    )

    return d


# ============================================================
# データ読み込み
# ============================================================

def load_prepared(
    path: str | Path,
) -> pd.DataFrame:
    """
    国分寺市抽出済みデータを読み込み、
    共通前処理を行う。

    対応形式：
    ・CSV
    ・CSV.GZ
    ・Parquet
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {path}"
        )

    if path.suffix.lower() == ".parquet":

        raw = pd.read_parquet(
            path
        )

    else:

        raw = pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )

    if raw.empty:
        raise ValueError(
            "入力データが0件です。"
        )

    return prepare(raw)


# ============================================================
# EDA用データ範囲
# ============================================================

def select_eda_scope(
    d: pd.DataFrame,
    years: list[int] | None = None,
    keep_duplicates: bool = True,
) -> pd.DataFrame:
    """
    探索的分析（EDA）で使用するデータを選択する。

    Kaplan-Meierや生存モデルでは、
    右打ち切りの扱いを確認したあと、
    別の対象定義を作成する。

    したがって、生存分析用データを
    この関数で作成してはいけない。
    """

    mask = (
        d["valid_complete_duration"]
        .copy()
    )

    # 掲載開始年を指定
    if years is not None:

        mask &= (
            d["start_year"]
            .isin(years)
        )

    # 重複IDを除外する場合のみ適用
    if not keep_duplicates:

        mask &= (
            ~d["missing_id"]
            & ~d["repeated_id"]
        )

    result = (
        d.loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    return result
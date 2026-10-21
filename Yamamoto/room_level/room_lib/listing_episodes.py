"""同一住戸の連続・重複掲載を一つの掲載エピソードへ統合する(修正版)。

建物IDと住戸IDの双方がある場合だけ住戸同定の候補とする。

統合の手順:
1. 同じ住戸キーの掲載を開始日順に並べ、時間的に重なる/max_gap_days以内に続く掲載を
   一つの「セグメント」にまとめる。
2. セグメントごとに、間取り・面積の矛盾を確認する。矛盾があるセグメントだけ統合せず、
   監査表に残す(住戸の全履歴ではなく、つなごうとしている掲載どうしだけで判定する)。
3. 最寄駅の違いは、主駅の選び方が掲載ごとに変わりうる弱い証拠なので、記録のみで
   統合は止めない。

掲載終了は成約・空室解消を意味しない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_IDS = {"", "0", "nan", "none", "null", "<na>", "-", "不明"}
AREA_TOLERANCE_SQM = 2.0


def _valid_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    result = str(value).strip()
    return None if result.lower() in MISSING_IDS else result


def segment_conflicts(segment: pd.DataFrame, area_tolerance: float = AREA_TOLERANCE_SQM) -> dict:
    """統合しようとする掲載どうしの情報の食い違いを調べる。"""
    layout_conflict = bool(
        "layout" in segment.columns and segment["layout"].dropna().astype(str).nunique() > 1
    )
    area_conflict = False
    if "area" in segment.columns:
        areas = pd.to_numeric(segment["area"], errors="coerce").dropna()
        area_conflict = bool(len(areas) and areas.max() - areas.min() > area_tolerance)
    station_differs = bool(
        "station" in segment.columns and segment["station"].dropna().astype(str).nunique() > 1
    )
    return {
        "layout_conflict": layout_conflict,
        "area_conflict": area_conflict,
        "station_differs": station_differs,
    }


def prepare_room_keys(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy().reset_index(drop=True)
    out["source_row"] = np.arange(len(out))
    for c in ("building_id", "unit_id"):
        if c not in out:
            out[c] = ""
    building = out["building_id"].map(_valid_identifier)
    unit = out["unit_id"].map(_valid_identifier)
    both = building.notna() & unit.notna()
    out["unit_key"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[both, "unit_key"] = building[both] + "::" + unit[both]
    out["has_unit_key"] = out["unit_key"].notna()
    return out


def build_episodes(
    data: pd.DataFrame,
    max_gap_days: int = 1,
    area_tolerance: float = AREA_TOLERANCE_SQM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(エピソード表, セグメント監査表)を返す。

    max_gap_days=0 は日付範囲が重なる掲載のみを結合する。
    max_gap_days=1 は前の終了日の翌日に始まる掲載も結合する。
    7/14 は再掲載の扱いに対する感度分析用であり、正解設定ではない。
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days は0以上にしてください")
    required = {"start_date", "end_date", "start_year", "start_month"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"欠けている列: {sorted(missing)}")
    d = prepare_room_keys(data)
    d = d.loc[
        d["start_date"].notna() & d["end_date"].notna()
        & (d["end_date"] >= d["start_date"])
    ].copy()
    if d.empty:
        raise ValueError("有効な開始日・終了日を持つ掲載がありません")

    # ID欠損の掲載は、仮の同一住戸キーを作らず必ず別掲載として残す。
    d["_group_key"] = np.where(
        d["has_unit_key"],
        "unit:" + d["unit_key"].fillna(""),
        "unknown:" + d["source_row"].astype(str),
    )
    entries: list[pd.Series] = []
    audit_rows: list[dict] = []
    next_id = 0

    def emit(group: pd.DataFrame, status: str, unit_key: str | None) -> None:
        nonlocal next_id
        selected = group.sort_values(["start_date", "end_date", "source_row"]).iloc[0].copy()
        start = group["start_date"].min()
        end = group["end_date"].max()
        next_id += 1
        selected["first_source_id"] = selected.get("id", pd.NA)
        selected["id"] = f"episode_{next_id:07d}"
        selected["start_date"] = start
        selected["end_date"] = end
        selected["duration"] = float((end - start).days)
        selected["start_year"] = int(start.year)
        selected["start_month"] = int(start.month)
        selected["source_rows"] = len(group)
        selected["episode_status"] = status
        selected["unit_key"] = unit_key if unit_key is not None else pd.NA
        entries.append(selected)

    def audit(group_key: str, key: str | None, number: int, rows: pd.DataFrame,
              flags: dict, available: bool, merged: bool) -> None:
        conflict = len(rows) > 1 and (flags["layout_conflict"] or flags["area_conflict"])
        audit_rows.append({
            "group_key": group_key,
            "unit_key": key,
            "segment_no": number,
            "source_rows": len(rows),
            "unique_listing_ids": rows["id"].astype(str).nunique() if "id" in rows else np.nan,
            "id_available": available,
            "layout_conflict": flags["layout_conflict"] if len(rows) > 1 else False,
            "area_conflict": flags["area_conflict"] if len(rows) > 1 else False,
            "station_differs": flags["station_differs"] if len(rows) > 1 else False,
            "metadata_conflict": bool(conflict),
            "merged": merged,
            "start_min": rows["start_date"].min(),
            "end_max": rows["end_date"].max(),
        })

    no_flags = {"layout_conflict": False, "area_conflict": False, "station_differs": False}
    for group_key, group in d.groupby("_group_key", sort=False):
        group = group.sort_values(["start_date", "end_date", "source_row"])
        if not bool(group["has_unit_key"].iloc[0]):
            for _, record in group.iterrows():
                single = group.loc[[record.name]]
                audit(group_key, None, 0, single, no_flags, False, False)
                emit(single, "missing_id", None)
            continue

        key = str(group["unit_key"].iloc[0])
        segments: list[list[int]] = []
        segment: list[int] = []
        current_end = pd.Timestamp.min
        for idx, row in group.iterrows():
            if segment and row["start_date"] > current_end + pd.Timedelta(days=max_gap_days):
                segments.append(segment)
                segment = []
                current_end = pd.Timestamp.min
            segment.append(idx)
            current_end = max(current_end, row["end_date"])
        if segment:
            segments.append(segment)

        for number, members in enumerate(segments, start=1):
            rows = group.loc[members]
            flags = segment_conflicts(rows, area_tolerance)
            blocked = len(rows) > 1 and (flags["layout_conflict"] or flags["area_conflict"])
            audit(group_key, key, number, rows, flags, True, merged=(len(rows) > 1 and not blocked))
            if blocked:
                for _, record in rows.iterrows():
                    emit(rows.loc[[record.name]], "conflicting_metadata", key)
            else:
                emit(rows, "identified_unit", key)

    episodes = pd.DataFrame(entries).reset_index(drop=True)
    episodes = episodes.drop(columns=["source_row", "_group_key", "has_unit_key"], errors="ignore")
    return episodes, pd.DataFrame(audit_rows)


def build_episodes_fast(
    data: pd.DataFrame,
    max_gap_days: int = 1,
    area_tolerance: float = AREA_TOLERANCE_SQM,
) -> pd.DataFrame:
    """build_episodes と同じ統合ルールを、ベクトル演算で高速に行う(エピソード表だけを返す)。

    数十万件の掲載でも数秒で終わる。結果は build_episodes と同じ(テストで照合)。
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days は0以上にしてください")
    d = prepare_room_keys(data)
    d = d.loc[
        d["start_date"].notna() & d["end_date"].notna() & (d["end_date"] >= d["start_date"])
    ].copy()
    d["_group_key"] = np.where(
        d["has_unit_key"], "unit:" + d["unit_key"].fillna(""), "unknown:" + d["source_row"].astype(str)
    )
    d = d.sort_values(["_group_key", "start_date", "end_date", "source_row"]).reset_index(drop=True)
    by_group = d.groupby("_group_key", sort=False)
    prev_end = by_group["end_date"].cummax().groupby(d["_group_key"], sort=False).shift(1)
    new_segment = prev_end.isna() | ((d["start_date"] - prev_end) > pd.Timedelta(days=max_gap_days))
    d["_segment"] = new_segment.cumsum()
    seg = d.groupby("_segment")
    size = seg["source_row"].transform("size")
    layout_kinds = seg["layout"].transform("nunique") if "layout" in d.columns else 0
    area = (pd.to_numeric(d["area"], errors="coerce") if "area" in d.columns else pd.Series(np.nan, index=d.index)).astype(float)
    area_range = area.groupby(d["_segment"]).transform("max") - area.groupby(d["_segment"]).transform("min")
    blocked = (d["has_unit_key"] & (size > 1) & ((layout_kinds > 1) | (area_range > area_tolerance))).fillna(False).astype(bool)
    d["_episode"] = np.where(blocked | ~d["has_unit_key"], -(np.arange(len(d)) + 1), d["_segment"])
    ep = d.groupby("_episode")
    first = d.loc[~d.duplicated("_episode", keep="first")].copy()
    first["first_source_id"] = first["id"] if "id" in first else pd.NA
    start = ep["start_date"].min()
    end = ep["end_date"].max()
    first["start_date"] = first["_episode"].map(start)
    first["end_date"] = first["_episode"].map(end)
    first["duration"] = (first["end_date"] - first["start_date"]).dt.days.astype(float)
    first["start_year"] = first["start_date"].dt.year.astype(int)
    first["start_month"] = first["start_date"].dt.month.astype(int)
    first["source_rows"] = first["_episode"].map(ep.size())
    first["episode_status"] = np.where(
        ~first["has_unit_key"], "missing_id", np.where(blocked.loc[first.index], "conflicting_metadata", "identified_unit")
    )
    first["unit_key"] = first["unit_key"].where(first["has_unit_key"], pd.NA)
    first["id"] = [f"episode_{i:07d}" for i in range(1, len(first) + 1)]
    return first.drop(columns=["source_row", "_group_key", "has_unit_key", "_segment", "_episode"]).reset_index(drop=True)

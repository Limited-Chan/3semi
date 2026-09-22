#!/usr/bin/env python3
"""Two-stage exploratory analysis for Kokubunji rental listing records.

Stage 1 (inspect): inspect distributions and compare mean/median/mode.
Stage 2 (relate): after the analyst explicitly chooses a representative statistic,
                  examine one-to-one relations and then a two-variable comparison.

The script deliberately avoids causal claims, p-values, arbitrary outlier removal,
implicit property-type filtering, and automatic representative-value selection.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "2.3.0"

# Only fields used for the planned analysis or for transparent data-quality audits.
SOURCE_COLUMNS = [
    "id", "building_id", "unit_id",
    "pub_start_date", "pub_end_date",
    "addr1_1_name", "addr1_2_name", "bukken_type",
    "kenchiku_date", "flg_new",
    "madori_number_all", "madori_kind_all", "madori_kind_all_label", "madori_code",
    "money_room", "money_kyoueki", "money_combo", "house_area",
    "genkyo_label", "usable_status_label", "usable_date",
    "confirm_date", "keiyaku_date",
]
REQUIRED_EXTRACT_COLUMNS = {
    "id", "pub_start_date", "pub_end_date", "addr1_1_name", "addr1_2_name"
}

VALID_LAYOUT_KINDS = ["R", "K", "DK", "LK", "LDK", "SK", "SDK", "SLDK"]

# Equal-width descriptive bins. These are visualization/grouping choices, not statistical thresholds.
# The chosen widths are explicit, configurable in Step 1, and stored in scope.json so Step 2
# necessarily reuses the same horizontal-axis definitions.
DEFAULT_BAND_CONFIG = {
    # step = statistical grouping width. bins_per_figure = display-only page width.
    # No overflow category is ever created.
    "duration": {"step": 30.0, "bins_per_figure": 12, "unit": "days"},
    "rent": {"step": 20000.0, "bins_per_figure": 10, "unit": "yen"},
    "area": {"step": 10.0, "bins_per_figure": 10, "unit": "sqm"},
    "age": {"step": 5.0, "bins_per_figure": 10, "unit": "years"},
}


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.normalize("NFKC").str.strip()


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(normalize_text(s).str.replace(",", "", regex=False), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _fmt_num(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


def normalize_band_config(config: dict | None = None) -> dict:
    base = json.loads(json.dumps(DEFAULT_BAND_CONFIG))
    if config:
        for key in base:
            if key in config:
                base[key].update(config[key])
    for key, spec in base.items():
        step = float(spec["step"])
        bins_per_figure = int(spec["bins_per_figure"])
        if step <= 0:
            raise ValueError(f"Invalid step for {key}: {step}")
        if bins_per_figure <= 0:
            raise ValueError(f"Invalid bins_per_figure for {key}: {bins_per_figure}")
        spec["step"] = step
        spec["bins_per_figure"] = bins_per_figure
    return base


def _full_equal_width_schema(values: pd.Series, step: float, *, scale: float = 1.0, suffix: str = "") -> dict:
    """Build finite, equal-width bins covering every observed nonnegative value.

    No top-coded/overflow bin is created. If the maximum falls exactly on a bin edge,
    one additional bin is created so right=False still contains the maximum.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[x >= 0]
    if x.empty:
        return {"step": float(step), "edges": [0.0, float(step)], "labels": [f"0-{_fmt_num(step/scale)}{suffix}"], "max_observed": None}
    max_observed = float(x.max())
    upper = (np.floor(max_observed / step) + 1.0) * step
    if upper <= 0:
        upper = step
    edges = np.arange(0.0, upper + step * 0.5, step, dtype=float)
    # numerical guard
    if edges[-1] <= max_observed:
        edges = np.append(edges, edges[-1] + step)
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        labels.append(f"{_fmt_num(a/scale)}-{_fmt_num(b/scale)}{suffix}")
    return {"step": float(step), "edges": edges.tolist(), "labels": labels, "max_observed": max_observed}


def build_band_schema(primary: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = normalize_band_config(config)
    schema = {
        "duration": {**cfg["duration"], **_full_equal_width_schema(primary["duration"], cfg["duration"]["step"], suffix="d")},
        "rent": {**cfg["rent"], **_full_equal_width_schema(primary["rent"], cfg["rent"]["step"], scale=1000.0, suffix="k")},
        "area": {**cfg["area"], **_full_equal_width_schema(primary["area"], cfg["area"]["step"], suffix="m2")},
        "age": {**cfg["age"], **_full_equal_width_schema(primary["age"], cfg["age"]["step"], suffix="y")},
    }
    return schema


def apply_band_schema(d: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = d.copy()
    mapping = {
        "duration": ("duration", "duration_band"),
        "rent": ("rent", "rent_band"),
        "area": ("area", "area_band"),
        "age": ("age", "age_band"),
    }
    for key, (value_col, band_col) in mapping.items():
        spec = schema[key]
        out[band_col] = pd.cut(
            out[value_col], spec["edges"], labels=spec["labels"],
            right=False, include_lowest=True
        )
    return out


def apply_equal_width_bands(d: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compatibility helper used by tests/direct calls; derives full-range bins from d itself."""
    return apply_band_schema(d, build_band_schema(d, config))


def band_config_from_args(args) -> dict:
    return normalize_band_config({
        "duration": {"step": args.duration_step_days, "bins_per_figure": args.duration_bins_per_figure},
        "rent": {"step": args.rent_step_yen, "bins_per_figure": args.rent_bins_per_figure},
        "area": {"step": args.area_step_sqm, "bins_per_figure": args.area_bins_per_figure},
        "age": {"step": args.age_step_years, "bins_per_figure": args.age_bins_per_figure},
    })

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    for c in SOURCE_COLUMNS:
        if c not in d:
            d[c] = ""
        d[c] = normalize_text(d[c])

    d["start"] = pd.to_datetime(d["pub_start_date"], format="%Y-%m-%d", errors="coerce")
    d["end"] = pd.to_datetime(d["pub_end_date"], format="%Y-%m-%d", errors="coerce")
    d["duration"] = (d["end"] - d["start"]).dt.days.astype(float)
    d["start_year"] = d["start"].dt.year
    d["valid_dates"] = d["start"].notna() & d["end"].notna() & d["duration"].ge(0)

    d["missing_id"] = d["id"].eq("")
    d["repeated_id"] = ~d["missing_id"] & d["id"].duplicated(keep=False)

    rent_raw = numeric(d["money_room"])
    area_raw = numeric(d["house_area"])
    d["rent"] = rent_raw.where(rent_raw.gt(0))
    d["area"] = area_raw.where(area_raw.gt(0))
    d["rent_missing_or_unparseable"] = rent_raw.isna()
    d["rent_nonpositive"] = rent_raw.notna() & rent_raw.le(0)
    d["area_missing_or_unparseable"] = area_raw.isna()
    d["area_nonpositive"] = area_raw.notna() & area_raw.le(0)

    build_text = d["kenchiku_date"].str.replace(r"\.0$", "", regex=True)
    valid_build_format = build_text.str.fullmatch(r"\d{6}").fillna(False)
    build = pd.to_datetime(build_text.where(valid_build_format), format="%Y%m", errors="coerce")
    age_months = (d["start"].dt.year - build.dt.year) * 12 + d["start"].dt.month - build.dt.month
    d["age_build_after_listing"] = age_months.lt(0)
    d["age_missing_or_unparseable"] = build.isna()
    d["age"] = (age_months / 12).where(age_months.ge(0))

    rooms = numeric(d["madori_number_all"])
    valid_rooms = rooms.notna() & rooms.ge(1) & rooms.mod(1).eq(0)
    kinds = d["madori_kind_all_label"].str.upper().str.replace(" ", "", regex=False)
    valid_kinds = kinds.isin(VALID_LAYOUT_KINDS)
    d["layout"] = pd.Series(pd.NA, index=d.index, dtype="string")
    layout_ok = valid_rooms & valid_kinds
    d.loc[layout_ok, "layout"] = rooms.loc[layout_ok].astype("int64").astype(str) + kinds.loc[layout_ok]
    d["layout_invalid"] = ~layout_ok
    d["layout_invalid_reason"] = pd.Series("", index=d.index, dtype="string")
    d.loc[~valid_rooms, "layout_invalid_reason"] = "missing_or_noninteger_room_count"
    d.loc[valid_rooms & ~valid_kinds, "layout_invalid_reason"] = "missing_or_unsupported_layout_kind"

    return d


def parse_years(value: str):
    if value.lower() == "all":
        return None
    years = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not years:
        raise ValueError("--years must be 'all' or a comma-separated year list")
    return years


def parse_types(value: str):
    if value.lower() == "all":
        return None
    codes = [v.strip() for v in value.split(",") if v.strip()]
    if not codes:
        raise ValueError("--types must be 'all' or a comma-separated code list")
    return codes


def apply_scope(d: pd.DataFrame, years, types, duplicates: str):
    mask = pd.Series(True, index=d.index)
    stages = []

    def stage(name: str, condition: pd.Series):
        nonlocal mask
        before = int(mask.sum())
        mask &= condition.fillna(False)
        stages.append({"step": name, "before": before, "excluded": before - int(mask.sum()), "remaining": int(mask.sum())})

    stage("valid start/end and nonnegative recorded date difference", d["valid_dates"])
    if years is not None:
        stage("explicitly selected listing-start years", d["start_year"].isin(years))
    if types is not None:
        stage("explicitly selected property-type codes", d["bukken_type"].isin(types))
    if duplicates == "exclude":
        stage("explicitly exclude missing/repeated listing IDs", ~d["missing_id"] & ~d["repeated_id"])

    return d.loc[mask].copy(), pd.DataFrame(stages)


def exact_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    modes = x.mode().sort_values()
    counts = x.value_counts()
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "mode_values": "|".join(f"{v:g}" for v in modes.tolist()),
        "mode_tie_count": int(len(modes)),
        "mode_frequency": int(counts.iloc[0]),
        "min": float(x.min()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def selected_stat_value(s: pd.Series, stat: str):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan, "no_valid_values"
    if stat == "mean":
        return float(x.mean()), ""
    if stat == "median":
        return float(x.median()), ""
    modes = x.mode().sort_values()
    if len(modes) != 1:
        return np.nan, f"mode_tie:{'|'.join(f'{v:g}' for v in modes.tolist())}"
    return float(modes.iloc[0]), ""


def categorical_counts(d: pd.DataFrame, column: str) -> pd.DataFrame:
    s = d[column]
    if isinstance(s.dtype, pd.CategoricalDtype):
        counts = s.value_counts(sort=False, dropna=False)
        # Drop the missing pseudo-category from the primary distribution table.
        rows = []
        for cat in s.cat.categories:
            rows.append({column: str(cat), "n": int(counts.get(cat, 0))})
        return pd.DataFrame(rows)
    return s.dropna().value_counts().rename_axis(column).reset_index(name="n")


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def split_bar_counts(d: pd.DataFrame, column: str, folder: Path, stem: str,
                     title: str, xlabel: str, bins_per_figure: int, horizontal=False) -> None:
    """Save one complete CSV; split only the PNG presentation into fixed numbers of bins."""
    folder.mkdir(parents=True, exist_ok=True)
    tab = categorical_counts(d, column)
    save_csv(tab, folder / f"{stem}.csv")
    if tab.empty:
        return

    if horizontal:  # nominal layout categories: one readable figure, no artificial numeric paging.
        fig_h = max(4.5, len(tab) * 0.30)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(tab[column].astype(str), tab["n"])
        ax.set_xlabel("Record count")
        ax.set_ylabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(folder / f"{stem}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    global_ymax = max(1, int(tab["n"].max()))
    for page_no, (a, b) in enumerate(_chunks(len(tab), bins_per_figure), start=1):
        part = tab.iloc[a:b].copy()
        first = str(part[column].iloc[0])
        last = str(part[column].iloc[-1])
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(part[column].astype(str), part["n"])
        ax.set_ylabel("Record count")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, global_ymax * 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"{title}  [{first} ... {last}]")
        fig.tight_layout()
        fig.savefig(folder / f"{stem}_part{page_no:02d}_{_safe_name(first)}__{_safe_name(last)}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def representative_outputs(primary: pd.DataFrame, folder: Path) -> None:
    stats = exact_stats(primary["duration"])
    table = pd.DataFrame([stats])
    save_csv(table, folder / "duration_mean_median_mode.csv")

    labels = ["Mean", "Median"]
    values = [stats["mean"], stats["median"]]
    if stats["mode_tie_count"] == 1:
        labels.append("Mode")
        values.append(float(stats["mode_values"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Recorded date difference (days)")
    ax.set_title("Mean / median / exact mode of recorded date difference")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")
    if stats["mode_tie_count"] != 1:
        ax.text(0.02, 0.97, f"Mode is tied: {stats['mode_values']}", transform=ax.transAxes, va="top")
    else:
        ax.text(0.02, 0.97, f"Mode frequency: {stats['mode_frequency']:,}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(folder / "duration_mean_median_mode.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def data_quality_tables(d: pd.DataFrame, primary: pd.DataFrame, folder: Path) -> None:
    flags = [
        "rent_missing_or_unparseable", "rent_nonpositive",
        "area_missing_or_unparseable", "area_nonpositive",
        "age_build_after_listing", "age_missing_or_unparseable",
        "layout_invalid", "missing_id", "repeated_id",
    ]
    save_csv(
        pd.DataFrame([
            {"flag": f, "regional_n": int(d[f].sum()), "analysis_scope_n": int(primary[f].sum())}
            for f in flags
        ]),
        folder / "quality_flags.csv",
    )

    for col, name in [
        ("bukken_type", "property_type_counts.csv"),
        ("start_year", "start_year_counts.csv"),
        ("pub_end_date", "end_date_counts.csv"),
        ("madori_kind_all_label", "layout_kind_label_counts.csv"),
    ]:
        tab = d[col].replace("", pd.NA).value_counts(dropna=False).rename_axis(col).reset_index(name="n")
        save_csv(tab, folder / name)

    invalid = d.loc[d["layout_invalid"], "layout_invalid_reason"].value_counts(dropna=False).rename_axis("reason").reset_index(name="n")
    save_csv(invalid, folder / "layout_invalid_reasons.csv")

    # Audit construction dates recorded after listing start without assigning a fabricated age.
    future = primary.loc[primary["age_build_after_listing"]].copy()
    cols = [c for c in ["flg_new", "genkyo_label", "usable_status_label"] if c in future.columns]
    if len(future) and cols:
        summary = future.groupby(cols, dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
        save_csv(summary, folder / "build_date_after_listing_context.csv")


def inspect(args) -> None:
    input_path = Path(args.input).resolve()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    for sub in ["00_scope", "01_distribution", "02_representative", "audit"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError("Input contains zero rows")
    if "source_row" not in raw:
        raw["source_row"] = np.arange(2, len(raw) + 2)
    d = prepare(raw)

    years = parse_years(args.years)
    types = parse_types(args.types)
    primary, flow = apply_scope(d, years, types, args.duplicates)
    if primary.empty:
        raise ValueError("No records remain after the explicitly requested scope filters")

    band_config = band_config_from_args(args)
    band_schema = build_band_schema(primary, band_config)
    primary = apply_band_schema(primary, band_schema)

    save_csv(flow, out / "00_scope" / "sample_flow.csv")
    data_quality_tables(d, primary, out / "audit")

    # Requirement 1: equal-width statistics across the entire observed range.
    # Only the image is paged; the underlying grouping is never top-coded.
    split_bar_counts(primary, "duration_band", out / "01_distribution" / "duration", "duration_distribution_bar",
                     "Recorded date difference: distribution", "Days (equal-width bands)", band_schema["duration"]["bins_per_figure"])
    split_bar_counts(primary, "rent_band", out / "01_distribution" / "rent", "money_room_distribution_bar",
                     "money_room: equal-width distribution", "money_room (equal-width bands)", band_schema["rent"]["bins_per_figure"])
    split_bar_counts(primary, "area_band", out / "01_distribution" / "area", "house_area_distribution_bar",
                     "house_area: equal-width distribution", "house_area (equal-width bands)", band_schema["area"]["bins_per_figure"])
    split_bar_counts(primary, "age_band", out / "01_distribution" / "age", "age_distribution_bar",
                     "Derived age at listing start: equal-width distribution", "Age in years (equal-width bands)", band_schema["age"]["bins_per_figure"])
    split_bar_counts(primary, "layout", out / "01_distribution" / "layout", "layout_distribution_bar",
                     "Layout: distribution", "Layout", 999999, horizontal=True)

    representative_outputs(primary, out / "02_representative")
    descriptive = []
    for field in ["duration", "rent", "area", "age"]:
        descriptive.append({"variable": field, "missing_n": int(len(primary) - primary[field].notna().sum()), **exact_stats(primary[field])})
    save_csv(pd.DataFrame(descriptive), out / "02_representative" / "descriptive_statistics.csv")

    input_hash = sha256_file(input_path)
    scope = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path_name": input_path.name,
        "input_sha256": input_hash,
        "regional_n": int(len(d)),
        "analysis_scope_n": int(len(primary)),
        "years": "all" if years is None else years,
        "property_types": "all" if types is None else types,
        "duplicates": args.duplicates,
        "duration_definition": "pub_end_date - pub_start_date in calendar days; zero retained",
        "rent_field": "money_room",
        "area_field": "house_area",
        "age_definition": "month difference from kenchiku_date YYYYMM to listing-start month / 12; build date after listing -> numeric age missing",
        "layout_definition": "madori_number_all + recognized madori_kind_all_label",
        "band_config": band_config,
        "band_schema": band_schema,
        "band_schema": band_schema,
        "band_note": "Every numeric group is finite and equal-width from zero through the observed maximum. No overflow/top-coded category exists. CSV keeps the full range; PNGs are display-only pages.",
        "automatic_representative_choice": False,
    }
    save_json(out / "scope.json", scope)
    save_json(out / "00_scope" / "band_config.json", band_config)
    save_json(out / "00_scope" / "band_schema.json", band_schema)

    report = "# Step 1: distribution inspection\n\n"
    report += f"Analysis scope: **{len(primary):,} listing records**.\n\n"
    report += "Numeric bins are equal-width across the complete observed range. No overflow category is used.\n"
    report += "Each variable has one complete CSV; PNGs are split into fixed numbers of adjacent bins only for readability.\n\n"
    report += "The exact steps, full finite edges, and page sizes are saved in `00_scope/band_schema.json`.\n"
    report += "The program does not choose mean/median/mode automatically.\n"
    (out / "INSPECT_REPORT.md").write_text(report, encoding="utf-8")

    print(f"Step 1 complete: {len(primary):,} records")
    print(f"Open: {out / 'INSPECT_REPORT.md'}")
    print("STOP here and choose mean / median / mode before running 'relate'.")

def group_statistics(d: pd.DataFrame, group_col: str, stat: str) -> pd.DataFrame:
    rows = []
    s = d[group_col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        # Preserve every numeric interval on the x-axis, even when a bin has n=0.
        # This prevents visual compression of equal-width intervals.
        iterator = [(cat, d.loc[s.eq(cat)]) for cat in s.cat.categories]
    else:
        iterator = d.dropna(subset=[group_col]).groupby(group_col, observed=True, sort=True)
    for key, g in iterator:
        base = exact_stats(g["duration"])
        selected, issue = selected_stat_value(g["duration"], stat)
        rows.append({group_col: str(key), **base, "selected_stat": stat, "selected_value": selected, "selected_value_issue": issue})
    return pd.DataFrame(rows)


def relation_plot(tab: pd.DataFrame, group_col: str, folder: Path, stem: str,
                  title: str, stat: str, bins_per_figure: int, nominal=False) -> None:
    if tab.empty:
        return
    folder.mkdir(parents=True, exist_ok=True)
    labels = tab[group_col].astype(str).tolist()
    values = pd.to_numeric(tab["selected_value"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(tab["n"], errors="coerce").fillna(0).to_numpy(dtype=int)

    if nominal:
        labels_n = [f"{lab}\nn={int(nn):,}" for lab, nn in zip(labels, n)]
        fig_h = max(5, len(labels) * 0.32)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(labels_n, values)
        ax.set_xlabel(f"{stat.title()} recorded date difference (days)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(folder / f"{stem}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    finite = values[np.isfinite(values)]
    global_ymax = float(finite.max()) if finite.size else 1.0
    global_ymax = max(global_ymax, 1.0)
    for page_no, (a, b) in enumerate(_chunks(len(labels), bins_per_figure), start=1):
        labs = labels[a:b]
        vals = values[a:b]
        ns = n[a:b]
        labels_n = [f"{lab}\nn={int(nn):,}" for lab, nn in zip(labs, ns)]
        x = np.arange(len(labs))
        fig, ax = plt.subplots(figsize=(10, 5))
        # NaN values intentionally break the line across bins with zero/no valid records.
        ax.plot(x, vals, marker="o")
        ax.set_xticks(x, labels_n, rotation=35, ha="right")
        ax.set_ylabel(f"{stat.title()} recorded date difference (days)")
        ax.set_ylim(0, global_ymax * 1.05)
        ax.set_title(f"{title}  [{labs[0]} ... {labs[-1]}]")
        fig.tight_layout()
        fig.savefig(folder / f"{stem}_part{page_no:02d}_{_safe_name(labs[0])}__{_safe_name(labs[-1])}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def cross_statistics(d: pd.DataFrame, row: str, col: str, stat: str) -> pd.DataFrame:
    rows = []
    for keys, g in d.dropna(subset=[row, col]).groupby([row, col], observed=True, sort=True):
        selected, issue = selected_stat_value(g["duration"], stat)
        rows.append({
            row: str(keys[0]), col: str(keys[1]), "n": int(len(g)),
            "mean": float(g["duration"].mean()), "median": float(g["duration"].median()),
            "mode_values": "|".join(f"{v:g}" for v in g["duration"].mode().sort_values().tolist()),
            "selected_stat": stat, "selected_value": selected, "selected_value_issue": issue,
        })
    return pd.DataFrame(rows)


def rent_by_area_plot(tab: pd.DataFrame, folder: Path, stat: str, band_schema: dict) -> None:
    """Display-only paging for the two-variable chart; all cells remain in the single CSV."""
    if tab.empty:
        return
    folder.mkdir(parents=True, exist_ok=True)
    rent_order = list(band_schema["rent"]["labels"])
    area_order = list(band_schema["area"]["labels"])
    rent_page = int(band_schema["rent"]["bins_per_figure"])
    # Limit lines per chart to keep the rendering readable; this is display-only.
    area_page = 5
    finite = pd.to_numeric(tab["selected_value"], errors="coerce")
    global_ymax = max(float(finite.max()) if finite.notna().any() else 1.0, 1.0)

    figure_no = 0
    for ra, rb in _chunks(len(rent_order), rent_page):
        rents = rent_order[ra:rb]
        for aa, ab in _chunks(len(area_order), area_page):
            areas = area_order[aa:ab]
            # Skip an image only when the whole display window has no observations; data remain in CSV.
            window = tab[tab["rent_band"].isin(rents) & tab["area_band"].isin(areas)]
            if window.empty or int(window["n"].sum()) == 0:
                continue
            figure_no += 1
            fig, ax = plt.subplots(figsize=(11, 6))
            x = np.arange(len(rents))
            plotted = 0
            for area in areas:
                part = tab[tab["area_band"].eq(area)].set_index("rent_band")
                y = [part.loc[r, "selected_value"] if r in part.index else np.nan for r in rents]
                y = np.asarray(y, dtype=float)
                if np.isfinite(y).sum() > 0:
                    ax.plot(x, y, marker="o", label=f"area={area}")
                    plotted += 1
            ax.set_xticks(x, rents, rotation=35, ha="right")
            ax.set_xlabel("money_room band")
            ax.set_ylabel(f"{stat.title()} recorded date difference (days)")
            ax.set_ylim(0, global_ymax * 1.05)
            ax.set_title(f"money_room × house_area: rent {rents[0]}...{rents[-1]}, area {areas[0]}...{areas[-1]}")
            if plotted:
                ax.legend(title="house_area band", fontsize=8)
            fig.tight_layout()
            fig.savefig(folder / f"money_room_by_house_area_{stat}_part{figure_no:03d}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

def relate(args) -> None:
    input_path = Path(args.input).resolve()
    scope_path = Path(args.scope).resolve()
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("version") != VERSION:
        raise ValueError(f"scope.json version {scope.get('version')} does not match script version {VERSION}")
    current_hash = sha256_file(input_path)
    if current_hash != scope.get("input_sha256"):
        raise ValueError("Input file differs from the file used in Step 1. Re-run inspect; do not mix scopes.")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    for sub in ["01_one_to_one", "02_multiple_variables", "audit"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    if "source_row" not in raw:
        raw["source_row"] = np.arange(2, len(raw) + 2)
    d = prepare(raw)
    band_config = normalize_band_config(scope.get("band_config"))
    band_schema = scope.get("band_schema")
    if not band_schema:
        raise ValueError("scope.json has no full-range band_schema; re-run inspect with v2.3")
    d = apply_band_schema(d, band_schema)
    years = None if scope["years"] == "all" else [int(v) for v in scope["years"]]
    types = None if scope["property_types"] == "all" else [str(v) for v in scope["property_types"]]
    primary, flow = apply_scope(d, years, types, scope["duplicates"])
    if len(primary) != int(scope["analysis_scope_n"]):
        raise ValueError("Scope record count differs from Step 1. Re-run inspect and relate from the same input.")
    save_csv(flow, out / "audit" / "sample_flow_reproduced.csv")

    # Requirement 3: one-to-one relations using the analyst's explicit statistic choice.
    specs = [
        ("rent_band", "money_room", False),
        ("area_band", "house_area", False),
        ("age_band", "age", False),
        ("layout", "layout", True),
    ]
    for group_col, name, nominal in specs:
        tab = group_statistics(primary, group_col, args.stat)
        save_csv(tab, out / "01_one_to_one" / f"{name}_vs_duration.csv")
        relation_plot(
            tab, group_col, out / "01_one_to_one" / name,
            f"{name}_vs_duration_{args.stat}",
            f"One-to-one: {name} vs recorded date difference", args.stat,
            999999 if nominal else int(band_schema[{"money_room":"rent","house_area":"area","age":"age"}.get(name,"rent")]["bins_per_figure"]),
            nominal=nominal,
        )

    # Requirement 4: increase to multiple variables. Use ordered rent bands on x and area bands as lines.
    # No cell is hidden by an arbitrary n threshold; n is retained in the CSV for interpretation.
    cross = cross_statistics(primary, "area_band", "rent_band", args.stat)
    save_csv(cross, out / "02_multiple_variables" / "money_room_by_house_area.csv")
    rent_by_area_plot(cross, out / "02_multiple_variables" / "money_room_by_house_area", args.stat, band_schema)

    # Secondary table for the other planned attributes. Kept as a table because layout is nominal and
    # drawing all layout categories as connected lines would imply an ordering that does not exist.
    age_layout = cross_statistics(primary, "layout", "age_band", args.stat)
    save_csv(age_layout, out / "02_multiple_variables" / "age_by_layout_table.csv")

    decision = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_file": str(scope_path),
        "input_sha256": current_hash,
        "representative_statistic": args.stat,
        "decision_reason": args.reason,
        "band_config": band_config,
        "band_schema": band_schema,
        "note": "The program does not infer this reason; it records the analyst's explicit decision after Step 1. The exact Step 1 band configuration is reused here.",
    }
    save_json(out / "analysis_decision.json", decision)

    report = "# Step 2: relationships\n\n"
    report += f"Representative statistic explicitly chosen: **{args.stat}**.\n\n"
    report += f"Recorded reason: **{args.reason}**\n\n"
    report += "## Order of analysis\n\n"
    report += "1. `01_one_to_one/` — one attribute at a time.\n"
    report += "2. `02_multiple_variables/money_room_by_house_area_...png` — then increase to two attributes.\n"
    report += "3. `02_multiple_variables/age_by_layout_table.csv` — age × layout is kept as a table because layout is nominal; lines would falsely imply layout order.\n\n"
    report += "## What is deliberately not done\n\n"
    report += "- No causal interpretation.\n"
    report += "- No p-values or independence assumption.\n"
    report += "- No automatic outlier deletion.\n"
    report += "- No heatmap or boxplot as a primary requirement output.\n"
    report += "- No top-coding/overflow grouping; display paging never changes the underlying CSV statistics.\n"
    report += "- No automatic property-type exclusion unless it was explicitly specified in Step 1.\n"
    (out / "RELATE_REPORT.md").write_text(report, encoding="utf-8")

    print(f"Step 2 complete: {len(primary):,} records; statistic={args.stat}")
    print(f"Open: {out / 'RELATE_REPORT.md'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract exactly one prefecture/city from the nationwide TSV")
    e.add_argument("--input", required=True)
    e.add_argument("--output", default="prepared/kokubunji.csv.gz")
    e.add_argument("--prefecture", default="東京都")
    e.add_argument("--city", default="国分寺市")
    e.add_argument("--encoding", default="utf-8-sig")
    e.add_argument("--chunk-size", type=int, default=50000)

    i = sub.add_parser("inspect", help="Step 1: distributions, representative values, and audit only")
    i.add_argument("--input", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--years", required=True, help="e.g. 2020,2021,2022,2023 or all")
    i.add_argument("--types", required=True, help="property-type codes or all; use all until a codebook justifies a filter")
    i.add_argument("--duplicates", required=True, choices=["keep", "exclude"], help="keep unless listing-ID semantics justify exclusion")
    i.add_argument("--duration-step-days", type=float, default=30, help="equal-width duration bin size in days (default: 30)")
    i.add_argument("--duration-bins-per-figure", type=int, default=12, help="display-only number of duration bins per PNG (default: 12)")
    i.add_argument("--rent-step-yen", type=float, default=20000, help="equal-width rent bin size in yen (default: 20000)")
    i.add_argument("--rent-bins-per-figure", type=int, default=10, help="display-only number of rent bins per PNG (default: 10)")
    i.add_argument("--area-step-sqm", type=float, default=10, help="equal-width area bin size in square metres (default: 10)")
    i.add_argument("--area-bins-per-figure", type=int, default=10, help="display-only number of area bins per PNG (default: 10)")
    i.add_argument("--age-step-years", type=float, default=5, help="equal-width age bin size in years (default: 5)")
    i.add_argument("--age-bins-per-figure", type=int, default=10, help="display-only number of age bins per PNG (default: 10)")

    r = sub.add_parser("relate", help="Step 2: one-to-one, then multiple variables, after explicit statistic choice")
    r.add_argument("--input", required=True)
    r.add_argument("--scope", required=True, help="scope.json produced by inspect")
    r.add_argument("--out", required=True)
    r.add_argument("--stat", required=True, choices=["mean", "median", "mode"])
    r.add_argument("--reason", required=True, help="analyst's explicit reason after inspecting Step 1")
    return p


def main():
    args = build_parser().parse_args()
    try:
        if args.command == "extract":
            extract(args)
        elif args.command == "inspect":
            inspect(args)
        else:
            relate(args)
    except (ValueError, FileNotFoundError, UnicodeError, pd.errors.ParserError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

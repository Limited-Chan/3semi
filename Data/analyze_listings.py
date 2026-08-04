#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large TSV real-estate listing analyzer.

Designed for large files:
- Reads the TSV in chunks instead of loading everything into memory.
- Reads only the columns needed by each analysis.
- Computes approximate percentiles from an integer-day histogram.
- Supports multiple analysis viewpoints through CLI subcommands.

Basic examples:
    python analyze_listings.py all listings.tsv --out-dir analysis
    python analyze_listings.py overview listings.tsv --out-dir analysis
    python analyze_listings.py factor listings.tsv --factor walk --out-dir analysis
    python analyze_listings.py group listings.tsv --group-by addr1_1_name,addr1_2_name
    python analyze_listings.py model listings.tsv --early-days 14 --sample-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "id",
    "pub_start_date",
    "pub_end_date",
    "bukken_type",
    "addr1_1_name",
    "addr1_2_name",
    "addr2_name",
    "rosen1_name",
    "eki1_name",
    "walk_distance1",
    "money_room",
    "house_area",
    "kenchiku_date",
    "madori_kind_all_label",
    "madori_code",
]

FACTOR_CHOICES = [
    "region",
    "prefecture",
    "city",
    "station",
    "layout",
    "walk",
    "age",
    "rent",
    "area",
    "price_per_sqm",
    "start_month",
    "start_year",
    "bukken_type",
]


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def decode_sep(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape")


def detect_encoding(path: Path, sep: str) -> str:
    candidates = ["utf-8-sig", "utf-8", "cp932"]
    for enc in candidates:
        try:
            pd.read_csv(path, sep=sep, encoding=enc, nrows=2)
            return enc
        except UnicodeDecodeError:
            continue
        except Exception:
            # The encoding may be correct even if malformed rows exist.
            try:
                with path.open("r", encoding=enc) as f:
                    f.readline()
                return enc
            except UnicodeDecodeError:
                continue
    raise UnicodeError("Could not detect encoding. Try --encoding cp932 or --encoding utf-8.")


def read_columns(path: Path, sep: str, encoding: str) -> list[str]:
    return list(pd.read_csv(path, sep=sep, encoding=encoding, nrows=0).columns)


def existing_columns(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    available_set = set(available)
    return [c for c in requested if c in available_set]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_bins(value: str) -> list[float]:
    result: list[float] = []
    for item in parse_csv_list(value):
        lowered = item.lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            result.append(float("inf"))
        elif lowered in {"-inf", "-infinity"}:
            result.append(float("-inf"))
        else:
            result.append(float(item))
    if len(result) < 2:
        raise ValueError("At least two bin edges are required.")
    if result != sorted(result):
        raise ValueError("Bin edges must be sorted.")
    return result


def safe_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def read_chunks(
    path: Path,
    *,
    columns: Sequence[str] | None,
    sep: str,
    encoding: str,
    chunk_size: int,
    on_bad_lines: str,
) -> Iterable[pd.DataFrame]:
    kwargs = {
        "filepath_or_buffer": path,
        "sep": sep,
        "encoding": encoding,
        "chunksize": chunk_size,
        "dtype": str,
        "on_bad_lines": on_bad_lines,
        "low_memory": False,
    }
    if columns is not None:
        kwargs["usecols"] = list(columns)
    yield from pd.read_csv(**kwargs)


def construction_year(series: pd.Series) -> pd.Series:
    # Examples expected: 200203, 2002-03, 2002/03, or 2002.
    text = series.astype("string").str.strip()
    year_text = text.str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(year_text, errors="coerce")


def cut_with_labels(series: pd.Series, bins: Sequence[float], suffix: str = "") -> pd.Series:
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if math.isinf(left) and left < 0:
            labels.append(f"< {right:g}{suffix}")
        elif math.isinf(right):
            labels.append(f"{left:g}{suffix}+")
        else:
            labels.append(f"{left:g}-{right:g}{suffix}")
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")


class QualityTracker:
    def __init__(self) -> None:
        self.rows_read = 0
        self.valid_rows = 0
        self.missing_start = 0
        self.missing_end = 0
        self.negative_duration = 0
        self.too_long_duration = 0
        self.column_non_null: defaultdict[str, int] = defaultdict(int)

    def observe_raw(self, df: pd.DataFrame) -> None:
        self.rows_read += len(df)
        for col in df.columns:
            self.column_non_null[col] += int(df[col].notna().sum())

    def observe_dates(
        self,
        start: pd.Series,
        end: pd.Series,
        duration: pd.Series,
        max_valid_days: int,
    ) -> None:
        self.missing_start += int(start.isna().sum())
        self.missing_end += int(end.isna().sum())
        self.negative_duration += int((duration < 0).fillna(False).sum())
        self.too_long_duration += int((duration > max_valid_days).fillna(False).sum())

    def to_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "valid_completed_listing_rows": self.valid_rows,
            "missing_pub_start_date": self.missing_start,
            "missing_pub_end_date": self.missing_end,
            "negative_duration_rows": self.negative_duration,
            "duration_over_limit_rows": self.too_long_duration,
        }

    def missingness_frame(self) -> pd.DataFrame:
        rows = []
        for col, non_null in self.column_non_null.items():
            missing = self.rows_read - non_null
            rows.append(
                {
                    "column": col,
                    "non_null_count": non_null,
                    "missing_count": missing,
                    "missing_rate": missing / self.rows_read if self.rows_read else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values(["missing_rate", "column"], ascending=[False, True])


def prepare_chunk(df: pd.DataFrame, args: argparse.Namespace, quality: QualityTracker) -> pd.DataFrame:
    quality.observe_raw(df)

    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    start = pd.to_datetime(df["pub_start_date"], errors="coerce")
    end = pd.to_datetime(df["pub_end_date"], errors="coerce")
    duration = (end - start).dt.days.astype("float64")

    quality.observe_dates(start, end, duration, args.max_valid_days)

    valid = (
        start.notna()
        & end.notna()
        & duration.ge(0)
        & duration.le(args.max_valid_days)
    )

    if args.start_from:
        valid &= start.ge(pd.Timestamp(args.start_from))
    if args.start_to:
        valid &= start.le(pd.Timestamp(args.start_to))

    if args.prefecture:
        valid &= df["addr1_1_name"].astype("string").eq(args.prefecture)
    if args.city:
        valid &= df["addr1_2_name"].astype("string").eq(args.city)

    selected_types = set(parse_csv_list(args.bukken_type))
    if selected_types:
        valid &= df["bukken_type"].astype("string").isin(selected_types)

    out = df.loc[valid].copy()
    if out.empty:
        return out

    out["pub_start_date_parsed"] = start.loc[valid]
    out["pub_end_date_parsed"] = end.loc[valid]
    out["duration_days"] = duration.loc[valid].astype("int32")

    out["rent_yen"] = safe_numeric(out["money_room"])
    out["area_sqm"] = safe_numeric(out["house_area"])
    out["walk_raw"] = safe_numeric(out["walk_distance1"])

    if args.walk_unit == "meters":
        out["walk_minutes"] = np.ceil(out["walk_raw"] / args.meters_per_minute)
    else:
        out["walk_minutes"] = out["walk_raw"]

    out["construction_year"] = construction_year(out["kenchiku_date"])
    out["building_age_years"] = (
        out["pub_start_date_parsed"].dt.year - out["construction_year"]
    )
    out.loc[out["building_age_years"] < 0, "building_age_years"] = np.nan

    layout = out["madori_kind_all_label"].astype("string").str.strip()
    fallback = out["madori_code"].astype("string").str.strip()
    out["layout"] = layout.mask(layout.isna() | layout.eq(""), fallback)
    out["layout"] = out["layout"].fillna("(missing)")

    out["prefecture"] = out["addr1_1_name"].astype("string").fillna("(missing)")
    out["city"] = out["addr1_2_name"].astype("string").fillna("(missing)")
    out["station"] = out["eki1_name"].astype("string").fillna("(missing)")
    out["region"] = out["prefecture"] + " / " + out["city"]

    out["start_month"] = out["pub_start_date_parsed"].dt.month.astype("Int64").astype("string")
    out["start_year"] = out["pub_start_date_parsed"].dt.year.astype("Int64").astype("string")
    out["price_per_sqm"] = out["rent_yen"] / out["area_sqm"].replace(0, np.nan)

    quality.valid_rows += len(out)
    return out


class GroupAccumulator:
    """Streaming group statistics with an integer-day histogram."""

    def __init__(self, group_columns: Sequence[str], duration_cap: int) -> None:
        self.group_columns = list(group_columns)
        self.duration_cap = duration_cap
        self.total_count: defaultdict[tuple, int] = defaultdict(int)
        self.duration_sum: defaultdict[tuple, int] = defaultdict(int)
        self.early_7: defaultdict[tuple, int] = defaultdict(int)
        self.early_14: defaultdict[tuple, int] = defaultdict(int)
        self.early_30: defaultdict[tuple, int] = defaultdict(int)
        self.hist: defaultdict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.duration_cap + 2, dtype=np.int64)
        )

    @staticmethod
    def _as_key(value: object) -> tuple:
        if isinstance(value, tuple):
            return value
        return (value,)

    def update(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        work = df.copy()
        if not self.group_columns:
            work["_all"] = "ALL"
            group_cols = ["_all"]
        else:
            group_cols = self.group_columns
            for col in group_cols:
                work[col] = work[col].astype("string").fillna("(missing)")

        work["_e7"] = work["duration_days"].le(7).astype("int8")
        work["_e14"] = work["duration_days"].le(14).astype("int8")
        work["_e30"] = work["duration_days"].le(30).astype("int8")
        work["_duration_bin"] = work["duration_days"].clip(
            upper=self.duration_cap + 1
        ).astype("int32")

        totals = (
            work.groupby(group_cols, dropna=False, observed=True)
            .agg(
                count=("duration_days", "size"),
                duration_sum=("duration_days", "sum"),
                early_7=("_e7", "sum"),
                early_14=("_e14", "sum"),
                early_30=("_e30", "sum"),
            )
            .reset_index()
        )

        for row in totals.itertuples(index=False, name=None):
            key = self._as_key(row[: len(group_cols)])
            self.total_count[key] += int(row[len(group_cols)])
            self.duration_sum[key] += int(row[len(group_cols) + 1])
            self.early_7[key] += int(row[len(group_cols) + 2])
            self.early_14[key] += int(row[len(group_cols) + 3])
            self.early_30[key] += int(row[len(group_cols) + 4])

        hist_counts = (
            work.groupby(group_cols + ["_duration_bin"], dropna=False, observed=True)
            .size()
            .reset_index(name="count")
        )
        for row in hist_counts.itertuples(index=False, name=None):
            key = self._as_key(row[: len(group_cols)])
            duration_bin = int(row[len(group_cols)])
            count = int(row[len(group_cols) + 1])
            self.hist[key][duration_bin] += count

    def _quantile(self, hist: np.ndarray, q: float) -> int | None:
        total = int(hist.sum())
        if total == 0:
            return None
        target = max(1, math.ceil(total * q))
        value = int(np.searchsorted(np.cumsum(hist), target))
        return value

    def finalize(self) -> pd.DataFrame:
        rows = []
        for key, count in self.total_count.items():
            row: dict[str, object] = {}
            if self.group_columns:
                for col, value in zip(self.group_columns, key):
                    row[col] = str(value)
            else:
                row["scope"] = "ALL"

            hist = self.hist[key]
            row.update(
                {
                    "count": count,
                    "mean_days": self.duration_sum[key] / count if count else np.nan,
                    "p25_days_approx": self._quantile(hist, 0.25),
                    "median_days_approx": self._quantile(hist, 0.50),
                    "p75_days_approx": self._quantile(hist, 0.75),
                    "ended_within_7d_rate": self.early_7[key] / count if count else np.nan,
                    "ended_within_14d_rate": self.early_14[key] / count if count else np.nan,
                    "ended_within_30d_rate": self.early_30[key] / count if count else np.nan,
                }
            )
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("count", ascending=False)

    def overall_histogram(self) -> pd.DataFrame:
        if self.group_columns:
            raise ValueError("overall_histogram is only available for the overall accumulator.")
        key = ("ALL",)
        values = self.hist.get(key, np.zeros(self.duration_cap + 2, dtype=np.int64))
        labels = [str(i) for i in range(self.duration_cap + 1)] + [f"{self.duration_cap + 1}+"]
        return pd.DataFrame({"duration_day": labels, "count": values})


def add_factor_column(df: pd.DataFrame, factor: str, args: argparse.Namespace) -> str:
    target = "factor_value"

    if factor in {
        "region",
        "prefecture",
        "city",
        "station",
        "layout",
        "start_month",
        "start_year",
        "bukken_type",
    }:
        df[target] = df[factor].astype("string").fillna("(missing)")
    elif factor == "walk":
        df[target] = cut_with_labels(
            df["walk_minutes"], parse_bins(args.walk_bins), " min"
        ).fillna("(missing)")
    elif factor == "age":
        df[target] = cut_with_labels(
            df["building_age_years"], parse_bins(args.age_bins), " yr"
        ).fillna("(missing)")
    elif factor == "rent":
        df[target] = cut_with_labels(
            df["rent_yen"], parse_bins(args.rent_bins), " yen"
        ).fillna("(missing)")
    elif factor == "area":
        df[target] = cut_with_labels(
            df["area_sqm"], parse_bins(args.area_bins), " sqm"
        ).fillna("(missing)")
    elif factor == "price_per_sqm":
        df[target] = cut_with_labels(
            df["price_per_sqm"], parse_bins(args.price_per_sqm_bins), " yen/sqm"
        ).fillna("(missing)")
    else:
        raise ValueError(f"Unsupported factor: {factor}")

    return target


def ensure_out_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def configure_matplotlib_font(plt) -> None:
    """Prefer a Japanese-capable font when available."""
    try:
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in [
            "Yu Gothic",
            "Yu Gothic UI",
            "Meiryo",
            "MS Gothic",
            "Noto Sans CJK JP",
            "IPAexGothic",
        ]:
            if candidate in available:
                plt.rcParams["font.family"] = candidate
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def maybe_plot_histogram(hist: pd.DataFrame, path: Path, duration_cap: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        eprint("matplotlib is not installed; skipping plots.")
        return

    configure_matplotlib_font(plt)
    numeric = hist.iloc[: duration_cap + 1]
    x = pd.to_numeric(numeric["duration_day"])
    y = numeric["count"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, y, width=1.0)
    ax.set_title("Listing duration distribution")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Listings")
    ax.set_xlim(0, min(duration_cap, 180))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def maybe_plot_summary(
    summary: pd.DataFrame,
    group_columns: Sequence[str],
    path: Path,
    *,
    metric: str = "ended_within_14d_rate",
    top_n: int = 30,
) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configure_matplotlib_font(plt)
    work = summary.head(top_n).copy()
    if group_columns:
        work["label"] = work[list(group_columns)].astype(str).agg(" / ".join, axis=1)
    else:
        work["label"] = "ALL"

    work = work.sort_values(metric)
    fig_height = max(5, 0.28 * len(work))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(work["label"], work[metric])
    ax.set_title(metric)
    ax.set_xlabel(metric)
    if metric.endswith("_rate"):
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def resolve_file_settings(args: argparse.Namespace) -> tuple[Path, str, str, list[str]]:
    path = Path(args.tsv)
    if not path.exists():
        raise FileNotFoundError(path)
    sep = decode_sep(args.sep)
    encoding = detect_encoding(path, sep) if args.encoding == "auto" else args.encoding
    available = read_columns(path, sep, encoding)
    return path, sep, encoding, available


def run_profile(args: argparse.Namespace) -> None:
    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)

    columns = available if args.profile_columns == "all" else existing_columns(BASE_COLUMNS, available)
    quality = QualityTracker()

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        quality.observe_raw(chunk)
        if i % 10 == 0:
            eprint(f"profile: processed {quality.rows_read:,} rows")

    summary = {
        "file": str(path),
        "encoding": encoding,
        "separator": repr(sep),
        "column_count_in_file": len(available),
        "profiled_column_count": len(columns),
        "rows_read": quality.rows_read,
    }
    save_json(summary, out / "profile_summary.json")
    quality.missingness_frame().to_csv(out / "profile_missingness.csv", index=False, encoding="utf-8-sig")
    print(f"Saved profile results to: {out}")


def run_overview(args: argparse.Namespace) -> None:
    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)
    columns = existing_columns(BASE_COLUMNS, available)

    quality = QualityTracker()
    acc = GroupAccumulator([], args.duration_cap)

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        clean = prepare_chunk(chunk, args, quality)
        if not clean.empty:
            acc.update(clean[["duration_days"]])
        if i % 10 == 0:
            eprint(f"overview: processed {quality.rows_read:,} rows")

    summary = acc.finalize()
    hist = acc.overall_histogram()
    summary.to_csv(out / "overview_summary.csv", index=False, encoding="utf-8-sig")
    hist.to_csv(out / "duration_histogram.csv", index=False, encoding="utf-8-sig")
    maybe_plot_histogram(hist, out / "duration_histogram.png", args.duration_cap)

    metadata = quality.to_dict()
    metadata.update(
        {
            "file": str(path),
            "encoding": encoding,
            "duration_percentiles_are_histogram_based": True,
            "duration_histogram_cap_days": args.duration_cap,
            "values_above_cap_are_grouped_into": f"{args.duration_cap + 1}+",
        }
    )
    save_json(metadata, out / "overview_quality.json")
    print(f"Saved overview results to: {out}")


def run_group(args: argparse.Namespace) -> None:
    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)
    group_cols = parse_csv_list(args.group_by)
    missing = [c for c in group_cols if c not in available]
    if missing:
        raise KeyError(f"Group columns not found: {missing}")

    columns = existing_columns(BASE_COLUMNS + group_cols, available)
    quality = QualityTracker()
    acc = GroupAccumulator(group_cols, args.duration_cap)

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        clean = prepare_chunk(chunk, args, quality)
        if not clean.empty:
            acc.update(clean[["duration_days", *group_cols]])
        if i % 10 == 0:
            eprint(f"group: processed {quality.rows_read:,} rows")

    summary = acc.finalize()
    slug = "_".join(group_cols)
    summary.to_csv(out / f"group_{slug}.csv", index=False, encoding="utf-8-sig")
    maybe_plot_summary(
        summary,
        group_cols,
        out / f"group_{slug}_14d_rate.png",
        top_n=args.top_n,
    )
    save_json(quality.to_dict(), out / f"group_{slug}_quality.json")
    print(f"Saved group results to: {out}")


def run_factor(args: argparse.Namespace) -> None:
    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)
    columns = existing_columns(BASE_COLUMNS, available)

    quality = QualityTracker()
    acc = GroupAccumulator(["factor_value"], args.duration_cap)

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        clean = prepare_chunk(chunk, args, quality)
        if not clean.empty:
            add_factor_column(clean, args.factor, args)
            acc.update(clean[["duration_days", "factor_value"]])
        if i % 10 == 0:
            eprint(f"factor={args.factor}: processed {quality.rows_read:,} rows")

    summary = acc.finalize()
    summary.to_csv(out / f"factor_{args.factor}.csv", index=False, encoding="utf-8-sig")
    maybe_plot_summary(
        summary,
        ["factor_value"],
        out / f"factor_{args.factor}_14d_rate.png",
        top_n=args.top_n,
    )
    save_json(quality.to_dict(), out / f"factor_{args.factor}_quality.json")
    print(f"Saved factor results to: {out}")


def run_all(args: argparse.Namespace) -> None:
    """
    Runs the main exploratory analyses in one file pass:
    overview, region, layout, walk, age, rent, area, price/sqm,
    start month, and property type.
    """
    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)
    columns = existing_columns(BASE_COLUMNS, available)

    factors = [
        "layout",
        "walk",
        "age",
        "rent",
        "area",
        "price_per_sqm",
        "start_month",
        "bukken_type",
    ]

    quality = QualityTracker()
    overall = GroupAccumulator([], args.duration_cap)
    region = GroupAccumulator(["prefecture", "city"], args.duration_cap)
    factor_accs = {
        factor: GroupAccumulator(["factor_value"], args.duration_cap)
        for factor in factors
    }

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        clean = prepare_chunk(chunk, args, quality)
        if clean.empty:
            continue

        overall.update(clean[["duration_days"]])
        region.update(clean[["duration_days", "prefecture", "city"]])

        for factor, acc in factor_accs.items():
            add_factor_column(clean, factor, args)
            acc.update(clean[["duration_days", "factor_value"]])

        if i % 5 == 0:
            eprint(f"all: processed {quality.rows_read:,} rows")

    overview_summary = overall.finalize()
    overview_hist = overall.overall_histogram()
    overview_summary.to_csv(out / "overview_summary.csv", index=False, encoding="utf-8-sig")
    overview_hist.to_csv(out / "duration_histogram.csv", index=False, encoding="utf-8-sig")
    maybe_plot_histogram(overview_hist, out / "duration_histogram.png", args.duration_cap)

    region_summary = region.finalize()
    region_summary.to_csv(out / "region_summary.csv", index=False, encoding="utf-8-sig")
    maybe_plot_summary(
        region_summary,
        ["prefecture", "city"],
        out / "region_14d_rate.png",
        top_n=args.top_n,
    )

    for factor, acc in factor_accs.items():
        summary = acc.finalize()
        summary.to_csv(out / f"factor_{factor}.csv", index=False, encoding="utf-8-sig")
        maybe_plot_summary(
            summary,
            ["factor_value"],
            out / f"factor_{factor}_14d_rate.png",
            top_n=args.top_n,
        )

    quality.missingness_frame().to_csv(
        out / "key_columns_missingness.csv", index=False, encoding="utf-8-sig"
    )
    metadata = quality.to_dict()
    metadata.update(
        {
            "file": str(path),
            "encoding": encoding,
            "analyzed_columns": columns,
            "duration_histogram_cap_days": args.duration_cap,
            "walk_distance_interpretation": args.walk_unit,
            "meters_per_minute": args.meters_per_minute if args.walk_unit == "meters" else None,
        }
    )
    save_json(metadata, out / "analysis_quality.json")
    print(f"Saved all analysis results to: {out}")


def run_model(args: argparse.Namespace) -> None:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "The model command requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        ) from exc

    path, sep, encoding, available = resolve_file_settings(args)
    out = ensure_out_dir(args.out_dir)
    columns = existing_columns(BASE_COLUMNS, available)

    quality = QualityTracker()
    samples: list[pd.DataFrame] = []
    sampled_rows = 0
    rng = np.random.default_rng(args.random_state)

    keep = [
        "pub_start_date_parsed",
        "duration_days",
        "prefecture",
        "city",
        "layout",
        "bukken_type",
        "rent_yen",
        "area_sqm",
        "walk_minutes",
        "building_age_years",
        "start_month",
        "price_per_sqm",
    ]

    for i, chunk in enumerate(
        read_chunks(
            path,
            columns=columns,
            sep=sep,
            encoding=encoding,
            chunk_size=args.chunk_size,
            on_bad_lines=args.on_bad_lines,
        ),
        start=1,
    ):
        clean = prepare_chunk(chunk, args, quality)
        if clean.empty:
            continue

        if args.sample_rate < 1:
            mask = rng.random(len(clean)) < args.sample_rate
            clean = clean.loc[mask]
        if clean.empty:
            continue

        samples.append(clean[keep])
        sampled_rows += len(clean)

        if sampled_rows > args.max_model_rows * 2:
            merged = pd.concat(samples, ignore_index=True)
            merged = merged.sample(
                n=min(args.max_model_rows, len(merged)),
                random_state=args.random_state,
            )
            samples = [merged]
            sampled_rows = len(merged)

        if i % 10 == 0:
            eprint(f"model sampling: processed {quality.rows_read:,} rows; sampled {sampled_rows:,}")

    if not samples:
        raise RuntimeError("No valid rows were available for modeling.")

    data = pd.concat(samples, ignore_index=True)
    if len(data) > args.max_model_rows:
        data = data.sample(n=args.max_model_rows, random_state=args.random_state)

    data = data.sort_values("pub_start_date_parsed").reset_index(drop=True)
    data["target"] = data["duration_days"].le(args.early_days).astype("int8")

    split_at = max(1, int(len(data) * 0.8))
    train = data.iloc[:split_at]
    test = data.iloc[split_at:]
    if test.empty or train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise RuntimeError(
            "Not enough class variation after the chronological split. "
            "Increase --sample-rate or --max-model-rows."
        )

    numeric_features = [
        "rent_yen",
        "area_sqm",
        "walk_minutes",
        "building_age_years",
        "price_per_sqm",
    ]
    categorical_features = [
        "prefecture",
        "city",
        "layout",
        "bukken_type",
        "start_month",
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    classifier = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=args.random_state,
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    x_train = train[numeric_features + categorical_features]
    y_train = train["target"]
    x_test = test[numeric_features + categorical_features]
    y_test = test["target"]

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "target": f"ended_within_{args.early_days}_days",
        "sample_rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_min": str(train["pub_start_date_parsed"].min().date()),
        "train_date_max": str(train["pub_start_date_parsed"].max().date()),
        "test_date_min": str(test["pub_start_date_parsed"].min().date()),
        "test_date_max": str(test["pub_start_date_parsed"].max().date()),
        "test_baseline_positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    save_json(metrics, out / f"model_{args.early_days}d_metrics.json")

    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients = model.named_steps["classifier"].coef_[0]
        coef = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        coef.to_csv(
            out / f"model_{args.early_days}d_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        eprint(f"Could not export coefficients: {exc}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model results to: {out}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tsv", help="Input TSV file")
    parser.add_argument("--out-dir", default="analysis_output", help="Output directory")
    parser.add_argument("--sep", default=r"\t", help=r"Delimiter; default is \t")
    parser.add_argument(
        "--encoding",
        default="auto",
        help="auto, utf-8, utf-8-sig, or cp932",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--on-bad-lines",
        choices=["error", "warn", "skip"],
        default="warn",
    )
    parser.add_argument(
        "--duration-cap",
        type=int,
        default=365,
        help="Histogram cap used for approximate percentiles",
    )
    parser.add_argument(
        "--max-valid-days",
        type=int,
        default=3650,
        help="Rows longer than this are treated as abnormal",
    )
    parser.add_argument("--start-from", help="Filter by publication start date, YYYY-MM-DD")
    parser.add_argument("--start-to", help="Filter by publication start date, YYYY-MM-DD")
    parser.add_argument("--prefecture", help="Filter by addr1_1_name")
    parser.add_argument("--city", help="Filter by addr1_2_name")
    parser.add_argument(
        "--bukken-type",
        default="",
        help="Comma-separated bukken_type codes to retain",
    )
    parser.add_argument(
        "--walk-unit",
        choices=["meters", "minutes"],
        default="meters",
        help="Interpretation of walk_distance1",
    )
    parser.add_argument("--meters-per-minute", type=float, default=80.0)
    parser.add_argument(
        "--walk-bins",
        default="0,5,10,15,20,inf",
        help="Walking-minute bin edges",
    )
    parser.add_argument(
        "--age-bins",
        default="0,5,10,20,30,40,inf",
        help="Building-age bin edges",
    )
    parser.add_argument(
        "--rent-bins",
        default="0,50000,70000,90000,120000,150000,200000,inf",
        help="Monthly-rent bin edges",
    )
    parser.add_argument(
        "--area-bins",
        default="0,20,30,40,50,70,100,inf",
        help="Area bin edges",
    )
    parser.add_argument(
        "--price-per-sqm-bins",
        default="0,1500,2000,2500,3000,4000,5000,inf",
        help="Rent-per-square-meter bin edges",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Groups shown in plots")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunked analysis of large real-estate listing TSV files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="Run the main exploratory analyses in one pass")
    add_common_arguments(p_all)
    p_all.set_defaults(func=run_all)

    p_overview = sub.add_parser("overview", help="Overall listing-duration distribution")
    add_common_arguments(p_overview)
    p_overview.set_defaults(func=run_overview)

    p_profile = sub.add_parser("profile", help="Missingness and basic file profile")
    add_common_arguments(p_profile)
    p_profile.add_argument(
        "--profile-columns",
        choices=["key", "all"],
        default="key",
        help="Use 'all' only when full-column missingness is needed",
    )
    p_profile.set_defaults(func=run_profile)

    p_group = sub.add_parser("group", help="Analyze arbitrary raw columns")
    add_common_arguments(p_group)
    p_group.add_argument(
        "--group-by",
        required=True,
        help="Comma-separated source columns, e.g. addr1_1_name,addr1_2_name",
    )
    p_group.set_defaults(func=run_group)

    p_factor = sub.add_parser("factor", help="Analyze one prepared factor")
    add_common_arguments(p_factor)
    p_factor.add_argument("--factor", choices=FACTOR_CHOICES, required=True)
    p_factor.set_defaults(func=run_factor)

    p_model = sub.add_parser("model", help="Fast sampled logistic model")
    add_common_arguments(p_model)
    p_model.add_argument("--early-days", type=int, default=14)
    p_model.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Fraction sampled from each chunk; 1.0 uses all rows until max",
    )
    p_model.add_argument("--max-model-rows", type=int, default=300_000)
    p_model.add_argument("--random-state", type=int, default=42)
    p_model.set_defaults(func=run_model)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
        return 0
    except Exception as exc:
        eprint(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

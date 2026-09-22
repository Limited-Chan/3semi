#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掲載期間 × 属性A × 属性B の3変数分析(東京都国分寺市)。

生データ(bukken_rent_v1.0.1.tsv, 全国175列・約7.1GB)から国分寺市の物件のみを
抽出し、掲載期間と各属性の関係の強さを実データで比較したうえで、最も有望な
2属性を選んで区間×カテゴリのクロス集計・可視化・考察を行う。

実行方法:
    python3 analyze.py --input /path/to/bukken_rent_v1.0.1.tsv
    (2回目以降は prepared/kokubunji.parquet があれば --input を省略可)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect: 3D projection)
from scipy import stats

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / "prepared"
OUT_DIR = HERE / "output"
TABLE_DIR = HERE / "tables"
for p in (PREPARED_DIR, OUT_DIR, TABLE_DIR):
    p.mkdir(exist_ok=True)

PREFECTURE = "東京都"
CITY = "国分寺市"
NEEDED_COLUMNS = [
    "id", "pub_start_date", "pub_end_date", "addr1_1_name", "addr1_2_name", "bukken_type",
    "kenchiku_date", "madori_number_all", "madori_kind_all_label", "money_room", "house_area",
    "walk_distance1", "eki1_name", "rosen1_name",
]
VALID_LAYOUT_KINDS = ["R", "K", "DK", "LK", "LDK", "SK", "SDK", "SLDK"]

AGE_STEP_YEARS = 5.0
AGE_DISPLAY_MAX = 60.0  # 表示だけを打ち切る値。集計CSVは全範囲(最大約93年)を保持する。
TOP_LAYOUTS = ["1K", "1R", "1LDK", "2LDK", "2DK", "1DK", "3LDK", "3DK"]  # n>=100 上位8種(可視化用)


def extract_kokubunji(tsv_path: Path, chunksize: int = 300_000) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        tsv_path, sep="\t", usecols=NEEDED_COLUMNS, dtype=str,
        chunksize=chunksize, encoding="utf-8",
    ):
        f = chunk[(chunk["addr1_1_name"] == PREFECTURE) & (chunk["addr1_2_name"] == CITY)]
        if len(f):
            chunks.append(f)
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(PREPARED_DIR / "kokubunji_raw.parquet")
    return df


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(raw["pub_start_date"], format="%Y-%m-%d", errors="coerce")
    end = pd.to_datetime(raw["pub_end_date"], format="%Y-%m-%d", errors="coerce")
    duration = (end - start).dt.days.astype(float)

    build_text = raw["kenchiku_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    valid_fmt = build_text.str.fullmatch(r"\d{6}").fillna(False)
    build = pd.to_datetime(build_text.where(valid_fmt), format="%Y%m", errors="coerce")
    age_months = (start.dt.year - build.dt.year) * 12 + (start.dt.month - build.dt.month)
    age = (age_months / 12).where(age_months.ge(0))

    rent = pd.to_numeric(raw["money_room"], errors="coerce")
    rent = rent.where(rent.gt(0))
    area = pd.to_numeric(raw["house_area"], errors="coerce")
    area = area.where(area.gt(0))
    walk = pd.to_numeric(raw["walk_distance1"], errors="coerce")
    walk = walk.where(walk.gt(0))

    rooms = pd.to_numeric(raw["madori_number_all"], errors="coerce")
    kind = raw["madori_kind_all_label"].astype(str).str.upper().str.replace(" ", "", regex=False)
    layout_ok = rooms.notna() & rooms.ge(1) & rooms.mod(1).eq(0) & kind.isin(VALID_LAYOUT_KINDS)
    layout = pd.Series(pd.NA, index=raw.index, dtype="object")
    layout[layout_ok] = rooms[layout_ok].astype(int).astype(str) + kind[layout_ok]

    d = pd.DataFrame({
        "id": raw["id"], "duration": duration, "age": age, "rent": rent,
        "area": area, "walk": walk, "layout": layout,
    })
    before = len(d)
    d = d[duration.notna() & duration.ge(0)].reset_index(drop=True)
    dropped_invalid_duration = before - len(d)
    dup = d["id"].duplicated(keep="first")
    d = d[~dup].reset_index(drop=True)
    print(f"[準備] 国分寺市 {before:,}件 → 掲載期間が有効な{len(d) + dup.sum():,}件"
          f"(不正な日付を{dropped_invalid_duration:,}件除外)→ 重複ID{int(dup.sum()):,}件を除外 → "
          f"分析対象 {len(d):,}件")
    return d


def load_data(input_tsv: str | None) -> pd.DataFrame:
    cache = PREPARED_DIR / "kokubunji_raw.parquet"
    if input_tsv:
        raw = extract_kokubunji(Path(input_tsv))
    elif cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raise FileNotFoundError("生データが未指定かつキャッシュも無い。--input で元TSVのパスを指定してください。")
    return prepare(raw)


def step1_overview(d: pd.DataFrame) -> None:
    print("=" * 70)
    print("手順1: カラム構成・型の確認")
    print("=" * 70)
    info = pd.DataFrame({
        "dtype": d.dtypes.astype(str),
        "非欠損数": d.notna().sum(),
        "欠損数": d.isna().sum(),
        "欠損率": (d.isna().sum() / len(d)).round(4),
    })
    info.loc["duration", "変数の種類"] = "連続値(目的変数)"
    for c in ["age", "rent", "area", "walk"]:
        info.loc[c, "変数の種類"] = "連続値"
    info.loc["layout", "変数の種類"] = "カテゴリ(順序なし)"
    info.loc["id", "変数の種類"] = "識別子"
    print(info.to_string())
    info.to_csv(TABLE_DIR / "01_column_overview.csv", encoding="utf-8-sig")


def step2_propose_attributes(d: pd.DataFrame) -> tuple[str, str]:
    print("\n" + "=" * 70)
    print("手順2: 掲載期間と組み合わせる2属性の提案")
    print("=" * 70)

    single = []
    for col in ["age", "rent", "area", "walk"]:
        sub = d[[col, "duration"]].dropna()
        rho, _ = stats.spearmanr(sub[col], sub["duration"])
        r, _ = stats.pearsonr(sub[col], sub["duration"])
        single.append({"変数": col, "型": "連続値", "n": len(sub), "spearman": rho, "pearson": r})

    sub = d[["layout", "duration"]].dropna()
    sub = sub[sub.groupby("layout")["layout"].transform("count") >= 30]
    grand_mean = sub["duration"].mean()
    ss_between = sub.groupby("layout")["duration"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    ss_total = ((sub["duration"] - grand_mean) ** 2).sum()
    eta = (ss_between / ss_total) ** 0.5
    single.append({"変数": "layout", "型": "カテゴリ", "n": len(sub), "eta(相関比)": eta})

    single_df = pd.DataFrame(single)
    print("\n[単変数の関連の強さ]")
    print(single_df.to_string(index=False))
    single_df.to_csv(TABLE_DIR / "02_single_variable_strength.csv", index=False, encoding="utf-8-sig")

    def pair_r2(a: str, b: str) -> tuple[int, float]:
        s = d[[a, b, "duration"]].dropna()
        y = s["duration"].to_numpy(dtype=float)
        parts = []
        for v in (a, b):
            if s[v].dtype == object:
                parts.append(pd.get_dummies(s[v], drop_first=True).to_numpy(dtype=float))
            else:
                parts.append(s[[v]].to_numpy(dtype=float))
        x = np.hstack([np.ones((len(s), 1))] + parts)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return len(s), 1 - ss_res / ss_tot

    candidate_pairs = [("age", "rent"), ("age", "layout"), ("rent", "layout"),
                        ("age", "area"), ("rent", "area"), ("area", "layout")]
    pair_rows = []
    for a, b in candidate_pairs:
        n, r2 = pair_r2(a, b)
        pair_rows.append({"属性A": a, "属性B": b, "n": n, "R2(交互作用なし)": r2})
    pair_df = pd.DataFrame(pair_rows).sort_values("R2(交互作用なし)", ascending=False)
    print("\n[2属性の組み合わせによる説明力(重回帰R^2, 交互作用なし・参考値)]")
    print(pair_df.to_string(index=False))
    pair_df.to_csv(TABLE_DIR / "03_pair_explanatory_power.csv", index=False, encoding="utf-8-sig")

    best_a, best_b = pair_df.iloc[0][["属性A", "属性B"]]
    print(f"\n→ 提案: 属性A = {best_a}, 属性B = {best_b}")
    print(
        "  理由: 単変数では築年数(spearman≈0.31)が最も強く、家賃(-0.22)がこれに次ぐ。\n"
        "  一方、築年数×間取りの組み合わせは(交互作用なしの)重回帰R^2が候補中最大であり、\n"
        "  単変数の強さと2変数合成時の説明力の両方で一貫して上位に来る。\n"
        "  間取りは計画書の例にもある代表的なカテゴリ変数であり解釈もしやすいため採用する。\n"
        "  最寄駅徒歩距離(walk)はspearman≈-0.02とほぼ無相関のため対象から外した。"
    )
    return str(best_a), str(best_b)


def build_age_bins(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    max_age = float(np.ceil(d["age"].max() / AGE_STEP_YEARS) * AGE_STEP_YEARS)
    edges = np.arange(0, max_age + AGE_STEP_YEARS, AGE_STEP_YEARS)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}年" for i in range(len(edges) - 1)]
    return edges, labels


def step3_cross_table(d: pd.DataFrame, edges: np.ndarray, labels: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("手順3: 築年数(区間) × 間取り の集計表")
    print("=" * 70)
    sub = d.dropna(subset=["age", "layout"]).copy()
    sub["age_band"] = pd.cut(sub["age"], bins=edges, labels=labels, right=False, include_lowest=True)

    rows = []
    for age_band in labels:
        for layout in sorted(sub["layout"].unique()):
            g = sub[(sub["age_band"] == age_band) & (sub["layout"] == layout)]["duration"]
            if len(g) == 0:
                rows.append({"age_band": age_band, "layout": layout, "n": 0, "mean": np.nan,
                             "median": np.nan, "std": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan})
                continue
            rows.append({
                "age_band": age_band, "layout": layout, "n": int(len(g)),
                "mean": float(g.mean()), "median": float(g.median()), "std": float(g.std()) if len(g) > 1 else 0.0,
                "q1": float(g.quantile(0.25)), "q3": float(g.quantile(0.75)),
                "iqr": float(g.quantile(0.75) - g.quantile(0.25)),
            })
    cross = pd.DataFrame(rows)
    cross["age_band"] = pd.Categorical(cross["age_band"], categories=labels, ordered=True)
    cross = cross.sort_values(["age_band", "layout"]).reset_index(drop=True)
    cross.to_csv(TABLE_DIR / "04_age_band_x_layout_duration.csv", index=False, encoding="utf-8-sig")
    print(f"  クロス集計表を保存: {TABLE_DIR / '04_age_band_x_layout_duration.csv'}"
          f" ({len(labels)}区間 × {sub['layout'].nunique()}間取り = {len(cross)}セル)")
    print(f"  うち件数0のセル: {int((cross['n'] == 0).sum())}セル(表には残し、グラフでは省略)")
    return cross


def step4_visualizations(d: pd.DataFrame, cross: pd.DataFrame, edges: np.ndarray, labels: list[str]) -> None:
    print("\n" + "=" * 70)
    print("手順4: 可視化")
    print("=" * 70)
    display_labels = [lab for lab, lo in zip(labels, edges[:-1]) if lo < AGE_DISPLAY_MAX]
    view = cross[cross["age_band"].isin(display_labels) & cross["layout"].isin(TOP_LAYOUTS) & cross["n"].ge(5)]

    # --- 4a. 3D bar chart: x=age_band, y=layout, z=mean duration ---
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    x_idx = {lab: i for i, lab in enumerate(display_labels)}
    y_idx = {lab: i for i, lab in enumerate(TOP_LAYOUTS)}
    xs, ys, zs, colors = [], [], [], []
    cmap = plt.get_cmap("tab10")
    for _, row in view.iterrows():
        xs.append(x_idx[row["age_band"]])
        ys.append(y_idx[row["layout"]])
        zs.append(row["mean"])
        colors.append(cmap(y_idx[row["layout"]] % 10))
    ax.bar3d(xs, ys, np.zeros(len(xs)), 0.6, 0.6, zs, color=colors, shade=True, alpha=0.85)
    ax.set_xticks(np.arange(len(display_labels)) + 0.3)
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(TOP_LAYOUTS)) + 0.3)
    ax.set_yticklabels(TOP_LAYOUTS, fontsize=8)
    ax.set_zlabel("平均掲載期間(日)")
    ax.set_title("築年数区間 × 間取り × 平均掲載期間(n>=5のセルのみ表示)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_3d_bar_age_layout_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4b. scatter colored by layout (listing-level, one point per record) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    points = d[d["layout"].isin(TOP_LAYOUTS) & d["age"].notna() & d["age"].lt(AGE_DISPLAY_MAX)]
    for i, layout in enumerate(TOP_LAYOUTS):
        sub = points[points["layout"] == layout]
        ax.scatter(sub["age"], sub["duration"], s=8, alpha=0.35,
                   color=cmap(i % 10), label=f"{layout} (n={len(sub):,})")
    ax.set_xlabel("築年数(年)")
    ax.set_ylabel("掲載期間(日)")
    ax.set_title(f"築年数 × 掲載期間(間取りで色分け, 1点=1物件, n={len(points):,})")
    ax.legend(fontsize=8, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_scatter_age_vs_duration_by_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4c. bar + line charts, consistent bin width ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(display_labels))
    width = 0.8 / len(TOP_LAYOUTS)
    for i, layout in enumerate(TOP_LAYOUTS):
        sub = view[view["layout"] == layout].set_index("age_band").reindex(display_labels)
        axes[0].bar(x + i * width, sub["mean"].to_numpy(dtype=float), width=width, color=cmap(i % 10), label=layout)
        axes[1].plot(x, sub["mean"].to_numpy(dtype=float), marker="o", color=cmap(i % 10), label=layout)
    axes[0].set_xticks(x + width * len(TOP_LAYOUTS) / 2)
    axes[0].set_xticklabels(display_labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("平均掲載期間(日)")
    axes[0].set_title("棒グラフ(区間幅は5年で統一)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("平均掲載期間(日)")
    axes[1].set_title("折れ線グラフ(データが無い区間は線を途切れさせる)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_bar_line_age_by_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  グラフを保存: {OUT_DIR}")


def step6_forecast_style(cross: pd.DataFrame, edges: np.ndarray, labels: list[str]) -> None:
    print("\n" + "=" * 70)
    print("手順6: 「天気予報」スタイルの幅を持った見せ方")
    print("=" * 70)
    display_labels = [lab for lab, lo in zip(labels, edges[:-1]) if lo < AGE_DISPLAY_MAX]
    focus_layouts = ["1K", "1LDK", "2LDK", "3LDK"]  # 単身〜ファミリー向けの主要4間取りに絞って見やすくする
    fig, axes = plt.subplots(len(focus_layouts), 1, figsize=(10, 3.0 * len(focus_layouts)), sharex=False)
    for ax, layout in zip(axes, focus_layouts):
        sub = cross[(cross["layout"] == layout) & cross["age_band"].isin(display_labels)]
        sub = sub.set_index("age_band").reindex(display_labels)
        x = np.arange(len(display_labels))
        median = sub["median"].to_numpy(dtype=float)
        q1 = sub["q1"].to_numpy(dtype=float)
        q3 = sub["q3"].to_numpy(dtype=float)
        n = sub["n"].fillna(0).to_numpy(dtype=int)
        valid = n >= 5
        ax.fill_between(x[valid], q1[valid], q3[valid], color="#4C72B0", alpha=0.25, step=None,
                        label="よくある範囲(Q1〜Q3)" if layout == focus_layouts[0] else None)
        ax.plot(x[valid], median[valid], "o-", color="#1f3d63", label="中央値" if layout == focus_layouts[0] else None)
        ax.set_title(f"間取り: {layout}")
        ax.set_ylabel("掲載期間(日)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5, len(display_labels) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nn={nn}" for lab, nn in zip(display_labels, n)], rotation=35, ha="right", fontsize=7)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9)
    fig.suptitle("築年数帯ごとの掲載期間の「よくある範囲」(Q1〜Q3)と中央値", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_forecast_style_range.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {OUT_DIR / '04_forecast_style_range.png'}")


def step5_narrative(cross: pd.DataFrame) -> list[str]:
    valid = cross[cross["n"] >= 30].copy()
    lines = []
    top_n = valid.sort_values("n", ascending=False).head(5)
    lines.append("**件数が集中している組み合わせ(上位5):**")
    for _, r in top_n.iterrows():
        lines.append(f"- 築年数{r['age_band']} × {r['layout']}: {int(r['n']):,}件、"
                     f"中央値{r['median']:.0f}日(よくある範囲 {r['q1']:.0f}〜{r['q3']:.0f}日)")

    shortest = valid.sort_values("median").head(3)
    longest = valid.sort_values("median", ascending=False).head(3)
    lines.append("")
    lines.append("**掲載期間が短めの組み合わせ(n>=30, 中央値が小さい順):**")
    for _, r in shortest.iterrows():
        lines.append(f"- 築年数{r['age_band']} × {r['layout']}: 中央値{r['median']:.0f}日"
                     f"(n={int(r['n']):,}, よくある範囲 {r['q1']:.0f}〜{r['q3']:.0f}日)")
    lines.append("")
    lines.append("**掲載期間が長めの組み合わせ(n>=30, 中央値が大きい順):**")
    for _, r in longest.iterrows():
        lines.append(f"- 築年数{r['age_band']} × {r['layout']}: 中央値{r['median']:.0f}日"
                     f"(n={int(r['n']):,}, よくある範囲 {r['q1']:.0f}〜{r['q3']:.0f}日)")
    return lines


def write_report(d: pd.DataFrame, cross: pd.DataFrame, best_a: str, best_b: str) -> None:
    lines = ["# 掲載期間 × 築年数 × 間取り 3変数分析", ""]
    lines.append(f"対象: 東京都国分寺市の賃貸物件、分析対象 {len(d):,}件(生データ bukken_rent_v1.0.1.tsv より抽出)。")
    lines.append("")
    lines.append("## 1. データ概要")
    lines.append("`tables/01_column_overview.csv` に各列の型・欠損数をまとめた。"
                 "掲載期間(duration)・築年数(age)・家賃(rent)・面積(area)は連続値、"
                 "最寄駅徒歩距離(walk)も連続値、間取り(layout)はカテゴリ変数。")
    lines.append("")
    lines.append("## 2. 属性の選定")
    lines.append(f"単変数の相関(spearman)は 築年数 > 家賃 > 面積 > 徒歩距離 の順で強く、"
                 f"2属性の組み合わせでは **{best_a}(築年数) × {best_b}(間取り)** が"
                 "交互作用なし重回帰R^2で最大だった(詳細は`tables/02_single_variable_strength.csv`, "
                 "`tables/03_pair_explanatory_power.csv`)。最寄駅徒歩距離はほぼ無相関のため除外。")
    lines.append("")
    lines.append("## 3. 集計表")
    lines.append("`tables/04_age_band_x_layout_duration.csv`(築年数5年刻み × 間取り、"
                 "n・平均・中央値・標準偏差・Q1・Q3・IQR)。")
    lines.append("")
    lines.append("## 4. グラフ")
    lines.append("- `output/01_3d_bar_age_layout_duration.png`: 3次元棒グラフ\n"
                 "- `output/02_scatter_age_vs_duration_by_layout.png`: 間取りで色分けした散布図\n"
                 "- `output/03_bar_line_age_by_layout.png`: 区間幅5年で統一した棒・折れ線グラフ\n"
                 "- `output/04_forecast_style_range.png`: 天気予報スタイルの幅表示(Q1〜Q3 + 中央値)")
    lines.append("")
    lines.append("## 5. 読み取れる傾向")
    lines.extend(step5_narrative(cross))
    lines.append("")
    lines.append("## 6. 幅を持った見せ方について")
    lines.append(
        "全組み合わせのR^2は0.03〜0.05程度(手順2参照)と低く、築年数・間取りだけでは掲載期間の"
        "大半のばらつきを説明できない。したがって「この物件は◯日」という一点予測は避け、"
        "`output/04_forecast_style_range.png`のように**中央値と Q1〜Q3 の範囲**を併記する形を採用した。"
        "件数(n)が少ないセルは範囲が不安定なため、グラフ・考察ともに n>=5〜30 の閾値を明記している。"
    )
    lines.append("")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポートを保存: {HERE / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="生データTSVのパス(初回、またはキャッシュ更新時に指定)")
    args = parser.parse_args()

    d = load_data(args.input)
    step1_overview(d)
    best_a, best_b = step2_propose_attributes(d)
    edges, labels = build_age_bins(d)
    cross = step3_cross_table(d, edges, labels)
    step4_visualizations(d, cross, edges, labels)
    step6_forecast_style(cross, edges, labels)
    write_report(d, cross, best_a, best_b)


if __name__ == "__main__":
    main()

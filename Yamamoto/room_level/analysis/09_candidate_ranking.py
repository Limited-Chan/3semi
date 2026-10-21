"""同時に掲載中の類似物件内で順位付け方法を比較(修正版)。

既存の Model1/Model2 は変更しない。住戸掲載エピソードを用いて再学習する。
検証は2023年掲載開始物件の、事前に指定した複数のスナップショット。
（2023年の結果を既に見て研究設計を変更した場合、探索的な追加検証。）

修正点:
- 基準は「掲載経過日数だけ」を使う overall_km。属性が順位付けに足す情報を測る。
- 類似物件の中央値は、掲載が続いているもの(打ち切り)を無視せずKaplan-Meierで求める。
- 符号が逆だった経過日数−中央値の簡易法をやめ、類似グループのKaplan-Meierから
  「残りの掲載期間の中央値」を求める km_remaining_median に置き換えた。
- 同じ類似グループが複数のスナップショットに現れる依存を考慮し、グループ単位の
  ブートストラップで95%区間を出す。
- 統合ルールの感度を見るため --gap を指定できる(08が出した episodes_gap{N}.parquet)。

実行例: python -m analysis.09_candidate_ranking --gap 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
from src.discrete_model import fit_model, prepare_feature_frame

AUDIT_DIR = HERE / "results" / "08_unit_audit"
OUTPUT_ROOT = HERE / "results" / "09_candidate_ranking"
TRAIN_CUTOFF = pd.Timestamp("2023-01-01")
OBSERVATION_END = pd.Timestamp("2024-03-31")
SNAPSHOTS = pd.to_datetime([
    "2023-04-01", "2023-06-01", "2023-08-01",
    "2023-10-01", "2023-12-01",
])
HORIZON = 30
MIN_CANDIDATES = 3
RENT_BAND_YEN = 20000
AGE_BINS = [0, 10, 20, 30, 40, np.inf]
AGE_LABELS = ["0-10", "10-20", "20-30", "30-40", "40+"]
REMAINING_CAP_DAYS = 730
BOOTSTRAP_REPEATS = 2000
REFERENCE = "overall_km"
METHODS = [
    "knn30_median", "naive_median", "km_remaining_median",
    "overall_km", "age_layout_km", "model1", "model2",
]
RISK_METHODS = {"overall_km", "age_layout_km", "model1", "model2"}


def make_survival_table(episodes: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """学習時点より未来の掲載終了を学習に渡さない。"""
    d = episodes.loc[episodes["start_date"] < cutoff].copy()
    event_end = d["end_date"].lt(cutoff) & d["end_date"].lt(OBSERVATION_END)
    d["observed_end_date"] = d["end_date"].where(event_end, cutoff)
    d["survival_days"] = (d["observed_end_date"] - d["start_date"]).dt.days
    d["event"] = event_end.astype(int)
    return d.loc[d["survival_days"].ge(0)].copy()


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_group"] = pd.cut(
        pd.to_numeric(d["age"], errors="coerce"),
        bins=AGE_BINS, labels=AGE_LABELS, right=False,
    ).astype("string").fillna("unknown")
    d["rent_group"] = (
        (pd.to_numeric(d["rent"], errors="coerce") // RENT_BAND_YEN)
        .astype("Int64").astype("string").fillna("unknown")
    )
    d["layout_group"] = d["layout"].astype("string").fillna("unknown")
    return d


def fit_km(days, events) -> KaplanMeierFitter:
    km = KaplanMeierFitter()
    km.fit(np.asarray(days, dtype=float), event_observed=np.asarray(events, dtype=int))
    return km


def km_median(days, events, default: float) -> float:
    """掲載が続いている物件(打ち切り)も含めたKaplan-Meierの中央値。"""
    value = float(fit_km(days, events).median_survival_time_)
    return value if np.isfinite(value) else default


class SimilarListingMedian:
    """当初案の近似: 同じ間取りの過去掲載から数値条件が近い30件を探し、
    その30件のKaplan-Meier中央値を使う(打ち切りを無視しない)。"""
    COLUMNS = ["age", "rent", "area", "walk_distance"]

    def __init__(self, train: pd.DataFrame, neighbors: int = 30):
        values = train[self.COLUMNS].apply(pd.to_numeric, errors="coerce")
        self.center = values.median().fillna(0)
        self.scale = (values.quantile(.75) - values.quantile(.25)).replace(0, 1).fillna(1)
        self.overall_median = km_median(train["survival_days"], train["event"], REMAINING_CAP_DAYS)
        self.models: dict[str, tuple[NearestNeighbors, np.ndarray, np.ndarray]] = {}
        keyed = train.assign(layout_key=train["layout"].astype("string").fillna("unknown"))
        for layout, group in keyed.groupby("layout_key"):
            x = (group[self.COLUMNS].apply(pd.to_numeric, errors="coerce")
                 .fillna(self.center) - self.center) / self.scale
            model = NearestNeighbors(n_neighbors=min(neighbors, len(group)))
            model.fit(x.to_numpy(dtype=float))
            self.models[str(layout)] = (
                model, group["survival_days"].to_numpy(dtype=float),
                group["event"].to_numpy(dtype=int),
            )

    def predict_median(self, candidates: pd.DataFrame) -> np.ndarray:
        medians = np.full(len(candidates), self.overall_median, dtype=float)
        layouts = candidates["layout"].astype("string").fillna("unknown").astype(str)
        for layout in layouts.unique():
            pair = self.models.get(layout)
            if pair is None:
                continue
            model, days, events = pair
            positions = np.flatnonzero(layouts.to_numpy() == layout)
            x = (candidates.iloc[positions][self.COLUMNS]
                 .apply(pd.to_numeric, errors="coerce").fillna(self.center)
                 - self.center) / self.scale
            neighbor_idx = model.kneighbors(x.to_numpy(dtype=float), return_distance=False)
            for position, idx in zip(positions, neighbor_idx):
                medians[position] = km_median(days[idx], events[idx], self.overall_median)
        return medians


def km_risk(km: KaplanMeierFitter, elapsed: int, horizon: int) -> float:
    s_now = float(km.predict(elapsed))
    if s_now <= 0:
        return np.nan
    s_after = float(km.predict(elapsed + horizon))
    return float(np.clip(1 - s_after / s_now, 0, 1))


def km_remaining_median(km: KaplanMeierFitter, elapsed: int) -> float:
    """いま elapsed 日目の物件が、あと何日で半数が終わるか(同じグループのKMから)。"""
    s_now = float(km.predict(elapsed))
    if not s_now > 0:
        return np.nan
    curve = km.survival_function_.iloc[:, 0]
    hit = curve[(curve.index > elapsed) & (curve.to_numpy() <= 0.5 * s_now)]
    return float(hit.index[0] - elapsed) if len(hit) else float(REMAINING_CAP_DAYS)


def predict_variable_elapsed(model, candidates: pd.DataFrame, use_combo: bool) -> np.ndarray:
    """物件ごとに異なる現在掲載日数を使って30日以内リスクを一括計算。"""
    n = len(candidates)
    if n == 0:
        return np.array([])
    positions = np.repeat(np.arange(n), HORIZON)
    future = candidates.iloc[positions][[
        "age", "rent", "area", "walk_distance", "layout", "station", "start_month"
    ]].reset_index(drop=True).copy()
    future["time_day"] = (
        np.repeat(candidates["elapsed_day"].to_numpy(dtype=int), HORIZON)
        + np.tile(np.arange(1, HORIZON + 1), n)
    )
    x = prepare_feature_frame(future, use_age_layout_combo=use_combo)
    daily = model.predict_proba(x)[:, 1].reshape(n, HORIZON)
    return 1 - np.exp(np.log1p(-np.clip(daily, 0, 1 - 1e-12)).sum(axis=1))


def create_candidates(episodes: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    # 2023年掲載開始という独立した物件群だけを使用。
    active = episodes.loc[
        (episodes["start_year"] == 2023)
        & (episodes["start_date"] <= snapshot)
        & (episodes["end_date"] > snapshot)
    ].copy()
    active["elapsed_day"] = (snapshot - active["start_date"]).dt.days.astype(int)
    # 観測期間内に30日後の正解が確定する例だけ使用。
    if not snapshot + pd.Timedelta(days=HORIZON) < OBSERVATION_END:
        raise ValueError(f"スナップショット {snapshot.date()} は観測終了日が近すぎます")
    active["target"] = (
        (active["end_date"] <= snapshot + pd.Timedelta(days=HORIZON))
        & (active["end_date"] < OBSERVATION_END)
    ).astype(int)
    active["snapshot"] = snapshot.date().isoformat()
    return add_bins(active)


def add_baselines(candidates: pd.DataFrame, global_km: KaplanMeierFitter,
                  subgroup_km: dict[tuple[str, str], KaplanMeierFitter],
                  group_medians: dict[tuple[str, str], float],
                  global_median: float) -> pd.DataFrame:
    d = candidates.copy()
    p_overall, p_group, remaining, naive = [], [], [], []
    for row in d.itertuples():
        elapsed = int(row.elapsed_day)
        overall = km_risk(global_km, elapsed, HORIZON)
        key = (str(row.age_group), str(row.layout_group))
        grouped = subgroup_km.get(key)
        grouped_risk = km_risk(grouped, elapsed, HORIZON) if grouped else np.nan
        left = km_remaining_median(grouped, elapsed) if grouped else np.nan
        if not np.isfinite(grouped_risk):
            grouped_risk = overall
        if not np.isfinite(left):
            left = km_remaining_median(global_km, elapsed)
        p_overall.append(overall)
        p_group.append(grouped_risk)
        remaining.append(-left)  # 残りが短いほど優先
        naive.append(float(-group_medians.get(key, global_median)))
    d["overall_km"] = p_overall
    d["age_layout_km"] = p_group
    d["km_remaining_median"] = remaining
    d["naive_median"] = naive
    return d


def method_metrics(groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """候補集合内ペアAUCと先頭の実終了率。"""
    rows: list[dict] = []
    ranked_outputs = []
    for (snapshot, segment), group in groups.groupby(["snapshot", "segment"], sort=True):
        if len(group) < MIN_CANDIDATES:
            continue
        y = group["target"].to_numpy(dtype=int)
        for method in METHODS:
            score = group[method].to_numpy(dtype=float)
            if not np.isfinite(score).all():
                continue
            order = np.argsort(-score, kind="stable")
            ordered = group.iloc[order].copy()
            ordered["method"] = method
            ordered["candidate_rank"] = np.arange(1, len(ordered) + 1)
            split = int(np.ceil(len(ordered) / 2))
            ordered["within_set_priority"] = np.where(
                ordered["candidate_rank"] <= split, "高（候補内）", "低（候補内）"
            )
            if split < len(ordered):
                upper = float(ordered.iloc[split - 1][method])
                lower = float(ordered.iloc[split][method])
                if np.isclose(upper, lower, rtol=0, atol=1e-12):
                    tied = np.isclose(ordered[method].to_numpy(dtype=float), upper, rtol=0, atol=1e-12)
                    ordered.loc[tied, "within_set_priority"] = "同順位"
            ordered["score_kind"] = (
                "30日以内の予測リスク" if method in RISK_METHODS
                else "過去掲載期間を使った順位用スコア（日数）"
            )
            detail_cols = [c for c in ["first_source_id", "layout", "age", "rent", "area", "station", "unit_key"]
                           if c in ordered.columns]
            ranked_outputs.append(ordered[[
                "snapshot", "segment", "id", *detail_cols, "method", "candidate_rank",
                "within_set_priority", "score_kind", "elapsed_day", "target", method
            ]].rename(columns={method: "risk_or_score"}))
            at_top = np.isclose(score, float(score.max()), rtol=0, atol=1e-12)
            auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
            rows.append({
                "snapshot": snapshot, "segment": segment, "method": method,
                "n_candidates": len(group), "n_ended": int(y.sum()),
                "top1_actual_rate": float(y[at_top].mean()),
                "random_top1_rate": float(y.mean()),
                "within_group_auc": auc,
                "positive_negative_pairs": int(y.sum() * (len(y) - y.sum())),
                "score_span": float(score.max() - score.min()),
            })
    metrics = pd.DataFrame(rows)
    ranks = pd.concat(ranked_outputs, ignore_index=True) if ranked_outputs else pd.DataFrame()
    return metrics, ranks


def summarize_with_intervals(metrics: pd.DataFrame, repeats: int = BOOTSTRAP_REPEATS,
                             seed: int = 20261012) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ペア数で重みを付けた候補内AUC。同じ類似グループが複数のスナップショットに
    現れる依存を考え、グループ(segment)単位で復元抽出して95%区間を出す。"""
    mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
    auc = mixed.pivot(index=["snapshot", "segment"], columns="method", values="within_group_auc")
    pairs = (mixed.pivot(index=["snapshot", "segment"], columns="method", values="positive_negative_pairs")
             .reindex(auc.index))
    auc = auc.dropna(axis=1)
    pairs = pairs[auc.columns]
    methods = list(auc.columns)
    a, w = auc.to_numpy(dtype=float), pairs.to_numpy(dtype=float)
    segments = auc.index.get_level_values("segment").to_numpy()
    by_segment = {s: np.flatnonzero(segments == s) for s in np.unique(segments)}
    keys = list(by_segment)
    rng = np.random.default_rng(seed)
    point = (a * w).sum(axis=0) / w.sum(axis=0)
    samples = np.empty((repeats, len(methods)))
    for r in range(repeats):
        chosen = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_segment[keys[i]] for i in chosen])
        samples[r] = (a[idx] * w[idx]).sum(axis=0) / w[idx].sum(axis=0)
    ref = methods.index(REFERENCE)
    rows, diff_rows = [], []
    for j, method in enumerate(methods):
        lo, hi = np.percentile(samples[:, j], [2.5, 97.5])
        g = metrics.loc[metrics["method"] == method]
        rows.append({
            "method": method, "candidate_sets": len(g), "mixed_outcome_sets": int(len(auc)),
            "positive_negative_pairs": int(w[:, j].sum()),
            "within_set_pair_weighted_auc": float(point[j]),
            "auc_ci_low": float(lo), "auc_ci_high": float(hi),
            "mean_top1_actual_rate": float(g["top1_actual_rate"].mean()),
            "mean_random_top1_rate": float(g["random_top1_rate"].mean()),
        })
        if method != REFERENCE:
            delta = samples[:, j] - samples[:, ref]
            dlo, dhi = np.percentile(delta, [2.5, 97.5])
            diff_rows.append({
                "method": method, "reference": REFERENCE,
                "auc_difference": float(point[j] - point[ref]),
                "ci_low": float(dlo), "ci_high": float(dhi),
                "clusters": len(keys),
            })
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=1, help="08で作った episodes_gap{N}.parquet の N")
    gap = parser.parse_args().gap
    source = AUDIT_DIR / f"episodes_gap{gap}.parquet"
    if not source.exists():
        raise FileNotFoundError("先に analysis/08_unit_audit.py を実行してください")
    output = OUTPUT_ROOT / f"gap{gap}"
    output.mkdir(parents=True, exist_ok=True)
    episodes = add_bins(pd.read_parquet(source))
    print(f"統合ルール: max_gap_days={gap}")

    train = episodes.loc[episodes["start_year"].isin([2020, 2021, 2022])].copy()
    train = make_survival_table(train, TRAIN_CUTOFF)
    print(f"学習対象: {len(train):,} 掲載エピソード（終了イベント: {int(train['event'].sum()):,}）")
    model1, _ = fit_model(train, use_age_layout_combo=False)
    model2, _ = fit_model(train, use_age_layout_combo=True)
    overall_km = fit_km(train["survival_days"], train["event"])
    global_median = km_median(train["survival_days"], train["event"], REMAINING_CAP_DAYS)
    subgroup_km, group_medians = {}, {}
    for key, sub in train.groupby(["age_group", "layout_group"], dropna=False):
        k = (str(key[0]), str(key[1]))
        subgroup_km[k] = fit_km(sub["survival_days"], sub["event"])
        group_medians[k] = km_median(sub["survival_days"], sub["event"], global_median)

    knn_reference = SimilarListingMedian(train, neighbors=30)
    raw_groups = []
    for snapshot in SNAPSHOTS:
        candidates = create_candidates(episodes, snapshot)
        if candidates.empty:
            continue
        candidates["segment"] = (
            candidates["layout_group"].astype(str) + " | age="
            + candidates["age_group"].astype(str) + " | rent20k="
            + candidates["rent_group"].astype(str)
        )
        candidates = add_baselines(candidates, overall_km, subgroup_km, group_medians, global_median)
        candidates["knn30_median"] = -knn_reference.predict_median(candidates)
        candidates["model1"] = predict_variable_elapsed(model1, candidates, False)
        candidates["model2"] = predict_variable_elapsed(model2, candidates, True)
        raw_groups.append(candidates)
        print(f"{snapshot.date()}: 対象 {len(candidates):,}件, 類似グループ {candidates['segment'].nunique():,}個")
    if not raw_groups:
        raise ValueError("評価に使える物件がありません")
    all_candidates = pd.concat(raw_groups, ignore_index=True)
    all_candidates.to_csv(output / "snapshot_candidates.csv", index=False, encoding="utf-8-sig")
    all_finite = all_candidates.groupby(["snapshot", "segment"])[METHODS].transform(
        lambda series: bool(np.isfinite(series.to_numpy(dtype=float)).all())
    ).all(axis=1)
    eligible = all_candidates.loc[all_finite].copy()
    metrics, ranks = method_metrics(eligible)
    if metrics.empty:
        print("比較できる3件以上の類似候補集合がありません。segment条件を勝手に緩めず、結果を確認してください。")
        return
    metrics.to_csv(output / "candidate_set_metrics.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(output / "candidate_rankings.csv", index=False, encoding="utf-8-sig")
    summary, differences = summarize_with_intervals(metrics)
    summary.to_csv(output / "ranking_comparison.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(output / "ranking_differences.csv", index=False, encoding="utf-8-sig")
    print("\n同時に掲載中の類似物件での比較(95%区間はグループ単位のブートストラップ):\n",
          summary.to_string(index=False))
    print(f"\n基準({REFERENCE})との差:\n", differences.to_string(index=False))
    print("\n注意: 2023年結果を見た後の追加分析なので、独立な未使用テストとは呼ばないでください。")
    print("掲載終了は成約ではありません。候補内の『低』も絶対的に安全な意味ではありません。")


if __name__ == "__main__":
    main()

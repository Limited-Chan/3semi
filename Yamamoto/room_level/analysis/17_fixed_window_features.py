"""履歴の見る範囲を過去365日に固定し、学習は2021〜2022年開始だけにする(公平な比較)。
(2020年開始の物件だけ過去が見えない不公平を避けるため。多摩の2020〜2023年開始だけを抽出しているので、履歴・需給は窓を固定する)

使う情報を増やすと、似た候補の中の当たりが良くなるか(探索的)。

足す情報(すべて「その部屋の掲載開始日」時点で分かるものだけ。未来は使わない):
  hist   部屋の履歴: 前の掲載の回数・長さ・終了からの日数
  bldg   同じ建物で、その日に掲載中だった他の部屋の数
  supply 近くの需給: 同じ駅×間取り / 同じ市×間取り×家賃帯で、その日に掲載中の他の部屋の数
  x      現況・入居可能時期(掲載の都合を表す可能性が高い。別枠で扱う)
学習: 多摩地域(東京都の23区以外)の2021〜2022年開始・部屋単位エピソード(統合ルール1日)
評価: 2023年の5つの日付で掲載中の候補。(a)国分寺市 (b)近隣10市+国分寺市。
      候補内AUC(15と同じ枠)+グループ単位ブートストラップ。基準は base(Model 2相当)。
モデル: ロジスティック(LR)と、勾配ブースティング(HGB: 特徴量どうしの組み合わせも拾える)。
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # リポジトリの直下(data/ がある場所)
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
m = importlib.import_module("analysis.09_candidate_ranking")
s9b = importlib.import_module("analysis.09b_segment_sensitivity")
s15 = importlib.import_module("analysis.15_scope_comparison")
from src.data import prepare
from src.discrete_model import BASE_CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_feature_frame
from room_lib.listing_episodes import build_episodes_fast

OUT = HERE / "results" / "17_fixed_window_features"
EXTRAS_PQ = ROOT / "data" / "prepared" / "tama_extras.parquet"
CAP_DAY = s15.CAP_DAY
HORIZONS = [7, 14, 30]
TARGET_CITY = s15.TARGET_CITY
NEAR = s15.NEAR
BASE_COLS = ["age", "rent", "area", "walk_distance", "layout", "station", "start_month", "city"]
FEATURES = {
    "hist": ["prev_n", "prev_len_log", "since_prev_end_log"],
    "bldg": ["bldg_active_log"],
    "supply": ["sup_station_layout_log", "sup_city_layout_rent_log"],
    "x": [],  # 数値ではなくカテゴリ。下で扱う
}
X_CAT = ["genkyo_label", "usable_status_label"]
SCENARIOS = s9b.SCENARIOS
rng = np.random.default_rng(20261014)


# ---------------------------------------------------------------- 特徴量の計算
def _active_count(group_ids: np.ndarray, start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    """同じグループで、日 t に掲載中(start <= t < end)の件数。自分も含む。"""
    big = 100_000
    s_key = np.sort(group_ids * big + start)
    e_key = np.sort(group_ids * big + end)
    q = group_ids * big + t
    return np.searchsorted(s_key, q, side="right") - np.searchsorted(e_key, q, side="right")


def add_features(ep: pd.DataFrame) -> pd.DataFrame:
    d = ep.copy().reset_index(drop=True)
    day0 = pd.Timestamp("2015-01-01")
    s = (d["start_date"] - day0).dt.days.to_numpy()
    e = (d["end_date"] - day0).dt.days.to_numpy()
    own = (e > s).astype(int)  # 自分自身が数に入るのは、start<end のとき

    # 部屋の履歴(前の掲載が掲載開始日までに終わっているものだけ使う)
    # building_id が 1 / 3 は仮の値(多摩の15%)。件数が極端に多い鍵も含め「ID不明」として扱う
    key = d["unit_key"].astype("string").fillna("")
    bad_key = key.map(key.value_counts()).gt(100) | d["building_id"].astype(str).isin(["1", "3"])
    d["_k"] = key.where(~bad_key, "")
    d["_s"], d["_e"] = s, e
    order = d.sort_values(["_k", "_s"], kind="stable").index
    o = d.loc[order]
    uk = o["_k"].to_numpy(dtype=str)
    has = uk != ""
    same = has & np.concatenate([[False], uk[1:] == uk[:-1]])
    prev_e = o["_e"].shift().to_numpy(dtype=float)
    prev_s = o["_s"].shift().to_numpy(dtype=float)
    ended = same & (prev_e <= o["_s"].to_numpy())
    LOOK = 365
    ended = ended & ((o["_s"].to_numpy() - prev_e) <= LOOK)
    gid_k = pd.factorize(o["_k"])[0].astype(np.int64)
    big = 100_000
    starts = np.sort(gid_k * big + o["_s"].to_numpy())
    q_hi = gid_k * big + o["_s"].to_numpy() - 1
    q_lo = gid_k * big + o["_s"].to_numpy() - LOOK - 1
    cnt = np.searchsorted(starts, q_hi, side="right") - np.searchsorted(starts, q_lo, side="right")
    d.loc[order, "prev_n"] = np.where(has, cnt, 0)
    d.loc[order, "prev_len_log"] = np.where(ended, np.log1p(np.where(ended, prev_e - prev_s, 0)), np.nan)
    d.loc[order, "since_prev_end_log"] = np.where(ended, np.log1p(np.where(ended, o["_s"].to_numpy() - prev_e, 0)), np.nan)

    def count_by(keys: list[str]) -> np.ndarray:
        gid = d.groupby(keys, dropna=False, sort=False).ngroup().to_numpy().astype(np.int64)
        return np.log1p(np.maximum(_active_count(gid, s, e, s) - own, 0))

    bkey = d["building_id"].fillna("").astype(str)
    d["_b"] = np.where((bkey == "") | bkey.isin(["1", "3"]), "nb" + d.index.astype(str), bkey)
    d["bldg_active_log"] = count_by(["_b"])
    d["sup_station_layout_log"] = count_by(["station", "layout"])
    d["sup_city_layout_rent_log"] = count_by(["city", "layout", "rent_group"])
    return d.drop(columns=["_s", "_e", "_b", "_k"])


def load_extras() -> pd.DataFrame:
    if EXTRAS_PQ.exists():
        return pd.read_parquet(EXTRAS_PQ)
    t0 = time.time()
    chunks = []
    for ch in pd.read_csv(s15.TSV, sep="\t", usecols=["id", "addr1_1_name", "addr1_2_name", *X_CAT], dtype=str,
                          chunksize=300_000, encoding="utf-8", keep_default_na=False):
        f = ch[(ch["addr1_1_name"] == "東京都") & (~ch["addr1_2_name"].str.endswith("区"))]
        if len(f):
            chunks.append(f[["id", *X_CAT]])
    d = pd.concat(chunks, ignore_index=True)
    d["id"] = d["id"].astype("string").str.normalize("NFKC").str.strip()
    d = d.drop_duplicates("id")
    d.to_parquet(EXTRAS_PQ, index=False)
    print(f"現況・入居可能時期を抽出: {len(d):,}件 ({time.time() - t0:.0f}秒)")
    return d


# ---------------------------------------------------------------- モデル
def person_period(train: pd.DataFrame, cols: list[str], target_neg: int = 2_500_000):
    days = train["survival_days"].astype(int).to_numpy()
    observed = np.minimum(days, CAP_DAY)
    event = (train["event"].to_numpy(int) == 1) & (days <= CAP_DAY)
    rows_per = observed + 1
    p_neg = min(1.0, target_neg / max(int(rows_per.sum() - event.sum()), 1))
    frames, ys, ws = [], [], []
    local = np.random.default_rng(1)  # 全モデルで同じ行を使う(比較を公平にする)
    for start in range(0, len(train), 20000):
        sl = slice(start, start + 20000)
        n = rows_per[sl]
        idx = np.repeat(np.arange(len(n)), n)
        t = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        y = np.zeros(len(idx), dtype=int)
        y[np.cumsum(n) - 1] = event[sl].astype(int)
        keep = (y == 1) | (local.random(len(y)) < p_neg)
        part = train.iloc[sl][cols].reset_index(drop=True).iloc[idx[keep]].reset_index(drop=True)
        part["time_day"] = t[keep]
        frames.append(part)
        ys.append(y[keep])
        ws.append(np.where(y[keep] == 1, 1.0, 1.0 / p_neg))
    return pd.concat(frames, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


class Hazard:
    """kind='lr' はModel 2と同じ作り(+足した特徴量)。kind='hgb' は勾配ブースティング。"""

    def __init__(self, kind: str, groups: list[str]):
        self.kind, self.groups = kind, groups
        self.num_extra = [c for g in groups if g != "x" for c in FEATURES[g]]
        self.cat_extra = X_CAT if "x" in groups else []
        self.cols = BASE_COLS + self.num_extra + self.cat_extra

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = prepare_feature_frame(df, use_age_layout_combo=True)
        x["city"] = df["city"].astype("string").fillna("欠損").astype(str).to_numpy()
        for c in self.num_extra:
            x[c] = df[c].to_numpy(dtype=float)
        for c in self.cat_extra:
            x[c] = df[c].astype("string").fillna("欠損").astype(str).to_numpy()
        return x

    def fit(self, train: pd.DataFrame) -> "Hazard":
        rows, y, w = person_period(train, self.cols)
        x = self._frame(rows)
        t0 = time.time()
        if self.kind == "lr":
            cat = BASE_CATEGORICAL_FEATURES + ["age_layout_combo", "city"] + self.cat_extra
            pre = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                 NUMERIC_FEATURES + self.num_extra),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ])
            self.pipe = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=500, C=1.0))])
            self.pipe.fit(x, y, m__sample_weight=w)
        else:
            top = x["station"].value_counts().index[:150]
            self.top_station = set(top)
            x = self._hgb_view(x)
            cats = ["layout", "station", "city", *self.cat_extra]
            pre = ColumnTransformer([
                ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
            ], remainder="passthrough")
            self.num_cols = [c for c in x.columns if c not in cats]
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6, min_samples_leaf=200, l2_regularization=1.0,
                categorical_features=list(range(len(cats))), early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=0)
            self.pipe = Pipeline([("pre", pre), ("m", model)])
            self.pipe.fit(x, y, m__sample_weight=w)
            print(f"    HGB 反復回数 {self.pipe['m'].n_iter_}")
        print(f"    学習 {time.time() - t0:.0f}秒 ({self.kind}, {'+'.join(self.groups) or 'base'})")
        return self

    def _hgb_view(self, x: pd.DataFrame) -> pd.DataFrame:
        v = x[["layout", "station", "city", *self.cat_extra, "time_day", "age", "rent", "area", "walk_distance",
               "start_month", *self.num_extra]].copy()
        v["station"] = v["station"].where(v["station"].isin(self.top_station), "他")
        v["layout"] = v["layout"].astype("string").fillna("欠損").astype(str)
        return v

    def risk(self, cand: pd.DataFrame, horizon: int) -> np.ndarray:
        n = len(cand)
        pos = np.repeat(np.arange(n), horizon)
        fut = cand.iloc[pos][self.cols].reset_index(drop=True).copy()
        day = np.repeat(cand["elapsed_day"].to_numpy(dtype=int), horizon) + np.tile(np.arange(1, horizon + 1), n)
        fut["time_day"] = np.minimum(day, CAP_DAY)
        x = self._frame(fut)
        if self.kind == "hgb":
            x = self._hgb_view(x)
        daily = self.pipe.predict_proba(x)[:, 1].reshape(n, horizon)
        return 1 - np.exp(np.log1p(-np.clip(daily, 0, 1 - 1e-12)).sum(axis=1))


MODELS = {
    "base": ("lr", []),
    "hist": ("lr", ["hist"]),
    "bldg": ("lr", ["bldg"]),
    "supply": ("lr", ["supply"]),
    "all3": ("lr", ["hist", "bldg", "supply"]),
    "all3x": ("lr", ["hist", "bldg", "supply", "x"]),
    "hgb_base": ("hgb", []),
    "hgb_all3": ("hgb", ["hist", "bldg", "supply"]),
    "hgb_all3x": ("hgb", ["hist", "bldg", "supply", "x"]),
}


def evaluate(cand: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diffs = [], []
    names = list(MODELS)
    m.METHODS, m.REFERENCE = names, "base"
    for h in HORIZONS:
        d = cand.copy()
        d["target"] = d[f"target_{h}"]
        for n in names:
            d[n] = d[f"{n}_{h}"]
        for scen, keys in SCENARIOS.items():
            dd = d.copy()
            seg_keys = ["city", *keys]
            dd["segment"] = dd[seg_keys].astype(str).agg(" | ".join, axis=1)
            metrics, _ = m.method_metrics(dd)
            mixed = metrics.loc[metrics["positive_negative_pairs"] > 0]
            row = {"評価対象": label, "予測期間(日)": h, "候補の似かた": scen,
                   "対の数": int(mixed.loc[mixed["method"] == "base", "positive_negative_pairs"].sum())}
            for n in names:
                g = mixed.loc[mixed["method"] == n]
                row[n] = float(np.average(g["within_group_auc"], weights=g["positive_negative_pairs"]))
            rows.append(row)
            if scen.startswith(("A", "D")):
                _, df_ = m.summarize_with_intervals(metrics, repeats=1000)
                for _, r in df_.iterrows():
                    diffs.append({"評価対象": label, "予測期間(日)": h, "候補の似かた": scen[:1], "方式": r["method"],
                                  "baseとの差": r["auc_difference"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = prepare(s15.extract_tama())
    raw["city"] = raw["addr1_2_name"]
    ep = build_episodes_fast(raw, max_gap_days=1)
    ep["city"] = ep["addr1_2_name"]
    ep = m.add_bins(ep)
    ep["first_source_id"] = ep["first_source_id"].astype("string").str.normalize("NFKC").str.strip()
    ep = ep.merge(load_extras().rename(columns={"id": "first_source_id"}), on="first_source_id", how="left")
    ep = add_features(ep)
    print(f"部屋エピソード {len(ep):,}件, 特徴量付与 ({time.time() - t0:.0f}秒)")
    print(ep[[c for g in ("hist", "bldg", "supply") for c in FEATURES[g]]].describe().round(2).T.to_string())
    print("現況・入居可能時期の付与率:", f"{ep[X_CAT].notna().all(axis=1).mean():.1%}")

    pool = ep.loc[ep["start_year"].isin([2021, 2022])]
    train = m.make_survival_table(pool.copy(), m.TRAIN_CUTOFF)
    print(f"学習: {len(train):,}件(終了 {int(train['event'].sum()):,})")
    models = {}
    for name, (kind, groups) in MODELS.items():
        print(f"[{name}]")
        models[name] = Hazard(kind, groups).fit(train)

    evals = ep.loc[ep["city"].isin(NEAR)]
    frames = []
    for snapshot in m.SNAPSHOTS:
        c = evals.loc[(evals["start_year"] == 2023) & (evals["start_date"] <= snapshot) & (evals["end_date"] > snapshot)].copy()
        c["elapsed_day"] = (snapshot - c["start_date"]).dt.days.astype(int)
        for h in HORIZONS:
            c[f"target_{h}"] = ((c["end_date"] <= snapshot + pd.Timedelta(days=h)) & (c["end_date"] < m.OBSERVATION_END)).astype(int)
        c["snapshot"] = snapshot.date().isoformat()
        c = m.add_bins(c)
        for name, model in models.items():
            for h in HORIZONS:
                c[f"{name}_{h}"] = model.risk(c, h)
        frames.append(c)
        print(f"{snapshot.date()}: 候補 {len(c):,}件 ({time.time() - t0:.0f}秒)")
    cand = pd.concat(frames, ignore_index=True)
    keep = [c for c in cand.columns if c.endswith(("_7", "_14", "_30"))] + [
        "snapshot", "id", "city", "layout_group", "age_group", "rent_group", "elapsed_day", "unit_key"]
    cand[keep].to_parquet(OUT / "candidates_2023.parquet", index=False)

    tabs, dfs = [], []
    for label, sub in [("国分寺市", cand[cand["city"] == TARGET_CITY]), ("近隣10市+国分寺市", cand)]:
        t, df_ = evaluate(sub, label)
        tabs.append(t)
        dfs.append(df_)
    table, diffs = pd.concat(tabs), pd.concat(dfs)
    table.to_csv(OUT / "auc_by_model.csv", index=False, encoding="utf-8-sig")
    diffs.to_csv(OUT / "auc_diff_vs_base.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print("\n=== 候補内AUC ===")
    print(table.round(3).to_string(index=False))
    print("\n=== baseとの差(95%区間) ===")
    print(diffs.round(3).to_string(index=False))
    print(f"\n総時間 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()

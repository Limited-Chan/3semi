"""発表スライドの図を、データから直接計算して描く(表を写して描くのではなく、ここで計算した結果をそのまま図にする)。

入力(すべて分析の途中で作ったデータ。手で入れた数字は使わない):
  data/prepared/kokubunji_raw.parquet                       国分寺市の掲載(2020〜2023年開始、22,332件)
  results/25_overall_ci_baselines/candidates_2022.parquet   2022年の候補(2021年開始で学習した確率つき)
  results/17_fixed_window_features/candidates_2023.parquet  2023年の候補(2021〜2022年開始で学習した確率つき)
  results/26_holdout_2024q1/candidates_2024q1.parquet       2024年1〜3月の候補(同じ確率の式、一度も使っていない)
出力:
  figures/31a〜31h_*.png                 スライドに貼る図(論文のように、出典は本文の図の説明に書く)
  summary_tables/31_slide_numbers.csv     図に描いた数字の一覧(これまでの表と照らし合わせられる)
計算のしかたは、これまでの分析(19・20・21・22・25・26・28・30)と同じ。
AUCは順位から計算する(同じ点は半分当たり。scikit-learn の roc_auc_score と同じ値)。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # リポジトリの直下(data/ がある場所)
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
from src.data import load_prepared  # noqa: E402

FIG, TAB = HERE / "figures", HERE / "summary_tables"
C22 = HERE / "results" / "25_overall_ci_baselines" / "candidates_2022.parquet"
C23 = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
C24 = HERE / "results" / "26_holdout_2024q1" / "candidates_2024q1.parquet"
END = pd.Timestamp("2024-03-31")
T = 0.15
SIM_A = ["city", "layout_group", "age_group", "rent_group"]
SIM_D = ["city", "layout_group"]
SRC = "出典: Yamamoto/room_level/analysis/31_slide_figures.py"
INK, ACC, BLUE, MID, GRAY, PINK, LACC = "#222222", "#D55E00", "#0072B2", "#8C8C8C", "#BDBDBD", "#222222", "#E69F00"  # 色は Okabe-Ito(色覚に配慮した配色)
STEEL, LSTEEL = "#5B7C99", "#9DB4C8"
plt.rcParams.update({
    "font.family": ["Hiragino Sans", "sans-serif"], "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
    "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
    "xtick.color": "#333333", "ytick.color": "#333333", "legend.frameon": False, "figure.facecolor": "white",
})
DPI = 150
NUMS: list[dict] = []


def note(fig_name: str, item: str, value: float) -> None:
    NUMS.append({"図": fig_name, "項目": item, "値": float(value)})


def new(w: int, h: int, ncols: int = 1, ratios=None):
    return plt.subplots(1, ncols, figsize=(w / DPI, h / DPI), gridspec_kw={"width_ratios": ratios} if ratios else None)


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_auc(d: pd.DataFrame, score: np.ndarray, y: np.ndarray, keys: list[str]) -> float:
    seg = d[["snapshot", *keys]].astype(str).agg("|".join, axis=1).to_numpy()
    num = den = 0.0
    for idx in pd.Series(np.arange(len(d))).groupby(seg).indices.values():
        yy = y[idx]
        npos = yy.sum()
        nneg = len(yy) - npos
        if npos == 0 or nneg == 0:
            continue
        num += npos * nneg * auc(yy, score[idx])
        den += npos * nneg
    return num / den


def members(d: pd.DataFrame) -> list[np.ndarray]:
    codes, uniq = pd.factorize(d["unit_key"].astype(str))
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0, True])
    return [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


# ---------- 31a 掲載日数 ----------
def fig_duration() -> None:
    d = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    s = d[d["start_year"].between(2020, 2023) & d["valid_complete_duration"]].copy()
    dur, ev = s["duration"].to_numpy(float), (s["end_date"] < END).to_numpy()
    times = np.unique(dur[ev])
    at_risk = np.array([(dur >= t).sum() for t in times])
    deaths = np.array([((dur == t) & ev).sum() for t in times])
    surv = np.cumprod(1 - deaths / at_risk)

    def ended(t: float) -> float:
        k = np.searchsorted(times, t, side="right") - 1
        return 0.0 if k < 0 else 1 - surv[k]

    bins = [-1, 6, 13, 29, 59, 89, 179, 10**6]
    labs = ["0〜6", "7〜13", "14〜29", "30〜59", "60〜89", "90〜179", "180〜"]
    share = 100 * pd.cut(s["duration"], bins=bins, labels=labs).value_counts(normalize=True, sort=False)
    note("31a", "件数", len(s))
    note("31a", "中央値(日)", np.median(dur))
    note("31a", "平均(日)", dur.mean())
    for k, v in share.items():
        note("31a", f"割合 {k}日", v)
    for t in [7, 14, 30, 90]:
        note("31a", f"{t}日後までに終了(KM)", 100 * ended(t))

    fig, ax = new(1664, 520, 2)
    a = ax[0]
    for i, v in enumerate(share.to_numpy()):
        a.bar(i, v, 0.72, color=STEEL)
        a.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=12)
    a.set_xticks(range(len(labs)))
    a.set_xticklabels(labs, fontsize=10.5)
    a.set_xlabel("掲載日数(日)", fontsize=12)
    a.set_ylabel("物件の割合(%)", fontsize=12)
    a.set_ylim(0, 36)
    a.set_title(f"掲載日数ごとの物件の割合({len(s):,}件)", loc="left", fontsize=12)
    b = ax[1]
    grid = np.arange(0, 181)
    yv = [100 * ended(t) for t in grid]
    b.fill_between(grid, yv, color=STEEL, alpha=0.12)
    b.plot(grid, yv, color=INK, lw=3)
    for t, dy in [(7, -9), (30, -12), (90, -10)]:
        b.plot(t, 100 * ended(t), "o", color=INK, ms=6)
        b.annotate(f"{t}日目 {100 * ended(t):.0f}%", (t, 100 * ended(t)), xytext=(t + 6, 100 * ended(t) + dy), fontsize=12, color=INK)
    b.axhline(50, color=GRAY, ls="--", lw=1)
    b.text(178, 52, f"真ん中の値 {np.median(dur):.0f}日、平均 {dur.mean():.0f}日", ha="right", color="#444444", fontsize=12)
    b.set_xlim(0, 182)
    b.set_xticks([0, 30, 60, 90, 120, 150, 180])
    b.set_ylim(0, 100)
    b.set_xlabel("掲載を始めてから何日目か", fontsize=12)
    b.set_ylabel("終わった物件の割合(%)", fontsize=12)
    b.set_title("終わった割合の推移(Kaplan–Meier法)", loc="left", fontsize=12)
    b.grid(color="#e5e5e5")
    fig.tight_layout()
    save(fig, "31a_duration.png")


# ---------- 31h ROC曲線 ----------
def fig_roc(c23: pd.DataFrame) -> None:
    y, p = c23["target_7"].to_numpy().astype(float), c23["base_7"].to_numpy()
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    tp, fp = np.cumsum(ys)[last], np.cumsum(1 - ys)[last]
    x, yy = np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
    a_ = auc(y, p)
    note("31h", "候補数", len(c23))
    note("31h", "AUC(全候補)", a_)
    fig, ax = new(820, 640)
    ax.fill_between(x, yy, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, yy, color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=1.5)
    ax.text(0.30, 0.07, f"塗った面積\n= AUC {a_:.3f}", fontsize=12, color=INK)
    ax.text(0.60, 0.38, "点線=ランダム\n(AUC 0.5)", fontsize=12, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_xlabel("偽陽性率(FPR)")
    ax.set_ylabel("真陽性率(TPR)")
    ax.set_title(f"ROC曲線(2023年、{len(c23):,}件)", loc="left", fontsize=12)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    save(fig, "31h_roc.png")


# ---------- 31b 予測期間 ----------
def fig_horizon(c23: pd.DataFrame) -> None:
    days, aucs, ratios = [7, 14, 30], [], []
    for h in days:
        p, y = c23[f"base_{h}"].to_numpy(), c23[f"target_{h}"].to_numpy()
        aucs.append(within_auc(c23, p, y, SIM_A))
        hi = p >= np.quantile(p, 0.7)
        ratios.append(y[hi].mean() / y[~hi].mean())
        note("31b", f"{h}日 似た物件どうしのAUC", aucs[-1])
        note("31b", f"{h}日 上位30%は残りの何倍", ratios[-1])
    cols = [BLUE, LSTEEL, LSTEEL]
    fig, ax = new(1100, 474, 2)
    for a, vals, fmt, ylim, title in [(ax[0], aucs, "{:.3f}", (0.48, 0.60), "AUC(似た物件どうし): ほぼ同じ"),
                                      (ax[1], ratios, "{:.2f}倍", (1.0, 2.25), "上位30%は、残りの何倍終わったか")]:
        for i, v in enumerate(vals):
            a.bar(i, v, 0.6, color=cols[i])
            a.text(i, v + (0.003 if ylim[1] < 1 else 0.04), fmt.format(v), ha="center", fontsize=12)
        a.set_ylim(*ylim)
        a.set_xticks(range(3))
        a.set_xticklabels(["7日", "14日", "30日"], fontsize=12)
        a.set_title(title, loc="left", fontsize=12)
    ax[0].axhline(0.5, color="black", ls="--", lw=1)
    ax[0].set_ylabel("点線=ランダム(0.5)", fontsize=10)
    fig.tight_layout()
    save(fig, "31b_horizon.png")


# ---------- 31c 2グループ版と3グループ版 ----------
def level(p: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.searchsorted(np.array(cuts), p, side="right")


def best_cuts(p: np.ndarray, y: np.ndarray, k: int) -> list[float]:
    grid = np.unique(np.quantile(p, np.arange(0.10, 0.91, 0.02)))
    best, bc = -1.0, None
    for cuts in itertools.combinations(grid, k - 1):
        lv = level(p, list(cuts))
        if (np.bincount(lv, minlength=k) / len(p)).min() < 0.10:
            continue
        a_ = auc(y, lv)
        if a_ > best:
            best, bc = a_, list(cuts)
    return bc


def fig_levels(c22: pd.DataFrame, c23: pd.DataFrame, c24: pd.DataFrame) -> None:
    p22, y22 = c22["base_7"].to_numpy(), c22["target_7"].to_numpy()
    cuts = {"2グループ版": best_cuts(p22, y22, 2), "3グループ版": best_cuts(p22, y22, 3)}
    for k, v in cuts.items():
        note("31c", f"{k} 線(2022年で決めた)", 100 * v[0] if len(v) == 1 else 100 * v[0])
        if len(v) == 2:
            note("31c", f"{k} 2本目の線", 100 * v[1])
    # 確認用: 単純に等分した線(2等分・3等分)でも、3グループ版のほうが高いか
    eq = {"2グループ版(等分)": [float(np.quantile(p22, 0.5))], "3グループ版(等分)": np.quantile(p22, [1 / 3, 2 / 3]).tolist()}
    for d, yname in [(c23, "2023"), (c24, "2024")]:
        p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
        a2, a3 = auc(y, level(p, eq["2グループ版(等分)"])), auc(y, level(p, eq["3グループ版(等分)"]))
        note("31c", f"{yname} 等分 AUC 2グループ版", a2)
        note("31c", f"{yname} 等分 AUC 3グループ版", a3)
        note("31c", f"{yname} 等分 3グループ版−2グループ版", a3 - a2)
        for nm, c in cuts.items():
            lv = level(p, c)
            for i, g in enumerate(["低", "高"] if len(c) == 1 else ["低", "中", "高"]):
                note("31c", f"{yname} {nm} {g}の割合(%)", 100 * (lv == i).mean())
    COL = {"低": BLUE, "中": MID, "高": ACC}
    sets = [(c23, "2023年"), (c24, "2024年1〜3月(一度も使っていない)")]
    fig, ax = new(1130, 500, 2)
    for a, (d, yname) in zip(ax, sets):
        p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
        x0, ticks, labs = 0, [], []
        for nm, c in cuts.items():
            lv = level(p, c)
            names = ["低", "高"] if len(c) == 1 else ["低", "中", "高"]
            xs = x0 + np.arange(len(names))
            for xx, i in zip(xs, range(len(names))):
                r = 100 * y[lv == i].mean()
                note("31c", f"{yname[:4]} {nm} {names[i]} 終了率(%)", r)
                a.bar(xx, r, 0.78, color=COL[names[i]])
                a.text(xx, r + 0.6, f"{r:.0f}%", ha="center", fontsize=12)
                ticks.append(xx)
                labs.append(names[i])
            a.text(xs.mean(), -0.20, nm, ha="center", fontsize=12, color=INK, transform=a.get_xaxis_transform())
            x0 = xs[-1] + 1.6
        a.set_xticks(ticks)
        a.set_xticklabels(labs, fontsize=12)
        a.set_ylim(0, 32)
        a.set_title(yname, loc="left", fontsize=12)
    ax[0].set_ylabel("7日以内に終わった割合(%)", fontsize=12)
    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "31c1_levels_rates.png")

    rows = [("分けない(確率そのまま)", GRAY, None), ("3グループ版", MID, cuts["3グループ版"]), ("2グループ版", ACC, cuts["2グループ版"]), ("2グループ版(15%・発表で使う線)", LACC, [T])]
    fig, b = new(820, 470)
    w = 0.21
    for i, (d, yname) in enumerate(sets):
        p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
        for j, (nm, c, cut) in enumerate(rows):
            v = auc(y, p if cut is None else level(p, cut))
            note("31c", f"{yname[:4]} AUC {nm}", v)
            b.bar(i + (j - 1.5) * w, v, w * 0.94, color=c, label=nm if i == 0 else None)
            b.text(i + (j - 1.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    b.axhline(0.5, color="black", ls="--", lw=1)
    b.set_ylim(0.5, 0.70)
    b.set_xticks([0, 1])
    b.set_xticklabels(["2023年", "2024年1〜3月"], fontsize=12)
    b.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2, columnspacing=0.8, handlelength=1.2)
    b.set_title("AUC(全物件)", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31c2_levels_auc.png")


# ---------- 31g 2グループの線 ----------
def pair_V(d: pd.DataFrame, t: float) -> float:
    p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
    g = d[["snapshot", *SIM_D]].astype(str).agg("|".join, axis=1).to_numpy()
    f = pd.DataFrame({"g": g, "hi": p >= t, "y": y})
    f["yh"], f["yl"] = f["y"] * f["hi"], f["y"] * (~f["hi"])
    a = f.groupby("g").agg(n=("y", "size"), H=("hi", "sum"), yH=("yh", "sum"), yL=("yl", "sum"))
    a["L"] = a["n"] - a["H"]
    pairs = (a["n"] * (a["n"] - 1) / 2).sum()
    win = (a["yH"] * (a["L"] - a["yL"])).sum()
    lose = ((a["H"] - a["yH"]) * a["yL"]).sum()
    return float((win - lose) / pairs)


def fig_boundary(c22: pd.DataFrame, c23: pd.DataFrame) -> None:
    grid = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
    fig, a = new(820, 330)
    for d, color, nm in [(c22, BLUE, "2022年"), (c23, ACC, "2023年")]:
        v = [100 * pair_V(d, t) for t in grid]
        for t, x in zip(grid, v):
            note("31g", f"{nm} 線{100 * t:.0f}% 効き目(100組あたり)", x)
        a.plot([100 * t for t in grid], v, marker="o", color=color, lw=2.5, ms=6, label=nm)
    a.axvspan(10, 15, color="#E6E6E6", alpha=0.8, lw=0)
    a.axvline(15, color=INK, ls="--", lw=1.2)
    a.text(15.4, 0.15, "15%", color=INK, fontsize=12, va="bottom")
    a.set_xlim(5, 30)
    a.set_ylim(0, 3.6)
    a.set_xlabel("線(7日以内の確率が何%以上を上のグループにするか)", fontsize=11)
    a.set_ylabel("効き目\n(100組あたり)", fontsize=11)
    a.legend(frameon=False, loc="upper right", fontsize=11)
    a.grid(color="#e5e5e5")
    fig.tight_layout()
    save(fig, "31g_boundary.png")


# ---------- 31d 2023年の結果 ----------
def within_set(d: pd.DataFrame, keys: list[str], rng) -> tuple[float, float, float]:
    seg = d[["snapshot", *keys]].astype(str).agg("|".join, axis=1)
    diffs, w, kind = [], [], []
    for s, g in d.assign(seg=seg).groupby("seg"):
        if len(g) < 3:
            continue
        hi = g["base_7"].to_numpy() >= T
        if hi.all() or (~hi).all():
            continue
        y = g["target_7"].to_numpy()
        diffs.append(y[hi].mean() - y[~hi].mean())
        w.append(len(g))
        kind.append(s.split("|", 1)[1])
    diffs, w, kind = np.array(diffs), np.array(w, float), np.array(kind)
    uk = np.unique(kind)
    idx = {k: np.flatnonzero(kind == k) for k in uk}
    est = 100 * np.average(diffs, weights=w)
    bs = [100 * np.average(diffs[p], weights=w[p]) for p in
          (np.concatenate([idx[uk[i]] for i in rng.integers(0, len(uk), len(uk))]) for _ in range(500))]
    lo, hi_ = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi_


def fig_result2023(c23: pd.DataFrame) -> None:
    rng = np.random.default_rng(20261101)
    p, y = c23["base_7"].to_numpy(), c23["target_7"].to_numpy()
    hi = p >= T
    lo_r, all_r, hi_r = 100 * y[~hi].mean(), 100 * y.mean(), 100 * y[hi].mean()
    for k, v in [("低 終了率(%)", lo_r), ("全体 終了率(%)", all_r), ("高 終了率(%)", hi_r), ("高の割合(%)", 100 * hi.mean()),
                 ("倍率", hi_r / lo_r)]:
        note("31d", k, v)
    ws = {"間取り・築年数・家賃\nまで同じ物件": within_set(c23, SIM_A, rng), "間取りだけ\n同じ物件": within_set(c23, SIM_D, rng)}
    for k, (e, l_, h_) in ws.items():
        note("31d", f"{k.replace(chr(10), '')} 差(ポイント)", e)
        note("31d", f"{k.replace(chr(10), '')} 95%区間 下", l_)
        note("31d", f"{k.replace(chr(10), '')} 95%区間 上", h_)
    fig, ax = new(1180, 508, 2, [1, 1.25])
    a = ax[0]
    for i, (n, v, c) in enumerate([("低", lo_r, BLUE), ("全体", all_r, GRAY), ("高", hi_r, ACC)]):
        a.bar(i, v, 0.62, color=c)
        a.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=12)
    a.text(1.0, 21.2, f"高 / 低 = {hi_r / lo_r:.2f}倍", ha="center", fontsize=12, color=INK)
    a.set_xticks(range(3))
    a.set_xticklabels(["低", "全体", "高"], fontsize=12)
    a.set_ylim(0, 24)
    a.set_title(f"7日以内に終わった割合(%)  {len(c23):,}件", loc="left", fontsize=12)
    b = ax[1]
    for i, (n, (e, l_, h_)) in enumerate(ws.items()):
        b.barh(i, e, 0.5, color=STEEL, label="棒=高と低の差" if i == 0 else None)
        b.plot([l_, h_], [i, i], color=PINK, lw=1.6, marker="|", ms=10, mew=1.6, zorder=3, label="誤差線=95%信頼区間" if i == 0 else None)
        b.text(h_ + 0.35, i, f"+{e:.1f}", va="center", fontsize=12)
    b.set_yticks(range(2))
    b.set_yticklabels(list(ws), fontsize=12)
    b.set_xlim(0, 10.5)
    b.set_ylim(-0.6, 2.1)
    b.set_xlabel("高と低の差(ポイント)", fontsize=12)
    b.legend(frameon=False, loc="upper right", fontsize=11)
    b.set_title("似た物件どうしの差は小さい", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31d_result2023.png")


# ---------- 31e 偶然か、簡単な目安か(25と同じ計算、同じ乱数) ----------
def fig_baselines(c22: pd.DataFrame, c23: pd.DataFrame) -> None:
    rng = np.random.default_rng(20261020)
    edges = [0, 7, 14, 21, 30, 45, 60, 90, 120, 10_000]

    def rate_score(keys: list[str]) -> np.ndarray:
        tr, te = c22.copy(), c23.copy()
        tr["_e"] = pd.cut(tr["elapsed_day"], edges, right=False).astype(str)
        te["_e"] = pd.cut(te["elapsed_day"], edges, right=False).astype(str)
        g = tr.groupby(keys)["target_7"].mean().rename("s").reset_index()
        return te.merge(g, on=keys, how="left")["s"].fillna(tr["target_7"].mean()).to_numpy()

    def labels(score: np.ndarray, share: float) -> np.ndarray:
        order = np.lexsort((rng.random(len(score)), -score))
        h = np.zeros(len(score), bool)
        h[order[: int(round(share * len(score)))]] = True
        return h

    y = c23["target_7"].to_numpy()
    model_hi = c23["base_7"].to_numpy() >= T
    share = model_hi.mean()
    scores = {"今回の方法": None, "掲載日数だけ": rate_score(["_e"]), "家賃だけ": rate_score(["rent_group"]), "間取りだけ": rate_score(["layout_group"]),
              "築年数だけ": rate_score(["age_group"]), "3つの条件を組み合わせた集計": rate_score(["layout_group", "age_group", "rent_group"]),
              "ランダムに分けた場合": rng.random(len(c23))}
    his = {k: (model_hi if v is None else labels(v, share)) for k, v in scores.items()}
    mem = members(c23)
    draws = {k: [] for k in his}
    ratios = []
    for _ in range(1000):
        idx = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
        yy = y[idx]
        for k, h in his.items():
            hh = h[idx]
            draws[k].append(np.nan if (hh.all() or (~hh).all()) else 100 * (yy[hh].mean() - yy[~hh].mean()))
        hh = model_hi[idx]
        ratios.append(yy[hh].mean() / yy[~hh].mean())
    note("31e", "今回の方法 倍率(高/低) 95%区間 下", np.percentile(ratios, 2.5))
    note("31e", "今回の方法 倍率(高/低) 95%区間 上", np.percentile(ratios, 97.5))
    rows = []
    for k, h in his.items():
        gap = 100 * (y[h].mean() - y[~h].mean())
        lo, hi_ = np.nanpercentile(draws[k], [2.5, 97.5])
        rows.append((k, gap, lo, hi_))
        note("31e", f"{k} 差(ポイント)", gap)
        note("31e", f"{k} 95%区間 下", lo)
        note("31e", f"{k} 95%区間 上", hi_)
    rows = rows[::-1]
    fig, ax = new(1060, 540)
    for i, (k, gap, lo, hi_) in enumerate(rows):
        c = ACC if k == "今回の方法" else (GRAY if k.startswith("ランダム") else STEEL)
        ax.barh(i, gap, 0.62, color=c)
        ax.plot([lo, hi_], [i, i], color=PINK, lw=1.6, marker="|", ms=10, mew=1.6, zorder=3)
        ax.text(max(hi_, gap) + 0.25, i, f"{gap:+.1f}", va="center", fontsize=12)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.5, 10)
    ax.set_xlabel("高と低の差(ポイント)= 高の終了率 − 低の終了率", fontsize=12)
    ax.set_title("棒=高と低の差、誤差線=95%信頼区間", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31e_baselines.png")


# ---------- 31f 2024年1〜3月 ----------
def fig_holdout(c23: pd.DataFrame, c24: pd.DataFrame) -> None:
    fig, a = new(900, 500)
    for i, (d, nm) in enumerate([(c23, "2023年"), (c24, "2024年1〜3月")]):
        p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
        hi = p >= T
        lo_r, hi_r = 100 * y[~hi].mean(), 100 * y[hi].mean()
        for k, v in [("件数", len(d)), ("低 終了率(%)", lo_r), ("高 終了率(%)", hi_r), ("倍率", hi_r / lo_r), ("高の割合(%)", 100 * hi.mean())]:
            note("31f", f"{nm} {k}", v)
        a.bar(i - 0.19, lo_r, 0.34, color=BLUE, label="低" if i == 0 else None)
        a.bar(i + 0.19, hi_r, 0.34, color=ACC, label="高" if i == 0 else None)
        a.text(i - 0.19, lo_r + 0.5, f"{lo_r:.1f}%", ha="center", fontsize=12)
        a.text(i + 0.19, hi_r + 0.5, f"{hi_r:.1f}%", ha="center", fontsize=12)
        a.text(i, max(hi_r, lo_r) + 5.5, f"高 / 低 = {hi_r / lo_r:.2f}倍", ha="center", fontsize=12, color=INK)
    a.set_xticks([0, 1])
    a.set_xticklabels([f"2023年\n({len(c23):,}件)", f"2024年1〜3月\n({len(c24):,}件、一度も使っていない)"], fontsize=12)
    a.set_ylim(0, 40)
    a.legend(frameon=False, loc="upper left", fontsize=12)
    a.set_title("7日以内に終わった割合(%)", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31f_holdout.png")


# ---------- 国分寺市と多摩地域: 学習する範囲を変えても、国分寺市での当たり具合は同じか(15番の表を図にする) ----------
def fig_scope() -> None:
    """入力は summary_tables/15_scope_comparison.csv(15_scope_comparison.py が、範囲ごとに学び直して国分寺市の2023年で評価した結果)。
    ここでは再学習しない(scikit-learn が必要)ため、表の値をそのまま図にし、31_slide_numbers.csv に写す。"""
    t = pd.read_csv(TAB / "15_scope_comparison.csv")
    r = t[(t["予測期間(日)"] == 7) & t["候補の似かた"].str.startswith("A")].iloc[0]
    items = [("国分寺市だけ", "S1", STEEL), ("近隣10市\n+国分寺市", "S2", STEEL), ("多摩地域\n全体", "S3", STEEL),
             ("近隣10市\nだけ", "S2x", BLUE), ("多摩地域\nだけ", "S3x", BLUE)]
    fig, a = new(900, 440)
    for i, (nm, k, c) in enumerate(items):
        x = i + (0.4 if i >= 3 else 0)
        v = float(r[k])
        note("31n", f"{nm.replace(chr(10), '')} AUC(7日・似た物件どうし・国分寺市2023年)", v)
        a.bar(x, v, 0.62, color=c)
        a.text(x, v + 0.002, f"{v:.3f}", ha="center", fontsize=12)
    a.axhline(0.5, color="black", ls="--", lw=1)
    a.text(-0.4, 0.5025, "ランダム(0.5)", fontsize=10)
    a.set_xticks([i + (0.4 if i >= 3 else 0) for i in range(5)])
    a.set_xticklabels([n for n, _, _ in items], fontsize=10.5)
    a.set_ylim(0.48, 0.60)
    a.set_title("国分寺市でのAUC(学習した範囲ごと、7日)", loc="left", fontsize=12)
    a.text(3.7, 0.585, "国分寺市を学習から除いた\n(外部検証)", ha="center", fontsize=10.5, color=BLUE)
    fig.tight_layout()
    save(fig, "31n_scope.png")


# ---------- 確率の出し方の例(実データ: 2023年の候補で、条件ごとの7日以内の確率の中央値) ----------
EXAMPLES = [("例A", "1LDK", "0-10", "4", "1LDK・築10年未満・家賃8〜10万円"), ("例B", "1K", "30-40", "2", "1K・築30〜40年・家賃4〜6万円")]


def app_numbers() -> None:
    """アプリ案の画面の数字: 1K・築10〜20年・家賃6〜8万円の過去の掲載(国分寺市)。24と同じ手順(Kaplan–Meier法)。"""
    d = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    s = d[d["start_year"].between(2020, 2023) & d["valid_complete_duration"]].copy()
    s = s[(s["layout"] == "1K") & (s["age"] >= 10) & (s["age"] < 20) & ((s["rent"] // 20000 * 2) == 6)]
    dur, ev = s["duration"].to_numpy(float), (s["end_date"] < END).to_numpy()
    times = np.unique(dur[ev])
    surv = np.cumprod(1 - np.array([((dur == t) & ev).sum() for t in times]) / np.array([(dur >= t).sum() for t in times]))

    def S(t: float) -> float:
        k = np.searchsorted(times, t, side="right") - 1
        return 1.0 if k < 0 else float(surv[k])

    note("アプリ", "1K・築10〜20年・家賃6〜8万円 件数", len(s))
    for q, nm in [(0.75, "25%が終わる日"), (0.5, "半分が終わる日"), (0.25, "75%が終わる日")]:
        note("アプリ", nm, float(times[np.argmax(surv <= q)]))
    note("アプリ", "12日載った物件が、あと7日以内に終わる割合(%)", 100 * (1 - S(19) / S(12)))


def examples(c23: pd.DataFrame) -> None:
    for name, lay, age, rent, label in EXAMPLES:
        g = c23[(c23["layout_group"] == lay) & (c23["age_group"].astype(str) == age) & (c23["rent_group"].astype(str) == rent)]
        p7 = float(g["base_7"].median())
        daily = 1 - (1 - p7) ** (1 / 7)
        note("例", f"{name} {label} 件数", len(g))
        note("例", f"{name} 7日以内の確率(中央値,%)", 100 * p7)
        note("例", f"{name} 1日あたり平均(%)", 100 * daily)
        for k in range(1, 8):
            note("例", f"{name} {k}日目まで終わらない(%)", 100 * (1 - daily) ** k)


# ---------- 補足: 属性ごとの掲載日数 ----------
def fig_attrs() -> None:
    from scipy.stats import spearmanr
    d = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    s = d[d["start_year"].between(2020, 2023) & d["valid_complete_duration"]].copy()
    fig, ax = new(1664, 430, 4, [1.3, 1, 1, 1.2])
    age = s.dropna(subset=["age"])
    ab = pd.cut(age["age"], list(range(0, 45, 5)) + [200], right=False)
    m = age.groupby(ab, observed=True)["duration"].median()
    labs = [f"{int(i.left)}〜" for i in m.index]
    for k, v in zip(labs, m.to_numpy()):
        note("31i", f"築年数 {k}年 中央値(日)", v)
    ax[0].bar(range(len(m)), m.to_numpy(), color=STEEL)
    ax[0].set_xticks(range(len(m)))
    ax[0].set_xticklabels(labs, fontsize=9, rotation=45)
    ax[0].set_xlabel("築年数(年)", fontsize=11)
    ax[0].set_title("築年数", loc="left", fontsize=12)
    ax[0].set_ylabel("掲載日数の中央値(日)", fontsize=11)
    rb = pd.cut(s["rent"], [0, 50000, 70000, 90000, 110000, 1e9], labels=["〜5万", "5〜7万", "7〜9万", "9〜11万", "11万〜"])
    m = s.groupby(rb, observed=True)["duration"].median()
    for k, v in m.items():
        note("31i", f"家賃 {k}円 中央値(日)", v)
    ax[1].bar(range(len(m)), m.to_numpy(), color=STEEL)
    ax[1].set_xticks(range(len(m)))
    ax[1].set_xticklabels(m.index, fontsize=9, rotation=45)
    ax[1].set_title("家賃", loc="left", fontsize=12)
    top = s["layout"].value_counts().head(6).index
    m = s[s["layout"].isin(top)].groupby("layout")["duration"].median().reindex(top)
    for k, v in m.items():
        note("31i", f"間取り {k} 中央値(日)", v)
    ax[2].bar(range(len(m)), m.to_numpy(), color=STEEL)
    ax[2].set_xticks(range(len(m)))
    ax[2].set_xticklabels(m.index, fontsize=9, rotation=45)
    ax[2].set_title("間取り(多い順に6つ)", loc="left", fontsize=12)
    m = s.groupby("start_month")["duration"].median()
    for k, v in m.items():
        note("31i", f"開始月 {k}月 中央値(日)", v)
    ax[3].bar(m.index, m.to_numpy(), color=STEEL)
    ax[3].set_xticks(range(1, 13))
    ax[3].tick_params(axis="x", labelsize=9)
    ax[3].set_xlabel("掲載を始めた月", fontsize=11)
    ax[3].set_title("開始月", loc="left", fontsize=12)
    for a in ax:
        for p_ in a.patches:
            a.text(p_.get_x() + p_.get_width() / 2, p_.get_height() + 0.8, f"{p_.get_height():.0f}", ha="center", fontsize=8)
    for col, nm in [("age", "築年数"), ("rent", "家賃"), ("area", "面積"), ("walk_distance", "駅までの距離")]:
        ok = s[col].notna()
        note("31i", f"{nm} 順位相関", spearmanr(s.loc[ok, col], s.loc[ok, "duration"])[0])
    for col, nm in [("layout", "間取り"), ("station", "最寄駅"), ("start_month", "開始月")]:
        x = s[[col, "duration"]].dropna()
        g = x.groupby(col, observed=True)["duration"]
        eta = np.sqrt((g.size() * (g.mean() - x["duration"].mean()) ** 2).sum() / ((x["duration"] - x["duration"].mean()) ** 2).sum())
        note("31i", f"{nm} 相関比", eta)
    fig.tight_layout()
    save(fig, "31i_attrs.png")


# ---------- 補足: 線の決め方(効き目の誤差の幅と、高の割合) ----------
def fig_boundary_detail(c22: pd.DataFrame, c23: pd.DataFrame) -> None:
    rng = np.random.default_rng(20261102)
    grid = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
    fig, ax = new(1180, 508, 2)
    for d, color, nm in [(c22, BLUE, "2022年"), (c23, ACC, "2023年")]:
        p, y = d["base_7"].to_numpy(), d["target_7"].to_numpy()
        seg = d[["snapshot", *SIM_D]].astype(str).agg("|".join, axis=1).to_numpy()
        clus = np.array([x.split("|", 1)[1] for x in seg])
        est, lo, hi_, share, recall = [], [], [], [], []
        for t in grid:
            f = pd.DataFrame({"g": seg, "c": clus, "hi": p >= t, "y": y})
            f["yh"], f["yl"] = f["y"] * f["hi"], f["y"] * (~f["hi"])
            a = f.groupby("g").agg(c=("c", "first"), n=("y", "size"), H=("hi", "sum"), yH=("yh", "sum"), yL=("yl", "sum"))
            a["L"] = a["n"] - a["H"]
            pairs = (a["n"] * (a["n"] - 1) / 2).to_numpy()
            net = (a["yH"] * (a["L"] - a["yL"]) - (a["H"] - a["yH"]) * a["yL"]).to_numpy()
            est.append(100 * net.sum() / pairs.sum())
            cl = a["c"].to_numpy()
            uk = np.unique(cl)
            idx = {k: np.flatnonzero(cl == k) for k in uk}
            bs = []
            for _ in range(300):
                pick = np.concatenate([idx[uk[i]] for i in rng.integers(0, len(uk), len(uk))])
                bs.append(100 * net[pick].sum() / pairs[pick].sum())
            lo.append(np.percentile(bs, 2.5))
            hi_.append(np.percentile(bs, 97.5))
            share.append(100 * (p >= t).mean())
            recall.append(100 * y[p >= t].sum() / y.sum())
            note("31j", f"{nm} 線{100 * t:.0f}% 効き目", est[-1])
            note("31j", f"{nm} 線{100 * t:.0f}% 高の割合(%)", share[-1])
            note("31j", f"{nm} 線{100 * t:.0f}% 終わった物件のうち高(%)", recall[-1])
        xs = [100 * t for t in grid]
        ax[0].fill_between(xs, lo, hi_, color=color, alpha=0.15)
        ax[0].plot(xs, est, marker="o", color=color, lw=2.5, label=nm)
        if nm == "2023年":
            ax[1].plot(xs, share, marker="o", color=INK, lw=2.5, label="高の割合")
            ax[1].plot(xs, recall, marker="s", color=ACC, lw=2.5, label="終わった物件のうち、高に入った割合")
    ax[0].axvline(15, color=ACC, ls="--", lw=1.5)
    ax[0].set_xlabel("線(何%以上を高にするか)", fontsize=11)
    ax[0].set_ylabel("効き目(100組あたり)", fontsize=11)
    ax[0].set_title("効き目(帯=95%信頼区間)", loc="left", fontsize=12)
    ax[0].legend(frameon=False, fontsize=11)
    ax[1].axvline(15, color=ACC, ls="--", lw=1.5)
    ax[1].set_xlabel("線(何%以上を高にするか)", fontsize=11)
    ax[1].set_ylabel("%", fontsize=11)
    ax[1].set_title("2023年: 線を上げると、高は減る", loc="left", fontsize=12)
    ax[1].legend(frameon=False, fontsize=10, loc="lower left")
    for a in ax:
        a.grid(color="#e5e5e5")
    fig.tight_layout()
    save(fig, "31j_boundary_detail.png")


# ---------- 補足: 物件の似かたとAUC ----------
def fig_auc_similarity(c23: pd.DataFrame) -> None:
    p, y = c23["base_7"].to_numpy(), c23["target_7"].to_numpy()
    rows = [("同じ日に掲載中の物件すべて", []), ("同じ市・同じ間取り", SIM_D), ("同じ市・間取り・築年数・家賃", SIM_A)]
    vals = []
    for nm, keys in rows:
        vals.append(within_auc(c23, p, y, keys))
        note("31k", f"AUC {nm}", vals[-1])
    fig, a = new(1180, 551)
    for i, v in enumerate(vals):
        a.barh(i, v, 0.55, color=[GRAY, STEEL, BLUE][i])
        a.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=12)
    a.axvline(0.5, color="black", ls="--", lw=1)
    a.text(0.502, 2.45, "コイン投げ(0.5)", fontsize=11)
    a.set_yticks(range(3))
    a.set_yticklabels([r[0] for r in rows], fontsize=12)
    a.set_xlim(0.45, 0.68)
    a.invert_yaxis()
    a.set_xlabel("AUC(比べる物件の中だけで計算)", fontsize=12)
    a.set_title(f"比べる物件が似ているほど、AUCは0.5に近づく(2023年、{len(c23):,}件)", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31k_auc_similarity.png")


# ---------- 補足: 確率の確認(予測した確率と、実際に終わった割合) ----------
def fig_calibration(c23: pd.DataFrame, c24: pd.DataFrame) -> None:
    fig, a = new(1180, 551)
    mx = 0
    for d, h, color, nm in [(c23, 7, ACC, "2023年 7日以内"), (c23, 14, INK, "2023年 14日以内"), (c24, 7, BLUE, "2024年1〜3月 7日以内")]:
        p, y = d[f"base_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        f = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").mean()
        a.plot(100 * f["p"], 100 * f["y"], marker="o", color=color, lw=2.5, label=nm)
        mx = max(mx, 100 * f["p"].max(), 100 * f["y"].max())
        note("31l", f"{nm} 平均の予測(%)", 100 * p.mean())
        note("31l", f"{nm} 実際に終わった割合(%)", 100 * y.mean())
    a.plot([0, mx + 3], [0, mx + 3], color="#555555", ls="--", lw=1.2)
    a.set_xlim(0, mx + 3)
    a.set_ylim(0, mx + 3)
    a.set_xlabel("予測した確率(%)(10等分したグループの平均)", fontsize=12)
    a.set_ylabel("実際に終わった割合(%)", fontsize=12)
    a.set_title("点線に近いほど予測どおり。点線より下=予測が高く出た", loc="left", fontsize=12)
    a.legend(frameon=False, fontsize=11, loc="upper left")
    a.grid(color="#e5e5e5")
    fig.tight_layout()
    save(fig, "31l_calibration.png")


# ---------- 補足: 情報を足したときのAUC(似た物件どうし) ----------
def fig_extra(c23: pd.DataFrame) -> None:
    y = c23["target_7"].to_numpy()
    base = within_auc(c23, c23["base_7"].to_numpy(), y, SIM_A)
    note("31m", "今回の方法 AUC(似た物件どうし)", base)
    rows = [("部屋の履歴を足す", "hist_7"), ("建物の空室数を足す", "bldg_7"), ("近くの掲載数を足す", "supply_7"), ("3つとも足す", "all3_7"),
            ("計算方法をHGBに替える", "hgb_base_7"), ("3つ足して、HGB", "hgb_all3_7"), ("3つ+現況・入居可能時期", "all3x_7")]
    vals = []
    for nm, col in rows:
        vals.append(within_auc(c23, c23[col].to_numpy(), y, SIM_A) - base)
        note("31m", f"{nm} AUCの変化", vals[-1])
    fig, a = new(994, 480)
    for i, v in enumerate(vals):
        a.barh(i, v, 0.6, color=ACC if rows[i][1] == "all3x_7" else STEEL)
        a.text(v + (0.001 if v >= 0 else -0.001), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=12)
    a.axvline(0, color="black", lw=0.8)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in rows], fontsize=12)
    a.invert_yaxis()
    a.set_xlim(-0.035, 0.062)
    a.set_xlabel(f"AUCの変化(似た物件どうし。今回の方法は{base:.3f})", fontsize=11)
    a.set_title("情報を足しても、ほとんど上がらない", loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "31m_extra.png")


# ---------- 文字だけで出てくる件数(約21万件、契約日14件など) ----------
TSV = Path("/Users/aki/Downloads/bukken_rent_v1.0.1.tsv")


def counts() -> None:
    """確率の式を学んだ件数(17・26と同じ手順)と、国分寺市の件数・契約日の件数(21と同じ手順)。"""
    from src.data import prepare
    from room_lib.listing_episodes import build_episodes_fast
    raw = prepare(pd.read_parquet(ROOT / "data" / "prepared" / "tama_raw.parquet"))
    ep = build_episodes_fast(raw, max_gap_days=1)
    cutoff = pd.Timestamp("2023-01-01")
    tr = ep.loc[ep["start_year"].isin([2021, 2022]) & (ep["start_date"] < cutoff)].copy()
    event_end = tr["end_date"].lt(cutoff) & tr["end_date"].lt(END)
    days = (tr["end_date"].where(event_end, cutoff) - tr["start_date"]).dt.days
    note("件数", "確率の式を学んだ件数(多摩地域、2021〜2022年開始の部屋単位エピソード)", int((days >= 0).sum()))
    k = load_prepared(ROOT / "data" / "prepared" / "kokubunji_raw.parquet")
    note("件数", "国分寺市の掲載(2020〜2023年開始)", int(k["start_year"].between(2020, 2023).sum()))
    if TSV.exists():
        n = 0
        for ch in pd.read_csv(TSV, sep="\t", usecols=["addr1_2_name", "pub_start_date", "keiyaku_date"], dtype=str,
                              chunksize=500_000, keep_default_na=False):
            f = ch[(ch["addr1_2_name"] == "国分寺市") & ch["pub_start_date"].str[:4].isin(["2020", "2021", "2022", "2023"])]
            n += int((f["keiyaku_date"].str.strip() != "").sum())
        note("件数", "国分寺市で契約日(keiyaku_date)が入っている掲載", n)
    else:
        print(f"元のTSVがないため、契約日の件数は数えない: {TSV}")


def main() -> None:
    c22, c23, c24 = (pd.read_parquet(p).reset_index(drop=True) for p in (C22, C23, C24))
    counts()
    fig_scope()
    app_numbers()
    examples(c23)
    fig_attrs()
    fig_boundary_detail(c22, c23)
    fig_auc_similarity(c23)
    fig_calibration(c23, c24)
    fig_extra(c23)
    fig_duration()
    fig_roc(c23)
    fig_horizon(c23)
    fig_levels(c22, c23, c24)
    fig_boundary(c22, c23)
    fig_result2023(c23)
    fig_baselines(c22, c23)
    fig_holdout(c23, c24)
    t = pd.DataFrame(NUMS)
    t.to_csv(TAB / "31_slide_numbers.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.assign(値=t["値"].round(4)).to_string(index=False))


if __name__ == "__main__":
    main()

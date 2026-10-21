"""今回の方針(部屋単位・7日以内・高と低の2段階)の分析結果を、図にする(探索的)。

入力は、git管理されている summary_tables/ の表だけ(再学習はしない)。
出力: figures/23_*.png
  23a 高と低の比較 / 23b 確率の較正 / 23c 境界の選び方 / 23d 情報を足す検証
  23e 候補が似ているほど当たりが下がる / 23f 部屋単位で数える
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import figstyle  # noqa: E402
from figstyle import DARK, LIGHT, MIDC, PALE, REF, errbar_h, panel  # noqa: E402

T = HERE / "summary_tables"
FIG = HERE / "figures"
figstyle.apply()


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


def finish(fig, name: str, title: str, note: str) -> None:
    figstyle.save(fig, FIG, name, title, note)


def fig_a() -> None:
    lab = rd("19_labels_by_year.csv")
    lab = lab[(lab["予測期間(日)"] == 7) & lab["境界の決め方"].str.startswith("a")]
    within = rd("19_within_set_2023.csv")
    within = within[(within["予測期間(日)"] == 7) & within["境界の決め方"].str.startswith("a")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(2)
    for i, (col, name, color) in enumerate([("低の実終了率", "低", LIGHT), ("全体", "全体", PALE), ("高の実終了率", "高", DARK)]):
        v = [100 * lab[lab["年"].str.startswith(y)][col].iloc[0] for y in ("2022", "2023")]
        bars = ax[0].bar(x + (i - 1) * 0.26, v, 0.26, color=color, label=name)
        for b, val in zip(bars, v):
            ax[0].text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax[0].set_xticks(x, ["2022年", "2023年"])
    ax[0].set_ylabel("7日以内に終わった割合(%)")
    ax[0].legend()
    ax[0].set_ylim(0, 20)
    panel(ax[0], "(a)")
    names = ["間取りだけ同じ", "間取り・築年数・家賃まで同じ"]
    est = within["高−低の実終了率の差pt"].to_numpy()
    ax[1].barh(range(2), est, color=MIDC, height=0.5)
    for i, (lo, hi) in enumerate(zip(within["ci_low"], within["ci_high"])):
        errbar_h(ax[1], i, lo, hi)
    ax[1].set_yticks(range(2), names)
    ax[1].axvline(0, color="#333", lw=0.8)
    ax[1].set_xlim(-0.5, 10)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("高と低の差(ポイント)")
    panel(ax[1], "(b)")
    finish(fig, "23a_high_low_comparison.png", "図a 高と低の7日以内の終了率(境界15%)",
           "(a) 近隣10市+国分寺市の、2022年(境界を決めた年)と2023年(境界を動かさずに適用)。(b) 似た候補どうしでの、高と低の差(ポイント)。棒=差、誤差線=95%区間。2021〜2022年開始で学習したModel 2。探索的。出典: summary_tables/19_*.csv")


def fig_b() -> None:
    c = rd("18_calibration.csv")
    c = c[(c["評価対象"] == "近隣10市+国分寺市") & (c["モデル"] == "base")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, h, pl in zip(ax, (7, 14), ("(a) 7日以内", "(b) 14日以内")):
        d = c[c["予測期間(日)"] == h]
        lim = 100 * max(d["予測"].max(), d["実際"].max()) * 1.1
        a.plot([0, lim], [0, lim], color=REF, ls="--", lw=1.0, label="予測どおり")
        a.plot(100 * d["予測"], 100 * d["実際"], "o-", color=DARK, lw=1.6, ms=5, label="予測を10等分した各グループ")
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel("予測した終了確率(%)")
        a.set_ylabel("実際の終了率(%)")
        a.set_title(pl, loc="left", fontsize=12)
        a.legend()
    finish(fig, "23b_calibration.png", "図b 予測した確率と、実際の終了率(2023年)",
           "近隣10市+国分寺市の2023年の候補を、予測の大きさで10等分(各約3,370件)。点線より下は、予測が実際より高く出ている。2021〜2022年開始で学習。出典: summary_tables/18_calibration.csv")


def fig_c() -> None:
    thr = rd("18_threshold_table.csv")
    thr = thr[(thr["評価対象"] == "近隣10市+国分寺市") & (thr["モデル"] == "base") & (thr["予測期間(日)"] == 7)]
    pb = rd("20_pair_boundary.csv")
    pb = pb[pb["候補の似かた"].str.startswith("D")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for year, color, mk in [("2022", MIDC, "s"), ("2023", DARK, "o")]:
        d = pb[pb["年"].str.startswith(year)]
        ax[0].plot(100 * d["境界"], 100 * d["V(当たり−外れ)"], marker=mk, color=color, lw=1.6, ms=5.5, label=f"{year}年")
        ax[0].fill_between(100 * d["境界"], 100 * d["V_ci_low"], 100 * d["V_ci_high"], color=color, alpha=0.13, lw=0)
    ax[0].axvline(12, color=REF, ls=":", lw=1)
    ax[0].axvline(15, color=REF, ls="--", lw=1)
    ax[0].set_xlabel("境界(7日以内の予測確率が何%以上を高にするか)")
    ax[0].set_ylabel("効き目(100組あたり)")
    ax[0].legend()
    panel(ax[0], "(a)")
    ax[1].plot(100 * thr["境界T"], 100 * thr["高の割合"], "o-", color=MIDC, lw=1.6, ms=5, label="高になる物件の割合")
    ax[1].plot(100 * thr["境界T"], 100 * thr["高が捕まえる終了の割合"], "s-", color=DARK, lw=1.6, ms=5, label="終わった物件のうち高に入る割合")
    ax[1].set_xlabel("境界(%)")
    ax[1].set_ylabel("割合(%)")
    ax[1].legend()
    panel(ax[1], "(b)")
    finish(fig, "23c_boundary_choice.png", "図c 高と低の境界の選び方(7日以内)",
           "(a) 同じ間取りの2件を比べ、高の方だけ終わった組が、低の方だけ終わった組より、100組あたり何組多いか(帯は95%区間。点線は12%、破線は15%)。(b) 2023年の全候補で、境界を上げたときの割合。探索的。出典: summary_tables/20_*.csv、18_threshold_table.csv")


def fig_d() -> None:
    d = rd("17_auc_diff_vs_base.csv")
    d = d[(d["評価対象"] == "近隣10市+国分寺市") & (d["候補の似かた"] == "A") & (d["予測期間(日)"] == 7)]
    names = {"hist": "部屋の履歴", "bldg": "建物の空室数", "supply": "近くの掲載数", "all3": "上の3つ全部",
             "hgb_base": "計算方法をHGBに替える", "hgb_all3": "HGB+3つ全部", "all3x": "3つ+現況・入居可能時期", "hgb_all3x": "HGB+3つ+現況・入居可能時期"}
    d = d.set_index("方式").loc[list(names)]
    est = d["baseとの差"].to_numpy()
    colors = [LIGHT, LIGHT, LIGHT, LIGHT, MIDC, MIDC, DARK, DARK]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(range(len(est)), est, color=colors, height=0.55)
    for i, (lo, hi) in enumerate(zip(d["ci_low"], d["ci_high"])):
        errbar_h(ax, i, lo, hi)
    ax.set_yticks(range(len(est)), [names[k] for k in d.index])
    ax.axvline(0, color="#333", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("基準(Model 2)からのAUCの変化")
    finish(fig, "23d_extra_features.png", "図d 情報を足したときの、似た候補の中のAUCの変化(7日以内)",
           "近隣10市+国分寺市、最も厳しい候補集合(間取り・築年数・家賃が近い物件どうし)。基準のAUCは約0.55。誤差線=95%区間。薄い色: 変化なし、中: わずか、濃い色: 現況・入居可能時期を使う(掲載側の都合を表す可能性が高い)。出典: summary_tables/17_auc_diff_vs_base.csv")


def fig_e() -> None:
    a = rd("17_auc_by_model.csv")
    a = a[a["予測期間(日)"] == 7]
    order = ["E", "D", "C", "B", "A"]
    labels = ["同じ日に掲載中の\n全物件", "間取りだけ\n同じ", "間取り×家賃\n同じ", "間取り×築年数\n同じ", "間取り×築年数×家賃\n同じ"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(order))
    for i, (pop, color, mk) in enumerate([("近隣10市+国分寺市", DARK, "o"), ("国分寺市", MIDC, "s")]):
        v = [a[(a["評価対象"] == pop) & a["候補の似かた"].str.startswith(k)]["base"].iloc[0] for k in order]
        ax.plot(x + (i - 0.5) * 0.12, v, mk, color=color, ms=8, label=pop if pop != "国分寺市" else "国分寺市のみ")
    ax.axhline(0.5, color=REF, ls="--", lw=1.0)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.49, 0.68)
    ax.set_ylabel("AUC(候補の中)")
    ax.legend(loc="upper right")
    finish(fig, "23e_auc_by_similarity.png", "図e 比べる物件の集合の作り方ごとの、候補内AUC(2023年、7日以内)",
           "点線は0.5(ランダム)。比べる物件が似ているほど、AUCは0.5に近づく。2021〜2022年開始で学習。アプリの比較に近いのは右端。出典: summary_tables/17_auc_by_model.csv")


def fig_f() -> None:
    cnt = rd("21_counts.csv").set_index("指標")["値"]
    vals = [cnt["掲載(2020〜2023年開始)"], cnt["部屋単位エピソード(統合ルール1日)"], cnt["部屋(building_id×unit_id)"]]
    names = ["掲載", "部屋単位で統合した件数", "部屋の数"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bars = ax.barh(range(3), vals, color=MIDC, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 150, b.get_y() + b.get_height() / 2, f"{int(v):,}", va="center", fontsize=11)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 26000)
    ax.set_xlabel("件数(国分寺市、2020〜2023年に掲載開始)")
    finish(fig, "23f_room_level_counts.png", "図f 掲載・統合後の件数・部屋の数",
           "同じ部屋の重なる・1日以内に連続する掲載を統合した(間取りや面積が食い違うものは統合しない)。22,332件の掲載は、統合すると約3割減って15,276件、部屋は7,357。出典: summary_tables/21_counts.csv、08_episode_sensitivity.csv")


def fig_g() -> None:
    s = rd("21_duration_summary.csv")
    pts = {0: 0.0}
    for _, r in s.iterrows():
        if "日後までに終了" in r["指標"]:
            pts[int(r["指標"].split("日後")[0])] = float(r["値"])
    days = sorted(pts)
    y = [100 * pts[d] for d in days]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(days, y, color=DARK, lw=2, marker="o", ms=4)
    for d in (7, 30, 90):
        ax.plot([d, d], [0, 100 * pts[d]], color="#999999", lw=0.8, ls=":")
        ax.plot([0, d], [100 * pts[d]] * 2, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 7, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel("掲載を始めてからの日数")
    ax.set_ylabel("終わった物件の割合(%)")
    finish(fig, "23g_ended_share_by_day.png", "図g 掲載を始めてからの日数と、終わった物件の割合",
           "国分寺市、2020〜2023年に掲載開始の22,332件(Kaplan–Meier法)。点線は7日(約30%)、30日(約58%)、90日(約85%)。半分は22日までに終わる。「終わった」=掲載が終了。成約とは限らない。出典: summary_tables/21_duration_summary.csv")


def _medians(prefix: str) -> pd.DataFrame:
    g = rd("21_group_medians.csv")
    return g[g["区分"].str.startswith(prefix)].reset_index(drop=True)


def _bars(ax, labels, vals, ylabel=None) -> None:
    bars = ax.bar(labels, vals, color=MIDC, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    if ylabel:
        ax.set_ylabel(ylabel)


def fig_h() -> None:
    d = _medians("家賃 ")
    labels = ["5万円\n以下", "5〜7\n万円", "7〜9\n万円", "9〜11\n万円", "11万円\n超"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _bars(ax, labels, d["掲載日数の中央値"].to_numpy(), "掲載日数の中央値(日)")
    finish(fig, "23h_rent_median.png", "図h 家賃の帯ごとの、掲載日数の中央値",
           "国分寺市、2020〜2023年開始の22,332件。家賃が安いほど、長く載る。出典: summary_tables/21_group_medians.csv")


def fig_i() -> None:
    lay = _medians("間取り ")
    lay["name"] = lay["区分"].str.replace("間取り ", "", regex=False)
    lay = lay.set_index("name").loc[["1K", "1R", "1LDK", "2LDK"]]
    mon = _medians("掲載開始月 ")
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.4))
    _bars(ax[0], list(lay.index), lay["掲載日数の中央値"].to_numpy(), "中央値(日)")
    _bars(ax[1], [f"{i}月" for i in range(1, 13)], mon["掲載日数の中央値"].to_numpy(), "中央値(日)")
    panel(ax[0], "(a) 間取り")
    panel(ax[1], "(b) 掲載を始めた月")
    finish(fig, "23i_layout_month_median.png", "図i 間取りと掲載開始月ごとの、掲載日数の中央値",
           "国分寺市、2020〜2023年開始の22,332件。1Rが長く、1LDKは短め。1月は短く、夏は長め。出典: summary_tables/21_group_medians.csv")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_a, fig_b, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h, fig_i):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()

"""今回の方針(部屋単位・7日以内・高と低の2段階)の分析結果を、図にする(探索的)。

入力は、git管理されている summary_tables/ の表だけ(再学習はしない)。
出力: figures/23_*.png
  23a 高と低の比較 / 23b 確率の較正 / 23c 境界の選び方 / 23d 情報を足す検証
  23e 候補が似ているほど当たりが下がる / 23f 部屋単位で数える
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
T = HERE / "summary_tables"
FIG = HERE / "figures"
INK, ACC, BLUE, GRAY = "#4A1D3A", "#C2185B", "#3949AB", "#A58A9A"
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


def finish(fig, name: str, title: str, note: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_a() -> None:
    lab = rd("19_labels_by_year.csv")
    lab = lab[(lab["予測期間(日)"] == 7) & lab["境界の決め方"].str.startswith("a")]
    within = rd("19_within_set_2023.csv")
    within = within[(within["予測期間(日)"] == 7) & within["境界の決め方"].str.startswith("a")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(2)
    for i, (col, name, color) in enumerate([("低の実終了率", "低", BLUE), ("全体", "全体", GRAY), ("高の実終了率", "高", ACC)]):
        v = [100 * lab[lab["年"].str.startswith(y)][col].iloc[0] for y in ("2022", "2023")]
        bars = ax[0].bar(x + (i - 1) * 0.26, v, 0.26, color=color, label=name)
        for b, val in zip(bars, v):
            ax[0].text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax[0].set_xticks(x, ["2022年\n(境界を決めた年)", "2023年\n(境界を固定して使用)"])
    ax[0].set_ylabel("7日以内に掲載が終わった割合(%)")
    ax[0].set_title("全体: 高は低の約2倍、終わりやすい", loc="left", fontsize=13, fontweight="bold")
    ax[0].legend(frameon=False)
    ax[0].set_ylim(0, 20)
    names = ["間取りだけ同じ", "間取り・築年数・家賃まで同じ"]
    est = within["高−低の実終了率の差pt"].to_numpy()
    err = np.vstack([est - within["ci_low"].to_numpy(), within["ci_high"].to_numpy() - est])
    ax[1].barh(names, est, xerr=err, color=INK, height=0.5, capsize=6)
    for i, (v, hi) in enumerate(zip(est, within["ci_high"].to_numpy())):
        ax[1].text(hi + 0.25, i, f"+{v:.1f}pt", fontsize=12, fontweight="bold", ha="left", va="center")
    ax[1].axvline(0, color="#333", lw=1)
    ax[1].set_xlim(-0.5, 10)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("同じ候補の中での、高−低の実終了率の差(ポイント、95%区間)")
    ax[1].set_title("似た候補どうし: 差は小さいが、ゼロではない", loc="left", fontsize=13, fontweight="bold")
    finish(fig, "23a_high_low_comparison.png", "図a 高の物件は、低の物件より実際に終わりやすい(7日以内、境界15%)",
           "近隣10市+国分寺市。2021〜2022年開始で学習したModel 2。境界15%は2022年のデータで確かめ、2023年にそのまま適用。探索的。出典: summary_tables/19_*.csv")


def fig_b() -> None:
    c = rd("18_calibration.csv")
    c = c[(c["評価対象"] == "近隣10市+国分寺市") & (c["モデル"] == "base")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for a, h in zip(ax, (7, 14)):
        d = c[c["予測期間(日)"] == h]
        lim = 100 * max(d["予測"].max(), d["実際"].max()) * 1.1
        a.plot([0, lim], [0, lim], color=GRAY, ls="--", label="予測どおり")
        a.plot(100 * d["予測"], 100 * d["実際"], "o-", color=ACC, label="予測を10等分した各グループ")
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel(f"予測した{h}日以内の終了確率(%)")
        a.set_ylabel("実際の終了率(%)")
        a.set_title(f"{h}日以内: 予測が高いほど、予測が実際を上回る", loc="left", fontsize=12, fontweight="bold")
        a.legend(frameon=False)
        a.grid(color="#e5e5e5")
    finish(fig, "23b_calibration.png", "図b 確率は高く出すぎる。%は見せず、高/低のラベルで示す",
           "近隣10市+国分寺市の2023年の候補を、予測の大きさで10等分(各約3,370件)。2021〜2022年開始で学習。出典: summary_tables/18_calibration.csv")


def fig_c() -> None:
    thr = rd("18_threshold_table.csv")
    thr = thr[(thr["評価対象"] == "近隣10市+国分寺市") & (thr["モデル"] == "base") & (thr["予測期間(日)"] == 7)]
    pb = rd("20_pair_boundary.csv")
    pb = pb[pb["候補の似かた"].str.startswith("D")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for year, color, mk in [("2022", BLUE, "o"), ("2023", ACC, "s")]:
        d = pb[pb["年"].str.startswith(year)]
        ax[0].plot(100 * d["境界"], 100 * d["V(当たり−外れ)"], marker=mk, color=color, label=f"{year}年")
        ax[0].fill_between(100 * d["境界"], 100 * d["V_ci_low"], 100 * d["V_ci_high"], color=color, alpha=0.15)
    ax[0].axvline(12, color=INK, ls=":", lw=1)
    ax[0].axvline(15, color=GRAY, ls=":", lw=1)
    ax[0].text(12, ax[0].get_ylim()[1] * 0.97, " 12%", color=INK, fontsize=10, va="top")
    ax[0].text(15, ax[0].get_ylim()[1] * 0.90, " 15%", color=GRAY, fontsize=10, va="top")
    ax[0].set_xlabel("境界(7日以内の予測確率が何%以上を「高」にするか)")
    ax[0].set_ylabel("ラベルの正味の効き目(100組あたり、当たり−外れ)")
    ax[0].set_title("同じ間取りの2件を比べたときの効き目:\n10〜12%で最大、15%を超えると下がる", loc="left", fontsize=12, fontweight="bold")
    ax[0].legend(frameon=False)
    ax[0].grid(color="#e5e5e5")
    ax[1].plot(100 * thr["境界T"], 100 * thr["高の割合"], "o-", color=INK, label="高になる物件の割合(%)")
    ax[1].plot(100 * thr["境界T"], 100 * thr["高が捕まえる終了の割合"], "s-", color=ACC, label="終わった物件のうち高に入る割合(%)")
    ax[1].set_xlabel("境界(%)")
    ax[1].set_ylabel("割合(%)")
    ax[1].set_title("境界を上げると、高は少なくなり、終わる物件を見逃す", loc="left", fontsize=12, fontweight="bold")
    ax[1].legend(frameon=False)
    ax[1].grid(color="#e5e5e5")
    finish(fig, "23c_boundary_choice.png", "図c 高・低の境界の選び方(7日以内)",
           "左: 同じ日・同じ市・同じ間取りの物件2件で、高の方だけ終わった組−低の方だけ終わった組(帯は95%区間)。右: 2023年の全候補(近隣10市+国分寺市)。探索的。出典: summary_tables/20_*.csv、18_threshold_table.csv")


def fig_d() -> None:
    d = rd("17_auc_diff_vs_base.csv")
    d = d[(d["評価対象"] == "近隣10市+国分寺市") & (d["候補の似かた"] == "A") & (d["予測期間(日)"] == 7)]
    names = {"hist": "部屋の履歴", "bldg": "建物の空室数", "supply": "近くの掲載数", "all3": "上の3つ全部",
             "hgb_base": "計算方法をHGBに替える", "hgb_all3": "HGB+3つ全部", "all3x": "3つ+現況・入居可能時期", "hgb_all3x": "HGB+3つ+現況・入居可能時期"}
    d = d.set_index("方式").loc[list(names)]
    est = d["baseとの差"].to_numpy()
    err = np.vstack([est - d["ci_low"].to_numpy(), d["ci_high"].to_numpy() - est])
    colors = [GRAY, GRAY, GRAY, GRAY, INK, INK, ACC, ACC]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.barh([names[k] for k in d.index], est, xerr=err, color=colors, height=0.55, capsize=5)
    ax.axvline(0, color="#333", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("基準(Model 2)からのAUCの変化(95%区間)")
    ax.text(0.99, 0.97, "灰: 変化なし  黒: わずか  橙: 現況・入居可能時期を使う\n(掲載の都合を表す可能性が高い)", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    finish(fig, "23d_extra_features.png", "図d 情報を足しても、似た候補の中の当たりはほとんど上がらない(7日以内)",
           "近隣10市+国分寺市、最も厳しい候補集合(間取り・築年数・家賃が近い物件どうし)。基準のAUCは約0.55。出典: summary_tables/17_auc_diff_vs_base.csv")


def fig_e() -> None:
    a = rd("17_auc_by_model.csv")
    a = a[a["予測期間(日)"] == 7]
    order = ["E", "D", "C", "B", "A"]
    labels = ["同じ日に掲載中の\n全物件", "間取りだけ\n同じ", "間取り×家賃\n同じ", "間取り×築年数\n同じ", "間取り×築年数×家賃\n同じ(最も厳しい)"]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(order))
    for i, (pop, color) in enumerate([("近隣10市+国分寺市", INK), ("国分寺市", ACC)]):
        v = [a[(a["評価対象"] == pop) & a["候補の似かた"].str.startswith(k)]["base"].iloc[0] for k in order]
        bars = ax.bar(x + (i - 0.5) * 0.36, v, 0.36, color=color, label=pop if pop != "国分寺市" else "国分寺市のみ")
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color=GRAY, ls="--")
    ax.text(len(order) - 0.5, 0.489, "点線 0.5 = サイコロ(当てずっぽう)", color="#555555", ha="right", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.48, 0.70)
    ax.set_ylabel("候補内AUC(Model 2、7日以内)")
    ax.legend(frameon=False, loc="upper right")
    finish(fig, "23e_auc_by_similarity.png", "図e 比べる物件が似ているほど、当たりは0.5に近づく",
           "比べる物件の集合の作り方を変えたときの候補内AUC。2023年。2021〜2022年開始で学習。アプリの比較に近いのは右端。出典: summary_tables/17_auc_by_model.csv")


def fig_f() -> None:
    cnt = rd("21_counts.csv").set_index("指標")["値"]
    vals = [cnt["掲載(2020〜2023年開始)"], cnt["部屋単位エピソード(統合ルール1日)"], cnt["部屋(building_id×unit_id)"]]
    names = ["掲載(件)", "部屋単位で統合した件数\n(重なる・連続する掲載を1つに)", "部屋の数\n(building_id×unit_id)"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars = ax.barh(names, vals, color=[GRAY, INK, ACC], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 150, b.get_y() + b.get_height() / 2, f"{int(v):,}", va="center", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 26000)
    ax.set_xlabel("国分寺市、2020〜2023年に掲載開始")
    finish(fig, "23f_room_level_counts.png", "図f 22,332件の掲載は、統合すると約3割減り、7,357部屋になる",
           "同じ部屋の重なる・1日以内に連続する掲載を統合(間取りや面積が食い違うものは統合しない)。出典: summary_tables/21_counts.csv、08_episode_sensitivity.csv")


def fig_g() -> None:
    s = rd("21_duration_summary.csv")
    pts = {0: 0.0}
    for _, r in s.iterrows():
        if "日後までに終了" in r["指標"]:
            pts[int(r["指標"].split("日後")[0])] = float(r["値"])
    days = sorted(pts)
    y = [100 * pts[d] for d in days]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.fill_between(days, y, color=INK, alpha=0.08)
    ax.plot(days, y, color=INK, lw=3)
    for d in (7, 30, 90):
        ax.plot(d, 100 * pts[d], "o", color=ACC, ms=11)
        ax.annotate(f"{d}日目: {100 * pts[d]:.0f}%が終了", (d, 100 * pts[d]), xytext=(d + 6, 100 * pts[d] - (15 if d == 30 else 9)),
                    fontsize=14, fontweight="bold", color=ACC)
    ax.axhline(50, color=GRAY, ls="--", lw=1)
    ax.text(178, 51.5, "半分(22日目で到達)", ha="right", color="#555555", fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 100)
    ax.set_xlabel("掲載を始めてから何日目か")
    ax.set_ylabel("掲載が終わった物件の割合(%)")
    ax.grid(color="#e5e5e5")
    finish(fig, "23g_ended_share_by_day.png", "図g 掲載を始めた直後ほど、どんどん終わる",
           "国分寺市、2020〜2023年に掲載開始の22,332件(Kaplan–Meier法)。「終わった」=掲載が終了。成約とは限らない。出典: summary_tables/21_duration_summary.csv")


def _medians(prefix: str) -> pd.DataFrame:
    g = rd("21_group_medians.csv")
    return g[g["区分"].str.startswith(prefix)].reset_index(drop=True)


def _bars(ax, labels, vals, color, title, ylabel=None, fs=13) -> None:
    bars = ax.bar(labels, vals, color=color, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=fs + 1, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(labelsize=fs - 1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs - 1)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)


def fig_h() -> None:
    d = _medians("家賃 ")
    labels = ["5万円\n以下", "5〜7\n万円", "7〜9\n万円", "9〜11\n万円", "11万円\n超"]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    _bars(ax, labels, d["掲載日数の中央値"].to_numpy(), INK, "家賃が安いほど、長く載る", "掲載日数の中央値(日)")
    finish(fig, "23h_rent_median.png", "図h 家賃の帯ごとの、掲載日数の中央値",
           "国分寺市、2020〜2023年開始の22,332件。出典: summary_tables/21_group_medians.csv")


def fig_i() -> None:
    lay = _medians("間取り ")
    lay["name"] = lay["区分"].str.replace("間取り ", "", regex=False)
    lay = lay.set_index("name").loc[["1K", "1R", "1LDK", "2LDK"]]
    mon = _medians("掲載開始月 ")
    fig, ax = plt.subplots(2, 1, figsize=(8, 7.4))
    _bars(ax[0], list(lay.index), lay["掲載日数の中央値"].to_numpy(), INK, "間取り: 1Rが長く、1LDKは短め", "中央値(日)")
    _bars(ax[1], [f"{i}月" for i in range(1, 13)], mon["掲載日数の中央値"].to_numpy(), ACC, "掲載を始めた月: 1月は短く、夏は長め", "中央値(日)", fs=12)
    finish(fig, "23i_layout_month_median.png", "図i 間取りと掲載開始月ごとの、掲載日数の中央値",
           "国分寺市、2020〜2023年開始の22,332件。出典: summary_tables/21_group_medians.csv")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_a, fig_b, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h, fig_i):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()

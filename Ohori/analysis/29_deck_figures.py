"""発表スライド用の図を、大きな文字・専門用語なしで作り直す。
入力は git管理されている summary_tables/ の表だけ(再学習しない)。出力は figures/29*.png。
スライドの幅に1対1で貼る前提(dpi=150)なので、文字は小さくしすぎない。
  29a 掲載日数(帯ごとの割合 + 終わった割合の推移)
  29b 予測期間(7日・14日・30日)
  29c 3グループと2グループ(旧。1枚版)
  29c1 3グループと2グループ: 実際に終わった割合
  29c2 3グループと2グループ: 当たり具合(AUC)
  29d 結果(2023年)
  29e 偶然か、簡単な目安か
  29f 2024年1〜3月での確認
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
T, FIG = HERE / "summary_tables", HERE / "figures"
INK, ACC, BLUE, MID, GRAY, PINK, LACC = "#4A1D3A", "#C2185B", "#3949AB", "#A58A9A", "#B9A5B0", "#FF9EC4", "#E8A0BF"
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.edgecolor"] = "#555555"
DPI = 150


def rd(n: str) -> pd.DataFrame:
    return pd.read_csv(T / n)


def new(w_px: int, h_px: int, ncols: int = 1, ratios=None):
    fig, ax = plt.subplots(1, ncols, figsize=(w_px / DPI, h_px / DPI), gridspec_kw={"width_ratios": ratios} if ratios else None)
    return fig, ax


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


def a_duration() -> None:
    s = rd("21_duration_summary.csv").set_index("指標")["値"]
    bands = [("0〜6", "0〜6日"), ("7〜13", "7〜13日"), ("14〜29", "14〜29日"), ("30〜59", "30〜59日"), ("60〜89", "60〜89日"), ("90〜179", "90〜179日"), ("180〜", "180日以上")]
    vals = [100 * float(s[f"掲載件数の割合 {k}"]) for _, k in bands]
    fig, ax = new(1664, 520, 2, [1, 1])
    a = ax[0]
    for i, v in enumerate(vals):
        a.bar(i, v, 0.72, color=ACC if i == 0 else INK)
        a.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=14, fontweight="bold")
    a.set_xticks(range(len(bands)))
    a.set_xticklabels([b for b, _ in bands], fontsize=10.5)
    a.set_xlabel("掲載日数(日)", fontsize=13)
    a.set_ylim(0, 36)
    a.tick_params(axis="y", labelsize=12)
    a.set_ylabel("物件の割合(%)", fontsize=13)
    a.set_title("掲載日数ごとの物件の割合", loc="left", fontsize=15, fontweight="bold")
    b = ax[1]
    pts = {0: 0.0}
    for k, v in s.items():
        if "日後までに終了" in k:
            pts[int(k.split("日後")[0])] = float(v)
    days = sorted(pts)
    y = [100 * pts[d] for d in days]
    b.fill_between(days, y, color=INK, alpha=0.08)
    b.plot(days, y, color=INK, lw=3)
    for d, dx, dy, lab in [(7, 6, -9, "7日目 約3割"), (30, 6, -12, "30日目 約6割"), (90, 6, -10, "90日目 約85%")]:
        b.plot(d, 100 * pts[d], "o", color=ACC, ms=11)
        b.annotate(lab, (d, 100 * pts[d]), xytext=(d + dx, 100 * pts[d] + dy), fontsize=14, fontweight="bold", color=ACC)
    b.axhline(50, color=GRAY, ls="--", lw=1)
    b.text(178, 52, "半分は22日までに終了", ha="right", color="#444444", fontsize=12)
    b.set_xlim(0, 182)
    b.set_xticks([0, 30, 60, 90, 120, 150, 180])
    b.set_ylim(0, 100)
    b.set_xlabel("掲載を始めてから何日目か", fontsize=13)
    b.set_ylabel("終わった物件の割合(%)", fontsize=13)
    b.tick_params(labelsize=12)
    b.set_title("日がたつほど、終わった割合は増える", loc="left", fontsize=15, fontweight="bold")
    b.grid(color="#e5e5e5")
    fig.tight_layout()
    save(fig, "29a_duration.png")


def b_horizon() -> None:
    h = rd("22_horizon_choice.csv")
    h = h[h["評価対象"] == "近隣10市+国分寺市"].set_index("予測期間(日)")
    days = [7, 14, 30]
    labs = ["7日", "14日", "30日"]
    cols = [ACC, INK, INK]
    fig, ax = new(1100, 474, 2)
    a = ax[0]
    v = [h.loc[d, "似た候補のAUC"] for d in days]
    for i, x in enumerate(v):
        a.bar(i, x, 0.6, color=cols[i])
        a.text(i, x + 0.003, f"{x:.3f}", ha="center", fontsize=13, fontweight="bold")
    a.axhline(0.5, color="black", ls="--", lw=1)
    a.set_ylabel("点線=当てずっぽう(0.5)", fontsize=10)
    a.set_ylim(0.48, 0.60)
    a.set_xticks(range(3))
    a.set_xticklabels(labs, fontsize=13)
    a.tick_params(axis="y", labelsize=11)
    a.set_title("当たり具合(AUC): ほぼ同じ", loc="left", fontsize=13, fontweight="bold")
    b = ax[1]
    v = [h.loc[d, "高は低の何倍"] for d in days]
    for i, x in enumerate(v):
        b.bar(i, x, 0.6, color=cols[i])
        b.text(i, x + 0.04, f"{x:.2f}倍", ha="center", fontsize=13, fontweight="bold")
    b.set_ylim(1.0, 2.25)
    b.set_xticks(range(3))
    b.set_xticklabels(labs, fontsize=13)
    b.tick_params(axis="y", labelsize=11)
    b.set_title("高は低の何倍終わりやすいか", loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "29b_horizon.png")


def c_levels() -> None:
    g = rd("28_fair_group_rates.csv")
    m, m27 = rd("28_fair_auc.csv"), rd("27_three_levels_auc.csv")
    Y23, Y24 = "2023(固定して適用)", "2024年1〜3月(未使用)"
    S2, S3 = "V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)"
    COL = {"低": BLUE, "中": MID, "高": ACC}
    fig, ax = new(1130, 470, 2, [0.8, 1.2])
    a = ax[0]
    x0, ticks, labs = 0, [], []
    for sch, nm in [(S2, "2グループ"), (S3, "3グループ")]:
        d = g[(g["対象"] == Y23) & (g["分け方"] == sch)]
        xs = x0 + np.arange(len(d))
        for xx, (_, r) in zip(xs, d.iterrows()):
            a.bar(xx, 100 * r["実際の終了率"], 0.78, color=COL[r["グループ"]])
            a.text(xx, 100 * r["実際の終了率"] + 0.5, f"{100 * r['実際の終了率']:.0f}%", ha="center", fontsize=13, fontweight="bold")
            ticks.append(xx)
            labs.append(r["グループ"])
        a.text(xs.mean(), -4.6, nm, ha="center", fontsize=13, fontweight="bold", color=INK, clip_on=False)
        x0 = xs[-1] + 1.6
    a.set_xticks(ticks)
    a.set_xticklabels(labs, fontsize=13)
    a.set_ylim(0, 19)
    a.tick_params(axis="y", labelsize=11)
    a.set_title("2023年: 終わった割合(%)", loc="left", fontsize=12, fontweight="bold")
    b = ax[1]
    rows = [("確率そのまま", GRAY, "分けない", m), (S3, MID, "3グループ", m), (S2, ACC, "2グループ", m), ("2段階(15%)", LACC, "2グループ(15%・最終)", m27)]
    w = 0.21
    for i, y in enumerate([Y23, Y24]):
        for j, (k, c, nm, tab) in enumerate(rows):
            v = tab[(tab["対象"] == y) & (tab["分け方"] == k)]["全体のAUC"].iloc[0]
            b.bar(i + (j - 1.5) * w, v, w * 0.94, color=c, label=nm if i == 0 else None)
            b.text(i + (j - 1.5) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    b.axhline(0.5, color="black", ls="--", lw=1)
    b.set_ylim(0.5, 0.70)
    b.set_xticks([0, 1])
    b.set_xticklabels(["2023年", "2024年1〜3月"], fontsize=12)
    b.tick_params(axis="y", labelsize=11)
    b.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2, columnspacing=0.8, handlelength=1.2)
    b.set_title("当たり具合(AUC)", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=3.0)
    save(fig, "29c_levels.png")


def d_result() -> None:
    l = rd("19_labels_by_year.csv")
    l = l[(l["予測期間(日)"] == 7) & l["境界の決め方"].str.startswith("a") & l["年"].str.startswith("2023")].iloc[0]
    w = rd("19_within_set_2023.csv")
    w = w[(w["予測期間(日)"] == 7) & w["境界の決め方"].str.startswith("a")].set_index("候補の似かた")
    fig, ax = new(1180, 508, 2, [1, 1.25])
    a = ax[0]
    items = [("低", 100 * l["低の実終了率"], BLUE), ("全体", 100 * l["全体"], GRAY), ("高", 100 * l["高の実終了率"], ACC)]
    for i, (n, v, c) in enumerate(items):
        a.bar(i, v, 0.62, color=c)
        a.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=16, fontweight="bold")
    a.text(1.0, 21.2, "高は低の約2倍", ha="center", fontsize=17, fontweight="bold", color=ACC)
    a.set_xticks(range(3))
    a.set_xticklabels([n for n, _, _ in items], fontsize=15)
    a.set_ylim(0, 24)
    a.tick_params(axis="y", labelsize=12)
    a.set_title("7日以内に終わった割合(%)", loc="left", fontsize=14, fontweight="bold")
    b = ax[1]
    rows = [("間取り・築年数・家賃\nまで同じ物件", w.loc["A 間取り×築年数×家賃"]), ("間取りだけ\n同じ物件", w.loc["D 間取りだけ同じ"])]
    for i, (n, r) in enumerate(rows):
        v = r["高−低の実終了率の差pt"]
        b.barh(i, v, 0.5, color=INK)
        b.plot([r["ci_low"], r["ci_high"]], [i, i], color=PINK, lw=3.5, zorder=3)
        b.text(r["ci_high"] + 0.35, i, f"+{v:.1f}", va="center", fontsize=17, fontweight="bold")
    b.set_yticks(range(2))
    b.set_yticklabels([n for n, _ in rows], fontsize=12)
    b.set_xlim(0, 10.5)
    b.tick_params(axis="x", labelsize=12)
    b.set_xlabel("高と低の差(ポイント)\nピンクの線は誤差の範囲", fontsize=11)
    b.set_title("似た物件どうしの差は小さい", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "29d_result2023.png")


def e_baselines() -> None:
    t = rd("25_overall_ci_baselines.csv")
    names = {"モデル(今回)": "今回の方法", "掲載日数だけ": "掲載日数だけ", "家賃帯だけ": "家賃だけ", "間取りだけ": "間取りだけ", "築年数帯だけ": "築年数だけ",
             "間取り×築年数×家賃": "3つの条件を組み合わせた集計", "でたらめ(並べ替え)": "でたらめに分けた場合"}
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = new(1060, 501, 1)
    for i, r in t.iterrows():
        k = r["目安"]
        c = ACC if k.startswith("モデル") else (GRAY if k.startswith("でたらめ") else INK)
        ax.barh(i, r["差pt"], 0.62, color=c)
        ax.plot([r["差pt_ci_low"], r["差pt_ci_high"]], [i, i], color=PINK, lw=3.5, zorder=3)
        ax.text(max(r["差pt_ci_high"], r["差pt"]) + 0.25, i, f"{r['差pt']:+.1f}", va="center", fontsize=14, fontweight="bold" if k.startswith("モデル") else "normal")
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([names[k] for k in t["目安"]], fontsize=13)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.5, 10)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("高と低の差(ポイント。ピンクの線は誤差の範囲)", fontsize=12)
    fig.tight_layout()
    save(fig, "29e_baselines.png")


def f_holdout() -> None:
    l = rd("19_labels_by_year.csv")
    l = l[(l["予測期間(日)"] == 7) & l["境界の決め方"].str.startswith("a") & l["年"].str.startswith("2023")].iloc[0]
    h = rd("26_holdout_2024q1.csv")
    h = h[(h["対象"] == "近隣10市+国分寺市") & h["目安"].str.startswith("モデル")].iloc[0]
    w23 = rd("19_within_set_2023.csv")
    w23 = w23[(w23["予測期間(日)"] == 7) & w23["境界の決め方"].str.startswith("a")].set_index("候補の似かた")
    w24 = rd("26_holdout_2024q1_within_set.csv").set_index("候補の似かた")
    b25 = rd("25_overall_ci_baselines.csv")
    b25 = b25[b25["目安"].str.startswith("モデル")].iloc[0]
    fig, ax = new(1060, 459, 2, [1, 1.3])
    a = ax[0]
    for i, (lo, hi, nm) in enumerate([(100 * l["低の実終了率"], 100 * l["高の実終了率"], "2023年"), (100 * h["低の実終了率"], 100 * h["高の実終了率"], "2024年1〜3月")]):
        a.bar(i - 0.19, lo, 0.34, color=BLUE, label="低" if i == 0 else None)
        a.bar(i + 0.19, hi, 0.34, color=ACC, label="高" if i == 0 else None)
        a.text(i - 0.19, lo + 0.5, f"{lo:.0f}%", ha="center", fontsize=13, fontweight="bold")
        a.text(i + 0.19, hi + 0.5, f"{hi:.0f}%", ha="center", fontsize=13, fontweight="bold")
        a.text(i, max(hi, lo) + 6.2, f"{hi / lo:.2f}倍", ha="center", fontsize=15, fontweight="bold", color=INK)
    a.set_xticks([0, 1])
    a.set_xticklabels(["2023年", "2024年1〜3月\n(一度も使っていない)"], fontsize=12)
    a.set_ylim(0, 46)
    a.tick_params(axis="y", labelsize=11)
    a.legend(frameon=False, loc="upper left", ncol=1, fontsize=12)
    a.set_title("7日以内に終わった割合(%)", loc="left", fontsize=13, fontweight="bold")
    b = ax[1]
    rows = [("全体", (b25["差pt"], b25["差pt_ci_low"], b25["差pt_ci_high"]), (h["差pt"], h["差pt_ci_low"], h["差pt_ci_high"])),
            ("間取りだけ同じ", tuple(w23.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]])),
            ("いちばん似た物件", tuple(w23.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]))]
    for i, (nm, v23, v24) in enumerate(rows[::-1]):
        for j, (v, c, yr) in enumerate([(v23, GRAY, "2023年"), (v24, ACC, "2024年1〜3月")]):
            y = i + (0.19 if j == 0 else -0.19)
            b.barh(y, v[0], 0.32, color=c, label=yr if i == 0 else None)
            b.plot([v[1], v[2]], [y, y], color=PINK if j else INK, lw=3, zorder=3)
            b.text(max(v[2], v[0]) + 0.3, y, f"{v[0]:+.1f}", va="center", fontsize=12, fontweight="bold" if j else "normal")
    b.axvline(0, color="black", lw=0.8)
    b.set_yticks(range(3))
    b.set_yticklabels([r[0] for r in rows[::-1]], fontsize=12)
    b.set_xlim(-5, 15)
    b.tick_params(axis="x", labelsize=11)
    b.set_xlabel("高と低の差(ポイント)", fontsize=11)
    b.text(0.0, 1.0, "灰=2023年　赤=2024年1〜3月", transform=b.transAxes, fontsize=11, color="#444444", va="bottom")
    b.set_title("最も似た物件は差が0(2024年)", loc="left", fontsize=12, fontweight="bold", pad=20)
    b.set_ylim(-0.75, 2.7)
    fig.tight_layout(w_pad=2.5)
    save(fig, "29f_holdout.png")


def c1_levels_rates() -> None:
    """3グループと2グループで、実際に終わった割合(2023年と2024年1〜3月)。"""
    g = rd("28_fair_group_rates.csv")
    Y = [("2023(固定して適用)", "2023年"), ("2024年1〜3月(未使用)", "2024年1〜3月(一度も使っていない)")]
    S2, S3 = "V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)"
    COL = {"低": BLUE, "中": MID, "高": ACC}
    fig, ax = new(1130, 470, 2, [1, 1])
    for a, (yk, yt) in zip(ax, Y):
        x0, ticks, labs = 0, [], []
        for sch, nm in [(S2, "2グループ"), (S3, "3グループ")]:
            d = g[(g["対象"] == yk) & (g["分け方"] == sch)]
            xs = x0 + np.arange(len(d))
            for xx, (_, r) in zip(xs, d.iterrows()):
                a.bar(xx, 100 * r["実際の終了率"], 0.78, color=COL[r["グループ"]])
                a.text(xx, 100 * r["実際の終了率"] + 0.6, f"{100 * r['実際の終了率']:.0f}%", ha="center", fontsize=15, fontweight="bold")
                ticks.append(xx)
                labs.append(r["グループ"])
            a.text(xs.mean(), -8.2, nm, ha="center", fontsize=14, fontweight="bold", color=INK, clip_on=False)
            x0 = xs[-1] + 1.6
        a.set_xticks(ticks)
        a.set_xticklabels(labs, fontsize=14)
        a.set_ylim(0, 30)
        a.tick_params(axis="y", labelsize=12)
        a.set_title(yt, loc="left", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("7日以内に終わった割合(%)", fontsize=13)
    fig.tight_layout(w_pad=3.0)
    save(fig, "29c1_levels_rates.png")


def c2_levels_auc() -> None:
    """当たり具合(AUC)の比較。"""
    m, m27 = rd("28_fair_auc.csv"), rd("27_three_levels_auc.csv")
    Y23, Y24 = "2023(固定して適用)", "2024年1〜3月(未使用)"
    S2, S3 = "V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)"
    fig, b = plt.subplots(1, 1, figsize=(820 / DPI, 470 / DPI))
    rows = [("確率そのまま", GRAY, "分けない", m), (S3, MID, "3グループ", m), (S2, ACC, "2グループ(同じ探し方)", m), ("2段階(15%)", LACC, "2グループ(15%・最終)", m27)]
    w = 0.2
    for i, y in enumerate([Y23, Y24]):
        for j, (k, c, nm, tab) in enumerate(rows):
            v = tab[(tab["対象"] == y) & (tab["分け方"] == k)]["全体のAUC"].iloc[0]
            b.bar(i + (j - 1.5) * w, v, w * 0.92, color=c, label=nm if i == 0 else None)
            b.text(i + (j - 1.5) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    b.axhline(0.5, color="black", ls="--", lw=1)
    b.set_ylim(0.5, 0.72)
    b.set_xticks([0, 1])
    b.set_xticklabels(["2023年", "2024年1〜3月"], fontsize=14)
    b.tick_params(axis="y", labelsize=12)
    b.set_ylabel("当たり具合(AUC)", fontsize=13)
    b.legend(frameon=False, loc="upper right", fontsize=11, ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout()
    save(fig, "29c2_levels_auc.png")


if __name__ == "__main__":
    for f in (a_duration, b_horizon, c_levels, c1_levels_rates, c2_levels_auc, d_result, e_baselines, f_holdout):
        f()
    print("ok")

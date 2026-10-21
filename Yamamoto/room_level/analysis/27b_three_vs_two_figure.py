"""27の結果(3段階と2段階の比較)を図にする。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ST = HERE / "summary_tables"
INK, ACC, BLUE, MID, GRAY = "#4A1D3A", "#C2185B", "#3949AB", "#A58A9A", "#B9A5B0"
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

g = pd.read_csv(ST / "27_three_levels_group_rates.csv")
m = pd.read_csv(ST / "27_three_levels_auc.csv")
YEARS = ["2023(固定して適用)", "2024年1〜3月(未使用)"]
TITLE = {"2023(固定して適用)": "2023年", "2024年1〜3月(未使用)": "2024年1〜3月(未使用)"}
COL = {"低": BLUE, "中": MID, "高": ACC}

fig, ax = plt.subplots(1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for a, y in zip(ax[:2], YEARS):
    x0 = 0
    ticks, labels = [], []
    for scheme, nm in [("2段階(15%)", "2グループ"), ("3段階A(2022の3分位)", "3グループ")]:
        d = g[(g["対象"] == y) & (g["分け方"] == scheme)]
        xs = x0 + np.arange(len(d))
        for xx, (_, r) in zip(xs, d.iterrows()):
            a.bar(xx, 100 * r["実際の終了率"], 0.8, color=COL[r["グループ"]])
            a.text(xx, 100 * r["実際の終了率"] + 0.5, f"{100 * r['実際の終了率']:.1f}%", ha="center", fontsize=10)
            ticks.append(xx)
            labels.append(f"{r['グループ']}\n({100 * r['割合']:.0f}%)")
        a.text(xs.mean(), -10.5, nm, ha="center", fontsize=12, fontweight="bold", color=INK, clip_on=False)
        x0 = xs[-1] + 1.8
    a.set_xticks(ticks)
    a.set_xticklabels(labels, fontsize=10)
    a.set_ylim(0, 31)
    a.set_title(TITLE[y], loc="left", fontsize=13, fontweight="bold")
    a.set_ylabel("実際に7日以内に終わった割合(%)" if y == YEARS[0] else "")
b = ax[2]
rows = [("確率そのまま", GRAY), ("3段階A(2022の3分位)", MID), ("2段階(15%)", ACC)]
names = {"確率そのまま": "分けない", "3段階A(2022の3分位)": "3グループ", "2段階(15%)": "2グループ"}
w = 0.26
for i, y in enumerate(YEARS):
    for j, (k, c) in enumerate(rows):
        v = m[(m["対象"] == y) & (m["分け方"] == k)]["全体のAUC"].iloc[0]
        b.bar(i + (j - 1) * w, v, w * 0.92, color=c, label=names[k] if i == 0 else None)
        b.text(i + (j - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
b.axhline(0.5, color="black", ls="--", lw=0.9, label="0.5=当てずっぽう")
b.set_ylim(0.48, 0.69)
b.set_xticks([0, 1])
b.set_xticklabels(["2023年", "2024年1〜3月"], fontsize=11)
b.set_ylabel("AUC(当たり具合)")
b.legend(frameon=False, loc="upper center", ncol=2, fontsize=10)
fig.suptitle("3グループ版は2グループ版よりAUCが少し高い(2023年)。ただし差は小さく、2024年はほぼ同じ", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.01, "近隣10市+国分寺市、7日以内。境界は2022年で決め、動かさずに適用(2グループ=15%、3グループ=2022年の確率の3分位)。()内は全体に占める割合。\n2024年は3分位の境界が合わず、低が5%、高が66%になった。AUCは同点を半分当たりとして数えた。探索的。出典: summary_tables/27_three_levels_*.csv", fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
fig.savefig(HERE / "figures" / "27_three_vs_two.png", dpi=150)

"""28の結果(V2: 2022年のAUCが最大になる境界)を図にする。本編用。"""
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

g = pd.read_csv(ST / "28_fair_group_rates.csv")
m = pd.read_csv(ST / "28_fair_auc.csv")
YEARS = ["2023(固定して適用)", "2024年1〜3月(未使用)"]
TITLE = {YEARS[0]: "2023年", YEARS[1]: "2024年1〜3月(未使用)"}
COL = {"低": BLUE, "中": MID, "高": ACC}
S2, S3 = "V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)"

fig, ax = plt.subplots(1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for a, y in zip(ax[:2], YEARS):
    x0, ticks, labels = 0, [], []
    for scheme, nm in [(S2, "2グループ"), (S3, "3グループ")]:
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
rows = [("確率そのまま", GRAY, "分けない"), (S3, MID, "3グループ"), (S2, ACC, "2グループ")]
w = 0.26
for i, y in enumerate(YEARS):
    for j, (k, c, nm) in enumerate(rows):
        v = m[(m["対象"] == y) & (m["分け方"] == k)]["全体のAUC"].iloc[0]
        b.bar(i + (j - 1) * w, v, w * 0.92, color=c, label=nm if i == 0 else None)
        b.text(i + (j - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
b.axhline(0.5, color="black", ls="--", lw=0.9, label="0.5=当てずっぽう")
b.set_ylim(0.48, 0.69)
b.set_xticks([0, 1])
b.set_xticklabels(["2023年", "2024年1〜3月"], fontsize=11)
b.set_ylabel("AUC(当たり具合)")
b.legend(frameon=False, loc="upper center", ncol=2, fontsize=10)
fig.suptitle("3グループ版は2グループ版よりAUCが約0.01〜0.02高い。ただし差は小さい", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.01, "近隣10市+国分寺市、7日以内。2グループ・3グループとも、2022年のAUCが最大になる境界を同じ探し方で選び(2グループ=11.7%、3グループ=8.5%と14.0%)、動かさずに適用。\n()内は全体に占める割合。2024年は境界が合わず、2グループの高が74%、3グループの低が8%・高が63%になった。AUCは同点を半分当たりとして数えた。探索的。出典: summary_tables/28_fair_*.csv", fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
fig.savefig(HERE / "figures" / "28_fair_three_vs_two.png", dpi=150)

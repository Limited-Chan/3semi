"""28の結果(V2: 2022年のAUCが最大になる境界)を図にする。本編用。図の説明は figures/CAPTIONS.md。"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import figstyle  # noqa: E402
from figstyle import DARK, LIGHT, MIDC, REF  # noqa: E402

ST = HERE / "summary_tables"
figstyle.apply()

g = pd.read_csv(ST / "28_fair_group_rates.csv")
m27 = pd.read_csv(ST / "27_three_levels_auc.csv")
m = pd.read_csv(ST / "28_fair_auc.csv")
YEARS = ["2023(固定して適用)", "2024年1〜3月(未使用)"]
TITLE = {YEARS[0]: "(a) 2023年", YEARS[1]: "(b) 2024年1〜3月"}
COL = {"低": LIGHT, "中": MIDC, "高": DARK}
S2, S3 = "V2 2グループ(AUC最大)", "V2 3グループ(AUC最大)"

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for a, y in zip(ax[:2], YEARS):
    x0, ticks, labels = 0, [], []
    for scheme, nm in [(S2, "2グループ"), (S3, "3グループ")]:
        d = g[(g["対象"] == y) & (g["分け方"] == scheme)]
        xs = x0 + np.arange(len(d))
        for xx, (_, r) in zip(xs, d.iterrows()):
            a.bar(xx, 100 * r["実際の終了率"], 0.8, color=COL[r["グループ"]])
            a.text(xx, 100 * r["実際の終了率"] + 0.5, f"{100 * r['実際の終了率']:.1f}", ha="center", fontsize=10)
            ticks.append(xx)
            labels.append(r["グループ"])
        a.text(xs.mean(), -0.2, nm, ha="center", transform=a.get_xaxis_transform())
        x0 = xs[-1] + 1.8
    a.set_xticks(ticks, labels)
    a.set_ylim(0, 30)
    a.set_title(TITLE[y], loc="left", fontsize=12)
    a.set_ylabel("7日以内に終わった割合(%)" if y == YEARS[0] else "")
b = ax[2]
rows = [("確率そのまま", "o", "分けない", m), (S3, "s", "3グループ", m), (S2, "^", "2グループ(同じ探し方)", m), ("2段階(15%)", "D", "2グループ(15%)", m27)]
offs = [-0.27, -0.09, 0.09, 0.27]
for i, y in enumerate(YEARS):
    for j, (k, mk, nm, tab) in enumerate(rows):
        v = tab[(tab["対象"] == y) & (tab["分け方"] == k)]["全体のAUC"].iloc[0]
        b.plot([i + offs[j]], [v], mk, color=DARK, ms=8, label=nm if i == 0 else None)
b.axhline(0.5, color=REF, ls="--", lw=1.0)
b.set_ylim(0.5, 0.68)
b.set_xlim(-0.5, 1.5)
b.set_xticks([0, 1], ["2023年", "2024年1〜3月"])
b.set_ylabel("AUC(全物件)")
b.legend(loc="upper right", fontsize=10)
b.set_title("(c)", loc="left", fontsize=12)
fig.subplots_adjust(bottom=0.2)
figstyle.save(fig, HERE / "figures", "28_fair_three_vs_two.png", "図 2グループと3グループの公平な比較(境界を同じ探し方で決めた)",
              "近隣10市+国分寺市、7日以内。2グループ・3グループとも、2022年のAUCが最大になる境界を同じ探し方で選び(2グループ=11.7%、3グループ=8.5%と14.0%)、動かさずに適用。(a)(b) 7日以内に終わった割合(%)。(c) AUC(全物件。点線は0.5)。「2グループ(15%)」は目的から決めた15%。2024年は境界が合わず、2グループの高が74%、3グループの低が8%・高が63%になった。3グループは2グループよりAUCが少し高い(2023年 約0.02、2024年 約0.004〜0.013)が、差は小さい。AUCは同点を半分当たりとして数えた。探索的。出典: summary_tables/28_fair_*.csv")

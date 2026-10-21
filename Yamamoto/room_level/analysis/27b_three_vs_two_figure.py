"""27の結果(3段階と2段階の比較)を図にする。図の説明は figures/CAPTIONS.md。"""
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

g = pd.read_csv(ST / "27_three_levels_group_rates.csv")
m = pd.read_csv(ST / "27_three_levels_auc.csv")
YEARS = ["2023(固定して適用)", "2024年1〜3月(未使用)"]
TITLE = {YEARS[0]: "(a) 2023年", YEARS[1]: "(b) 2024年1〜3月"}
COL = {"低": LIGHT, "中": MIDC, "高": DARK}

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for a, y in zip(ax[:2], YEARS):
    x0, ticks, labels = 0, [], []
    for scheme, nm in [("2段階(15%)", "2グループ"), ("3段階A(2022の3分位)", "3グループ")]:
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
    a.set_ylim(0, 31)
    a.set_title(TITLE[y], loc="left", fontsize=12)
    a.set_ylabel("7日以内に終わった割合(%)" if y == YEARS[0] else "")
b = ax[2]
rows = [("確率そのまま", "o", "分けない"), ("3段階A(2022の3分位)", "s", "3グループ"), ("2段階(15%)", "^", "2グループ(15%)")]
offs = [-0.2, 0, 0.2]
for i, y in enumerate(YEARS):
    for j, (k, mk, nm) in enumerate(rows):
        v = m[(m["対象"] == y) & (m["分け方"] == k)]["全体のAUC"].iloc[0]
        b.plot([i + offs[j]], [v], mk, color=DARK, ms=8, label=nm if i == 0 else None)
b.axhline(0.5, color=REF, ls="--", lw=1.0)
b.set_ylim(0.5, 0.66)
b.set_xlim(-0.5, 1.5)
b.set_xticks([0, 1], ["2023年", "2024年1〜3月"])
b.set_ylabel("AUC(全物件)")
b.legend(loc="upper right")
b.set_title("(c)", loc="left", fontsize=12)
fig.subplots_adjust(bottom=0.2)
figstyle.save(fig, HERE / "figures", "27_three_vs_two.png", "図 3グループと2グループの比較(境界の決め方が違う、参考)",
              "近隣10市+国分寺市、7日以内。境界は2022年で決め、動かさずに適用(2グループ=目的から決めた15%、3グループ=2022年の確率の3分位)。(a)(b) 7日以内に終わった割合(%)。(c) AUC(全物件。点線は0.5)。2024年は3分位の境界が合わず、低が5%、高が66%になった。AUCは同点を半分当たりとして数えた。探索的。出典: summary_tables/27_three_levels_*.csv")

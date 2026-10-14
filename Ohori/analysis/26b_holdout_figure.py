"""26の結果と、2023年の結果(19・25)を並べた図を作る。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ST = HERE / "summary_tables"
INK, ACC, GRAY = "#4A1D3A", "#C2185B", "#B9A5B0"
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

h24 = pd.read_csv(ST / "26_holdout_2024q1.csv")
h24 = h24[(h24["対象"] == "近隣10市+国分寺市") & h24["目安"].str.startswith("モデル")].iloc[0]
w24 = pd.read_csv(ST / "26_holdout_2024q1_within_set.csv").set_index("候補の似かた")
b25 = pd.read_csv(ST / "25_overall_ci_baselines.csv")
b25 = b25[b25["目安"].str.startswith("モデル")].iloc[0]
l19 = pd.read_csv(ST / "19_labels_by_year.csv")
l19 = l19[(l19["予測期間(日)"] == 7) & l19["境界の決め方"].str.startswith("a") & l19["年"].str.startswith("2023")].iloc[0]
w23 = pd.read_csv(ST / "19_within_set_2023.csv")
w23 = w23[(w23["予測期間(日)"] == 7) & w23["境界の決め方"].str.startswith("a")].set_index("候補の似かた")

fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
vals = {"2023年": (100 * l19["低の実終了率"], 100 * l19["高の実終了率"]), "2024年1〜3月": (100 * h24["低の実終了率"], 100 * h24["高の実終了率"])}
x = np.arange(2)
for i, (k, (lo, hi)) in enumerate(vals.items()):
    for j, (v, c, nm) in enumerate([(lo, "#3949AB", "低"), (hi, ACC, "高")]):
        b = a.bar(i + (j - 0.5) * 0.36, v, 0.34, color=c, label=nm if i == 0 else None)
        a.text(i + (j - 0.5) * 0.36, v + 0.4, f"{v:.1f}%", ha="center", fontsize=11)
    a.text(i, max(hi, lo) + 3.2, f"{hi / lo:.2f}倍", ha="center", fontsize=13, fontweight="bold", color=INK)
a.set_xticks(x)
a.set_xticklabels(["2023年\n(境界を決めた後の確認)", "2024年1〜3月\n(未使用データ)"])
a.set_ylim(0, 33)
a.set_ylabel("7日以内に掲載が終わった割合(%)")
a.legend(frameon=False, loc="upper left")
a.set_title("高は低より終わりやすい(両方の年で)", loc="left", fontsize=13, fontweight="bold")

b = ax[1]
rows = [
    ("全体", (b25["差pt"], b25["差pt_ci_low"], b25["差pt_ci_high"]), (h24["差pt"], h24["差pt_ci_low"], h24["差pt_ci_high"])),
    ("間取りだけ同じ", tuple(w23.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]])),
    ("間取り・築年数・家賃まで同じ", tuple(w23.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]])),
]
for i, (nm, v23, v24) in enumerate(rows[::-1]):
    for j, (v, c, yr) in enumerate([(v23, GRAY, "2023年"), (v24, ACC, "2024年1〜3月")]):
        y = i + (0.18 if j == 0 else -0.18)
        b.barh(y, v[0], 0.32, color=c, label=yr if i == 0 else None)
        b.plot([v[1], v[2]], [y, y], color="#FF9EC4" if j == 1 else INK, lw=2.4, zorder=3)
        b.text(max(v[2], v[0]) + 0.3, y, f"{v[0]:+.1f}", va="center", fontsize=11, fontweight="bold" if j else "normal")
b.axvline(0, color="black", lw=0.8)
b.set_yticks(range(3))
b.set_yticklabels([r[0] for r in rows[::-1]], fontsize=11)
b.set_xlim(-5, 14)
b.set_xlabel("高−低の実終了率の差(ポイント、線は95%区間)")
b.legend(frameon=False, loc="lower right")
b.set_title("似た物件どうしでは、2024年は差が消えた", loc="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.01, "近隣10市+国分寺市、7日以内、境界15%・モデルは2023年と同じまま。2024年は1〜3月の5つの日付(候補15,400件)。\n2024年は高の割合が58%(2023年は約30%)。探索的。出典: summary_tables/26_holdout_2024q1*.csv、25_*.csv、19_*.csv", fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "figures" / "26_holdout_2024q1.png", dpi=150)

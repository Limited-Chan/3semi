"""26の結果と、2023年の結果(19・25)を並べた図を作る。図の説明は figures/CAPTIONS.md。"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import figstyle  # noqa: E402
from figstyle import DARK, LIGHT, errbar_h, panel  # noqa: E402

ST = HERE / "summary_tables"
figstyle.apply()

h24 = pd.read_csv(ST / "26_holdout_2024q1.csv")
h24 = h24[(h24["対象"] == "近隣10市+国分寺市") & h24["目安"].str.startswith("モデル")].iloc[0]
w24 = pd.read_csv(ST / "26_holdout_2024q1_within_set.csv").set_index("候補の似かた")
b25 = pd.read_csv(ST / "25_overall_ci_baselines.csv")
b25 = b25[b25["目安"].str.startswith("モデル")].iloc[0]
l19 = pd.read_csv(ST / "19_labels_by_year.csv")
l19 = l19[(l19["予測期間(日)"] == 7) & l19["境界の決め方"].str.startswith("a") & l19["年"].str.startswith("2023")].iloc[0]
w23 = pd.read_csv(ST / "19_within_set_2023.csv")
w23 = w23[(w23["予測期間(日)"] == 7) & w23["境界の決め方"].str.startswith("a")].set_index("候補の似かた")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
vals = {"2023年": (100 * l19["低の実終了率"], 100 * l19["高の実終了率"]), "2024年1〜3月": (100 * h24["低の実終了率"], 100 * h24["高の実終了率"])}
for i, (k, (lo, hi)) in enumerate(vals.items()):
    for j, (v, c, nm) in enumerate([(lo, LIGHT, "低"), (hi, DARK, "高")]):
        a.bar(i + (j - 0.5) * 0.36, v, 0.34, color=c, label=nm if i == 0 else None)
        a.text(i + (j - 0.5) * 0.36, v + 0.4, f"{v:.1f}", ha="center", fontsize=10.5)
a.set_xticks(range(2), ["2023年", "2024年1〜3月"])
a.set_ylim(0, 30)
a.set_ylabel("7日以内に終わった割合(%)")
a.legend(loc="upper left")
panel(a, "(a)")

b = ax[1]
rows = [
    ("全体", (b25["差pt"], b25["差pt_ci_low"], b25["差pt_ci_high"]), (h24["差pt"], h24["差pt_ci_low"], h24["差pt_ci_high"])),
    ("間取りだけ同じ", tuple(w23.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["D 間取りだけ同じ", ["高−低の実終了率の差pt", "ci_low", "ci_high"]])),
    ("間取り・築年数・家賃まで同じ", tuple(w23.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]]), tuple(w24.loc["A 間取り×築年数×家賃", ["高−低の実終了率の差pt", "ci_low", "ci_high"]])),
]
for i, (nm, v23, v24) in enumerate(rows[::-1]):
    for j, (v, c, yr) in enumerate([(v23, LIGHT, "2023年"), (v24, DARK, "2024年1〜3月")]):
        y = i + (0.18 if j == 0 else -0.18)
        b.barh(y, v[0], 0.32, color=c, label=yr if i == 0 else None)
        errbar_h(b, y, v[1], v[2])
b.axvline(0, color="#222222", lw=0.8)
b.set_yticks(range(3), [r[0] for r in rows[::-1]])
b.set_xlim(-5, 14)
b.set_xlabel("高と低の差(ポイント)")
b.legend(loc="lower right")
panel(b, "(b)")
figstyle.save(fig, HERE / "figures", "26_holdout_2024q1.png", "図 2024年1〜3月(一度も使っていないデータ)での確認",
              "近隣10市+国分寺市、7日以内、境界15%、モデルは2023年と同じまま。2024年は1〜3月の5つの日付(候補15,400件)。(a) 高と低の7日以内の終了率(2023年は1.95倍、2024年は1.61倍)。(b) 高と低の差(ポイント)。棒=差、誤差線=95%区間。2024年は高の割合が58%(2023年は約30%)。探索的。出典: summary_tables/26_holdout_2024q1*.csv、25_*.csv、19_*.csv")

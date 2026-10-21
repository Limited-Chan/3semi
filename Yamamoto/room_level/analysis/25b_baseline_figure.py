"""25の結果(summary_tables/25_overall_ci_baselines.csv)から図を作る。"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
INK, ACC, GRAY = "#4A1D3A", "#C2185B", "#B9A5B0"
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

t = pd.read_csv(HERE / "summary_tables" / "25_overall_ci_baselines.csv").iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 5.2))
for i, r in t.iterrows():
    model = r["目安"].startswith("モデル")
    ax.barh(i, r["差pt"], color=ACC if model else (GRAY if r["目安"].startswith("でたらめ") else INK), height=0.62)
    ax.plot([r["差pt_ci_low"], r["差pt_ci_high"]], [i, i], color="#FF9EC4", lw=3.2, solid_capstyle="butt", zorder=3)
    for x in (r["差pt_ci_low"], r["差pt_ci_high"]):
        ax.plot([x, x], [i - 0.18, i + 0.18], color="#FF9EC4", lw=2.5, zorder=3)
    ax.text(max(r["差pt_ci_high"], r["差pt"]) + 0.25, i, f"{r['差pt']:+.1f}pt ({r['倍率(高/低)']:.2f}倍)", va="center", fontsize=11,
            fontweight="bold" if model else "normal")
ax.set_yticks(range(len(t)))
ax.set_yticklabels(t["目安"], fontsize=11)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-1.5, 11.5)
ax.set_xlabel("高−低の実終了率の差(ポイント、ピンクの線は95%区間)")
ax.set_title("高と低の差は偶然ではなく、1つの条件だけの目安より大きい", loc="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.01, "近隣10市+国分寺市の2023年候補(33,672件、7日以内、高の割合はどれも約30%にそろえた)。\n目安は2022年の候補から作成。95%区間は部屋単位の抽出し直し。探索的。出典: summary_tables/25_overall_ci_baselines.csv",
         fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "figures" / "25_baselines.png", dpi=150)

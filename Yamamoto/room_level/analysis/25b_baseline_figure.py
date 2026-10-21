"""25の結果(summary_tables/25_overall_ci_baselines.csv)から図を作る。図の説明は figures/CAPTIONS.md。"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import figstyle  # noqa: E402
from figstyle import MIDC, errbar_h  # noqa: E402

figstyle.apply()
NAMES = {"モデル(今回)": "今回の方法", "掲載日数だけ": "掲載日数だけ", "家賃帯だけ": "家賃だけ", "間取りだけ": "間取りだけ", "築年数帯だけ": "築年数だけ",
         "間取り×築年数×家賃": "3つの条件を組み合わせた集計", "でたらめ(並べ替え)": "ランダムに分けた場合"}

t = pd.read_csv(HERE / "summary_tables" / "25_overall_ci_baselines.csv").iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, r in t.iterrows():
    ax.barh(i, r["差pt"], color=MIDC, height=0.6)
    errbar_h(ax, i, r["差pt_ci_low"], r["差pt_ci_high"])
ax.set_yticks(range(len(t)), [NAMES[k] for k in t["目安"]])
ax.axvline(0, color="#222222", lw=0.8)
ax.set_xlim(-1.5, 9)
ax.set_xlabel("高と低の差(ポイント)")
figstyle.save(fig, HERE / "figures", "25_baselines.png", "図 高と低の差と、1つの条件だけの目安との比較(2023年)",
              "近隣10市+国分寺市の2023年候補(33,672件、7日以内、高の割合はどれも約30%にそろえた)。棒=高の終了率−低の終了率(ポイント)、誤差線=95%区間(部屋単位の抽出し直し)。目安は2022年の候補から作成。探索的。出典: summary_tables/25_overall_ci_baselines.csv")

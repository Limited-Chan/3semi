"""「なぜ7日以内か」の根拠の図(探索的)。

左: 似た候補の中のAUC(間取り・築年数・家賃が近い物件どうし)。7日・14日・30日でほとんど変わらない。
右: リスク上位30%を「高」にしたとき、高の終了率は低の何倍か。短い日数ほど区別がつく。
入力: summary_tables/17_auc_by_model.csv(AUC)、results/17_fixed_window_features/candidates_2023.parquet(倍率)
出力: figures/22_horizon_choice.png、summary_tables/22_horizon_choice.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CAND = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
AUC = HERE / "summary_tables" / "17_auc_by_model.csv"
FIG = HERE / "figures" / "22_horizon_choice.png"
TABLE = HERE / "summary_tables" / "22_horizon_choice.csv"
HORIZONS = [7, 14, 30]
POPS = [("近隣10市+国分寺市", "#4A1D3A"), ("国分寺市のみ", "#C2185B")]
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Maru Gothic Pro", "sans-serif"]


def lift_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _ in POPS:
        d = c if label.startswith("近隣") else c[c["city"] == "国分寺市"]
        for h in HORIZONS:
            p, y = d[f"base_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
            hi = p >= np.quantile(p, 0.7)
            rows.append({"評価対象": label, "予測期間(日)": h, "候補数": len(d), "全体の終了率": y.mean(),
                         "高の終了率": y[hi].mean(), "低の終了率": y[~hi].mean(),
                         "高は低の何倍": y[hi].mean() / y[~hi].mean(), "高は全体の何倍": y[hi].mean() / y.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    lift = lift_table(pd.read_parquet(CAND))
    auc = pd.read_csv(AUC)
    auc = auc[auc["候補の似かた"].str.startswith("A")][["評価対象", "予測期間(日)", "base"]].rename(columns={"base": "似た候補のAUC"})
    auc["評価対象"] = auc["評価対象"].replace({"国分寺市": "国分寺市のみ"})
    table = lift.merge(auc, on=["評価対象", "予測期間(日)"])
    table.to_csv(TABLE, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    x = np.arange(len(HORIZONS))
    w = 0.36
    for ax, col, title, ylim, fmt in [
        (axes[0], "似た候補のAUC", "似た候補の中のAUC\n(日数による差は小さい。近隣10市+国分寺市は0.54〜0.55)", (0.5, 0.6), "{:.3f}"),
        (axes[1], "高は低の何倍", "「高」は「低」の何倍終わりやすいか\n(短い日数ほど、区別がつく)", (1.0, 2.4), "{:.2f}倍"),
    ]:
        for i, (label, color) in enumerate(POPS):
            v = table[table["評価対象"] == label].sort_values("予測期間(日)")[col].to_numpy()
            bars = ax.bar(x + (i - 0.5) * w, v, w, label=label, color=color)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [f"{h}日以内" for h in HORIZONS])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("AUC(0.5=サイコロ)")
    axes[1].set_ylabel("高の終了率 ÷ 低の終了率")
    axes[0].legend(frameon=False, loc="upper right", fontsize=10)
    fig.suptitle("図 予測期間の選び方: 7日・14日・30日の比較(2023年の候補)", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01,
             "左: 同時に掲載中で間取り・築年数・家賃が近い物件どうしの候補内AUC。右: リスク上位30%を高、残りを低とした。"
             "7日と14日の差は小さい。モデル: 2021〜2022年開始で学習したModel 2。探索的。",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG, dpi=150)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

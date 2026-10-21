"""「なぜ7日以内か」の根拠の図(探索的)。

左: 似た候補の中のAUC(間取り・築年数・家賃が近い物件どうし)。7日・14日・30日でほとんど変わらない。
右: リスク上位30%を「高」にしたとき、高の終了率は低の何倍か。短い日数ほど区別がつく。
入力: summary_tables/17_auc_by_model.csv(AUC)、results/17_fixed_window_features/candidates_2023.parquet(倍率)
出力: figures/22_horizon_choice.png(図の説明は figures/CAPTIONS.md)、summary_tables/22_horizon_choice.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import figstyle  # noqa: E402
from figstyle import DARK, MIDC, REF, panel  # noqa: E402

CAND = HERE / "results" / "17_fixed_window_features" / "candidates_2023.parquet"
AUC = HERE / "summary_tables" / "17_auc_by_model.csv"
FIGDIR = HERE / "figures"
TABLE = HERE / "summary_tables" / "22_horizon_choice.csv"
HORIZONS = [7, 14, 30]
POPS = [("近隣10市+国分寺市", DARK, "o"), ("国分寺市のみ", MIDC, "s")]
figstyle.apply()


def lift_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _, _ in POPS:
        d = c if label.startswith("近隣") else c[c["city"] == "国分寺市"]
        for h in HORIZONS:
            p, y = d[f"base_{h}"].to_numpy(), d[f"target_{h}"].to_numpy()
            hi = p >= np.quantile(p, 0.7)
            rows.append({"評価対象": label, "予測期間(日)": h, "候補数": len(d), "全体の終了率": y.mean(),
                         "高の終了率": y[hi].mean(), "低の終了率": y[~hi].mean(),
                         "高は低の何倍": y[hi].mean() / y[~hi].mean(), "高は全体の何倍": y[hi].mean() / y.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    lift = lift_table(pd.read_parquet(CAND))
    auc = pd.read_csv(AUC)
    auc = auc[auc["候補の似かた"].str.startswith("A")][["評価対象", "予測期間(日)", "base"]].rename(columns={"base": "似た候補のAUC"})
    auc["評価対象"] = auc["評価対象"].replace({"国分寺市": "国分寺市のみ"})
    table = lift.merge(auc, on=["評価対象", "予測期間(日)"])
    table.to_csv(TABLE, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(HORIZONS))
    for ax, col, ylim, ylab in [(axes[0], "似た候補のAUC", (0.49, 0.60), "AUC(似た候補の中)"), (axes[1], "高は低の何倍", (1.0, 2.5), "高の終了率 ÷ 低の終了率")]:
        for i, (label, color, mk) in enumerate(POPS):
            v = table[table["評価対象"] == label].sort_values("予測期間(日)")[col].to_numpy()
            ax.plot(x + (i - 0.5) * 0.12, v, mk, color=color, ms=8, label=label)
        ax.set_xticks(x, [f"{h}日以内" for h in HORIZONS])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
    axes[0].axhline(0.5, color=REF, ls="--", lw=1.0)
    axes[0].legend(loc="upper right")
    panel(axes[0], "(a)")
    panel(axes[1], "(b)")
    figstyle.save(fig, FIGDIR, "22_horizon_choice.png", "図 予測期間(7日・14日・30日)の比較(2023年の候補)",
                  "(a) 同時に掲載中で間取り・築年数・家賃が近い物件どうしの候補内AUC。点線は0.5(ランダム)。(b) リスク上位30%を高、残りを低としたときの、高の終了率÷低の終了率。モデル: 2021〜2022年開始で学習したModel 2。探索的。")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""発表の補足スライド(想定問答)に貼る図を作る。

数字は、これまでの分析の表から読む(手で入れた数字は使わない):
  summary_tables/31_slide_numbers.csv     2023年の倍率・AUC・終了率(31_slide_figures.py の出力)
  summary_tables/26_holdout_2024q1.csv     2024年1〜3月の結果(国分寺市のみ・近隣10市+国分寺市)
31e(単純な分類との比較)と31g(閾値)は、31_slide_figures.py の関数をそのまま使い、補足スライド用の大きさで描き直す。
出力: presentation/figures/33a_ratio_ci.png, 33b_level.png, 33c_boundary.png, 33d_auc_schemes.png, 33e_baselines.png
実行: cd Yamamoto/room_level && python3 analysis/33_appendix_figures.py
"""
from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))
m = importlib.import_module("analysis.31_slide_figures")
m.set_talk(True)
m.TARGET_PX = 36  # 補足スライドは、左の列(幅360pt)に貼るので、本編より少し大きい文字にする
W = 960  # 左の列に貼る幅(1920幅のスライドでのpx)
OUT = HERE / "presentation" / "figures"
HI, BASE, PALE = m.HI, m.BASE, m.PALE

NUMS = pd.read_csv(HERE / "summary_tables" / "31_slide_numbers.csv")
H24 = pd.read_csv(HERE / "summary_tables" / "26_holdout_2024q1.csv")


def nv(fig: str, item: str) -> float:
    return float(NUMS[(NUMS["図"] == fig) & (NUMS["項目"] == item)]["値"].iloc[0])


def row24(scope: str) -> pd.Series:
    return H24[(H24["対象"] == scope) & (H24["目安"] == "モデル(今回)")].iloc[0]


def fig_ratio() -> None:
    """倍率と95%区間(2023年・2024年1〜3月・国分寺市のみ)。1を含まなければ、差があると読める。"""
    near, kok = row24("近隣10市+国分寺市"), row24("国分寺市のみ")
    rows = [
        ("2023年", nv("31d", "倍率"), nv("31e", "今回の方法 倍率(高/低) 95%区間 下"), nv("31e", "今回の方法 倍率(高/低) 95%区間 上"), HI),
        ("2024年1〜3月", near["倍率(高/低)"], near["倍率_ci_low"], near["倍率_ci_high"], HI),
        ("2024年1〜3月\n国分寺市のみ", kok["倍率(高/低)"], kok["倍率_ci_low"], kok["倍率_ci_high"], BASE),
    ]
    fig, a = m.new(W, 760, target=36)
    size = plt.rcParams["font.size"]
    gap = 1.9  # 行の間隔
    box = dict(facecolor="white", edgecolor="none", pad=1.5)
    for i, (lab, v, lo, hi, c) in enumerate(rows):
        y = (len(rows) - 1 - i) * gap
        a.plot([lo, hi], [y, y], color=c, lw=4, solid_capstyle="butt", zorder=2)
        a.plot([v], [y], "o", color=c, ms=15, zorder=3)
        a.text(v, y + 0.30, f"{v:.2f}倍", ha="center", va="bottom", fontsize=size, bbox=box, zorder=4)
        a.text(v, y - 0.30, f"({lo:.2f}〜{hi:.2f})", ha="center", va="top", fontsize=size * 0.86, color="#333333", bbox=box, zorder=4)
        m.note("33a", f"{lab.replace(chr(10), '')} 倍率", v)
    a.axvline(1, color="#222222", ls="--", lw=1.6)
    a.text(1.0, 2 * gap + 1.0, "差なし(1倍)", ha="center", va="bottom", fontsize=size * 0.9)
    a.set_yticks([k * gap for k in range(len(rows))])
    a.set_yticklabels([r[0] for r in rows][::-1])
    a.set_ylim(-1.2, 2 * gap + 1.7)
    a.set_xlim(0.5, 2.5)
    a.grid(axis="x", visible=True)
    a.grid(axis="y", visible=False)
    a.set_xlabel("倍率(終わりやすい群÷終わりにくい群)\n点は推定値、線は95%区間")
    fig.tight_layout()
    m.save(fig, "33a_ratio_ci.png")


def fig_level() -> None:
    """2023年と2024年1〜3月で、全体の終了率の水準が違う(終わりやすい群の割合も違う)。"""
    near = row24("近隣10市+国分寺市")
    data = [
        ("2023年", nv("31d", "低 終了率(%)"), nv("31d", "全体 終了率(%)"), nv("31d", "高 終了率(%)"), nv("31d", "高の割合(%)")),
        ("2024年\n1〜3月", 100 * near["低の実終了率"], 100 * near["全体"], 100 * near["高の実終了率"], 100 * near["高の割合"]),
    ]
    fig, a = m.new(W, 680, target=36)
    size = plt.rcParams["font.size"]
    w = 0.27
    for j, (nm, col) in enumerate([("終わりにくい群", PALE), ("全体", BASE), ("終わりやすい群", HI)]):
        xs = [i + (j - 1) * (w + 0.03) for i in range(2)]
        vs = [d[1 + j] for d in data]
        a.bar(xs, vs, w, color=col, label=nm)
        for x, v in zip(xs, vs):
            a.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=size * 0.92)
    a.set_xticks(range(2))
    a.set_xticklabels([f"{d[0]}\n(終わりやすい群 {d[4]:.0f}%)" for d in data])
    a.set_ylim(0, 36)
    a.set_ylabel("掲載終了率(%)")
    a.legend(loc="upper left", fontsize=size * 0.86)
    for d in data:
        m.note("33b", f"{d[0].replace(chr(10), '')} 高の割合(%)", d[4])
    fig.tight_layout()
    m.save(fig, "33b_level.png")


def fig_auc_schemes() -> None:
    """分け方ごとのAUC(95%区間つき)。2023年と2024年1〜3月。"""
    schemes = [
        ("分けない\n(確率のまま)", "分けない(確率そのまま)"),
        ("3グループ", "3グループ版"),
        ("2グループ\n(AUC最大の境界)", "2グループ版"),
        ("2グループ\n(15%・採用)", "2グループ版(15%)"),
    ]
    fig, a = m.new(W, 700, target=36)
    size = plt.rcParams["font.size"]
    for k, (yr, col, mk, off) in enumerate([("2023", HI, "o", 0.17), ("2024", BASE, "s", -0.17)]):
        for i, (lab, key) in enumerate(schemes):
            y = len(schemes) - 1 - i + off
            v = nv("31c", f"{yr} AUC {key}")
            lo, hi = nv("31c", f"{yr} AUC {key} 95%区間 下"), nv("31c", f"{yr} AUC {key} 95%区間 上")
            a.plot([lo, hi], [y, y], color=col, lw=3.5, solid_capstyle="butt", zorder=2)
            a.plot([v], [y], mk, color=col, ms=12, zorder=3, label="2023年" if (i == 0 and k == 0) else ("2024年1〜3月" if (i == 0 and k == 1) else None))
            m.note("33d", f"{yr} {lab.replace(chr(10), '')} AUC", v)
    a.axvline(0.5, color="#222222", ls="--", lw=1.4)
    a.set_yticks(range(len(schemes)))
    a.set_yticklabels([s[0] for s in schemes][::-1])
    a.set_ylim(-0.6, 3.9)
    a.set_xlim(0.49, 0.65)
    a.set_xticks([0.5, 0.55, 0.6, 0.65])
    a.grid(axis="x", visible=True)
    a.grid(axis="y", visible=False)
    a.set_xlabel("AUC(0.5 = 半分だけ当たる)")
    a.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=size * 0.86, columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    m.save(fig, "33d_auc_schemes.png")


def fig_baselines(c22: pd.DataFrame, c23: pd.DataFrame) -> None:
    """単純な分類との比較(31eと同じ計算・同じ数字。ラベルを短くして、補足スライドの幅に合わせる)。"""
    short = {"3つの条件を組み合わせた集計": "3条件の集計", "ランダムに分けた場合": "ランダム"}
    rows = m.baseline_rows(c22, c23)
    fig, a = m.new(W, 700, target=36)
    size = plt.rcParams["font.size"]
    for i, (k, gap, lo, hi) in enumerate(rows):
        y = i  # 31eと同じ並び(表の最後が下、「今回の方法」が上)
        a.barh(y, gap, 0.6, color=HI if k == "今回の方法" else BASE)
        a.plot([lo, hi], [y, y], color="#222222", lw=1.6, marker="|", ms=11, mew=1.6, zorder=3)
        if k in ("今回の方法", "3つの条件を組み合わせた集計"):
            a.text(hi + 0.25, y, f"{gap:+.1f}", va="center", fontsize=size * 0.92, color=HI if k == "今回の方法" else "#333333")
        m.note("33e", f"{k} 差(ポイント)", gap)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([short.get(r[0], r[0]) for r in rows])
    a.axvline(0, color="#222222", lw=0.8)
    a.set_xlim(-1.5, 12)
    a.set_xticks([0, 5, 10])
    a.grid(axis="x", visible=True)
    a.grid(axis="y", visible=False)
    a.set_xlabel("差(ポイント)\n終わりやすい群 − 終わりにくい群")
    fig.tight_layout()
    m.save(fig, "33e_baselines.png")


def fig_boundary(c22: pd.DataFrame, c23: pd.DataFrame) -> None:
    """閾値ごとの差(31gと同じ計算・同じ数字。補足スライドの幅に合わせて、軸ラベルを2行にする)。"""
    grid = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
    fig, a = m.new(W, 680, target=36)
    for d, color, mk, lab in [(c22, m.MIDC, "s", "2022年(決めた年)"), (c23, m.DARK, "o", "2023年(確認)")]:
        v = [100 * m.pair_V(d, t) for t in grid]
        a.plot([100 * t for t in grid], v, marker=mk, color=color, lw=1.8, ms=7, label=lab)
        for t, x in zip(grid, v):
            m.note("33c", f"{lab} 線{100 * t:.0f}%", x)
    a.axvspan(10, 15, color=HI, alpha=0.15, lw=0)
    a.axvline(15, color=HI, ls="--", lw=2.0)
    a.set_xlim(5, 30)
    a.set_ylim(0, 3.4)
    a.set_xticks([5, 10, 15, 20, 25, 30])
    a.set_xlabel("閾値(7日以内の確率が何%以上を\n「終わりやすい」にするか)")
    a.set_ylabel("差(100組あたり)")
    a.legend(loc="upper right", fontsize=plt.rcParams["font.size"] * 0.86)
    fig.tight_layout()
    m.save(fig, "33c_boundary.png")


def via_alt(fn, args, src: str, dst: str, keys: tuple[int, ...]) -> None:
    """31_slide_figures.py の図を、補足スライド用の大きさで描き直し、別名で保存する(31の図は上書きしない)。"""
    tmp, old_fig = Path(tempfile.mkdtemp()), m.FIG
    old_w = {k: m.DISPLAY_W.get(k) for k in keys}
    m.FIG = tmp
    for k in keys:
        m.DISPLAY_W[k] = W
    try:
        fn(*args)
    finally:
        m.FIG = old_fig
        for k, v in old_w.items():
            if v is None:
                m.DISPLAY_W.pop(k, None)
            else:
                m.DISPLAY_W[k] = v
    shutil.move(str(tmp / src), str(OUT / dst))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c22, c23 = (pd.read_parquet(p).reset_index(drop=True) for p in (m.C22, m.C23))
    fig_ratio()
    fig_level()
    fig_auc_schemes()
    fig_boundary(c22, c23)
    fig_baselines(c22, c23)
    print(pd.DataFrame(m.NUMS).query("図.str.startswith('33')", engine="python").to_string(index=False))


if __name__ == "__main__":
    main()

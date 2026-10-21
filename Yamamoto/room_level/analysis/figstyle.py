"""論文調の図の共通設定と、図の説明(キャプション)の記録。

- 色は青の濃淡を基本にする(色覚に配慮)。強調のための色づけはしない。
- 図の中にはタイトルや説明文を入れない。図の説明は figures/CAPTIONS.md に、図の名前ごとに書く。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DARK, MIDC, LIGHT, PALE, REF = "#1F4E79", "#5B7C99", "#A9BCCF", "#C9C9C9", "#777777"


def apply() -> None:
    plt.rcParams.update({
        "font.family": ["Hiragino Sans", "sans-serif"], "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
        "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "legend.frameon": False, "figure.facecolor": "white",
    })


def panel(ax, label: str) -> None:
    ax.set_title(label, loc="left", fontsize=12)


def errbar_h(ax, y: float, lo: float, hi: float, color: str = "#222222") -> None:
    """横向きの誤差線(両端に縦棒)。"""
    ax.plot([lo, hi], [y, y], color=color, lw=1.5, marker="|", ms=9, mew=1.5, zorder=3)


def errbar_v(ax, x: float, lo: float, hi: float, color: str = DARK) -> None:
    ax.plot([x, x], [lo, hi], color=color, lw=1.4, marker="_", ms=8, mew=1.4, zorder=2)


def save(fig, fig_dir: Path, name: str, title: str, note: str) -> None:
    """図を保存し、図の説明を figures/CAPTIONS.md に書く(同じ名前なら上書き)。"""
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=150)
    plt.close(fig)
    cap_json = fig_dir / "captions.json"
    caps = json.loads(cap_json.read_text(encoding="utf-8")) if cap_json.exists() else {}
    caps[name] = {"title": title, "note": note}
    cap_json.write_text(json.dumps(caps, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    lines = ["# 図の説明\n", "図の中にはタイトルや説明文を入れていない。ここに、図の名前ごとの説明を書く(スクリプトが自動で更新する)。\n"]
    for k in sorted(caps):
        lines.append(f"## {k}\n\n**{caps[k]['title']}**\n\n{caps[k]['note']}\n")
    (fig_dir / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")

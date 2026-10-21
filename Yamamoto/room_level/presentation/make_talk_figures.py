"""発表用の図を作る(文字を大きく、見せたい1つの比較だけに強調色を付ける)。

図の計算は analysis/31_slide_figures.py と同じ(同じデータ・同じ数字)。見た目だけを発表用に切り替える。
出力: presentation/figures/31*.png(数字の表 summary_tables/31_slide_numbers.csv は更新しない)
実行: cd Yamamoto/room_level && python3 presentation/make_talk_figures.py
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "Ohori"))  # src/ (Ohori のコード)を読む
m = importlib.import_module("analysis.31_slide_figures")

if __name__ == "__main__":
    m.set_talk(True)
    m.main()

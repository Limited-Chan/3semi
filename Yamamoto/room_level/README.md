# 部屋単位・7日以内の「掲載終了しやすさ」の分析(Yamamoto)

国分寺市(と多摩地域)の賃貸掲載データで、物件の条件から「7日以内に掲載が終わりやすいか」を、2グループ(境界15%)で示す方法を作り、比べて選び、確かめた分析です。発表スライドの図と数字は、すべてここのスクリプトで計算しています。

| 場所 | 内容 |
|---|---|
| `analysis/08〜28` | 部屋単位への統合、似た物件の中での順位付け、予測期間・グループ数・境界の比較、誤差範囲、2024年1〜3月での確認 |
| `analysis/30〜31` | 発表スライド用の図。**31番が、スライドの全グラフ・例・件数をデータから直接計算する** |
| `analysis/figstyle.py` | 論文調の図の共通設定。図の説明は `figures/CAPTIONS.md` に自動で書かれる |
| `room_lib/listing_episodes.py` | 建物ID×住戸IDで、同じ部屋の重複・連続掲載を統合する(`tests/` で検証) |
| `summary_tables/` | 各分析の結果の要約表(`31_slide_numbers.csv` は、スライドに出した数字の一覧) |
| `figures/` | 図(`31*` が発表スライド用) |
| `ROOM_LEVEL_ANALYSIS.md` | 結果の要点と、実行順(依存関係つき) |
| `FIGURE_SOURCES.md` | スライドのページごとに、図・入力データ・件数・コードを対応づけた表(面談用) |

## 実行
```
cd Yamamoto/room_level
pip install -r requirements.txt
python -m analysis.17_fixed_window_features   # 実行順は ROOM_LEVEL_ANALYSIS.md
python analysis/31_slide_figures.py           # スライドの図と数字(約1分)
python -m unittest discover -s tests -p "test_listing_episodes.py"
```

## 他の人のコードを読む部分
確率の式の部品(`Ohori/src/data.py`、`discrete_model.py`、`survival.py`、`person_period.py`)は、Ohori のコードをそのまま読みます(複製していません)。各スクリプトが `Ohori/` を import の探索先に足しています。
元のTSV(`bukken_rent_v1.0.1.tsv`)と、`data/`・`results/` はGitHubに含めません。

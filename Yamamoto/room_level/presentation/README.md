# 発表用の資料

| ファイル | 内容 |
|---|---|
| `figures/31*.png` | スライドに入れる図(文字を大きくし、見せたい比較だけに強調色)。**スライドにはこれを使う** |
| `app_mockup.html` / `.png` | アプリ案の画面イメージ(唯一、HTMLで作った絵。右側の数字は24・31番で確認済み) |
| `FIGURE_SOURCES.md` | スライドのページごとに、図・入力データ・件数・コードを対応づけた表(面談用) |
| `FIVE_SECOND_TEST.md` | 5秒テスト(スライドが一目で分かるかの確認手順と、ページごとの「言えてほしいこと」) |
| `make_talk_figures.py` | 発表用の図を作る。計算は `analysis/31_slide_figures.py` と同じで、見た目だけ発表用にする |

```
cd Yamamoto/room_level
python3 presentation/make_talk_figures.py   # presentation/figures/ を作り直す(約1分)
```

論文用の図(落ち着いた1色の濃淡)と数字の表 `summary_tables/31_slide_numbers.csv` は、`python3 analysis/31_slide_figures.py` が作る。

# 3semi

国分寺市の賃貸物件データを用いて、物件属性と掲載期間の関係を分析するプロジェクトです。

## 分析担当

### Ohori
等幅区間を用いた探索分析。

- 掲載期間・家賃・専有面積・築年数・間取りの分布確認
- 平均値・中央値・最頻値の比較
- 掲載期間と各属性の1対1分析
- 複数属性を組み合わせた分析
- 分析結果のPNG・CSVを保存

詳細は Ohori/README.md を参照してください。

### Yamamoto
築年数・間取り・掲載期間の3変数分析。

- 各属性と掲載期間の関連度比較
- 築年数5年刻み × 間取りのクロス集計
- 3D棒グラフ・散布図
- 中央値とQ1〜Q3を用いた範囲表示

詳細は Yamamoto/report.md を参照してください。

## ディレクトリ

3semi/
├─ Ohori/
│  ├─ analyze.py
│  ├─ results_inspect/
│  └─ results_relate/
├─ Yamamoto/
│  ├─ analyze.py
│  ├─ output/
│  ├─ tables/
│  └─ report.md
└─ README.md

元のTSVおよび分析用の一時データはGitHubには含めません。

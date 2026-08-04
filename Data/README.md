# Data

## 分析対象データ

分析には次のLIFULL HOME'S掲載期間データを使用します。

- ファイル名: bukken_rent_v1.0.1.tsv
- 容量: 約7.1GB

ファイルサイズがGitHubの上限を超えるため、TSV本体はリポジトリには含めません。

各自でデータを用意し、ファイルパスを指定して分析してください。

## 実行例

py .\Data\analyze_listings.py all "C:\path\to\bukken_rent_v1.0.1.tsv" --out-dir .\analysis_output

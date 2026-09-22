# Changelog

## v2.3.0

- overflow/top-coded numeric binsを廃止
- 対象範囲の最大値まで完全な等幅区間を生成
- 集計CSVは全区間を1ファイルに保持
- PNGのみ固定区間数で分割
- 分割PNG間で共通y軸を使用
- 0件区間を保持
- 折れ線で空区間を接続しない
- Step 1で生成した完全なband_schemaをStep 2で再利用
- 2変数図も表示のみページ分割

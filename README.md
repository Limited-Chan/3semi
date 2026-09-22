# 国分寺市 賃貸掲載データ探索分析 v2.3

この版は、数値属性の横軸を全範囲で等間隔に保ち、上側の値を overflow / top-code でまとめません。
統計上の区間と、画像上の表示範囲を分離しています。

## 固定方針

- 家賃: 2万円刻み
- 面積: 10m²刻み
- 築年数: 5年刻み
- 掲載日数分布: 30日刻み
- 最大値まで同じ幅で区切る
- `20万円以上` や `100m²以上` のようなまとめ区間は作らない
- 全区間の集計結果は1つのCSVに保存する
- PNGだけ複数枚に分割する
- 0件区間もCSV・横軸上に保持する
- 折れ線では値がない区間をNaNとして線を途切れさせる
- 同じ変数の分割PNGは共通のy軸範囲を使う
- グラフはすべてPython / matplotlibが生成する

## 環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_analysis.py
```

## 1. 元TSVから国分寺市を抽出

入力パスが実ファイルの場合:

```powershell
.\.venv\Scripts\python.exe analyze.py extract `
  --input "C:\path\to\bukken_rent_v1.0.1.tsv" `
  --output "prepared\kokubunji.csv.gz" `
  --prefecture "東京都" `
  --city "国分寺市"
```

## 2. Step 1: 分布確認

```powershell
.\.venv\Scripts\python.exe analyze.py inspect `
  --input "prepared\kokubunji.csv.gz" `
  --out "results_inspect" `
  --years "2020,2021,2022,2023" `
  --types all `
  --duplicates keep `
  --rent-step-yen 20000 `
  --rent-bins-per-figure 10 `
  --area-step-sqm 10 `
  --area-bins-per-figure 10 `
  --age-step-years 5 `
  --age-bins-per-figure 10 `
  --duration-step-days 30 `
  --duration-bins-per-figure 12
```

`bins-per-figure` は表示だけを分割する設定です。集計区間には影響しません。

例: 家賃2万円刻み・10区間/図なら、1枚目は0–20万円、2枚目は20–40万円…となります。

### 出力例

```text
results_inspect/
├─ 00_scope/
│  ├─ band_config.json
│  └─ band_schema.json
├─ 01_distribution/
│  ├─ rent/
│  │  ├─ money_room_distribution_bar.csv       # 全範囲
│  │  ├─ ...part01....png                      # 表示だけ分割
│  │  └─ ...part02....png
│  ├─ area/
│  ├─ age/
│  ├─ duration/
│  └─ layout/
└─ 02_representative/
```

## 3. Step 2: 代表値を決めて関係を見る

Step 1を見て中央値を採用した場合の例:

```powershell
.\.venv\Scripts\python.exe analyze.py relate `
  --input "prepared\kokubunji.csv.gz" `
  --scope "results_inspect\scope.json" `
  --out "results_relate" `
  --stat median `
  --reason "分布を確認した結果、長期側に裾があり、典型的な掲載日数の比較には中央値が適切と判断したため"
```

Step 2はStep 1の `band_schema` をそのまま再利用するため、横軸区間が途中で変わりません。

## 画像と統計の関係

- CSV = 分析結果そのもの。全範囲を保持。
- PNG = CSVを読みやすく表示するために分割したもの。
- PNGの分割境界は統計的なカテゴリ境界ではありません。

## GitHubに含めないもの

`.gitignore` により次を除外します。

- 元TSV
- `prepared/`
- `results_*`
- `.venv/`
- 生成ZIP

# すぐ埋まる？ナビ — Webアプリ（発表デモ）

このフォルダは、研究リポジトリ `Limited-Chan/3semi` 内の **アプリ専用フォルダ** です。既存の分析コードは変更しません。

## 起動方法

リポジトリの `app/` に移動し、次を実行します。

```bash
cd app
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

ブラウザで **http://127.0.0.1:8000/** を開いてください。

## 実装内容

- `static/index.html`: 既存 `Yamamoto/room_level/presentation/app_mockup.html` の検索・物件表示・詳細などのインタラクティブ画面を移植
- `main.py`: FastAPIで画面配信と集計データの参照API
- `historical.csv`: `Yamamoto/room_level/summary_tables/24_same_condition_duration.csv` をコピー

## API

- `GET /api/status`: モデルの状態
- `GET /api/historical?layout=1K&age=7&rent_man=7.2`: 同条件の過去実績。例では約27.1%。
- `POST /api/predict`: **未対応 (503)**。学習済みモデルが必要。

## 重要な注意事項

**これはアプリの動く試作版であり、個別物件の機械学習予測が完了した版ではありません。**

- 画面の物件と「高・低」の印はデモ用。研究モデルの推論結果ではない。
- `historical.csv` は研究の過去集計であり、物件ごとの予測ではない。12日掲載された物件の「次の7日間」の実績。
- 掲載終了は成約を意味しない。
- NII/LIFULL由来の個別掲載データ、元TSV、学習済みモデルは含めない。
- 次の実装では `Ohori/src/discrete_model.py` と `Yamamoto/room_level/analysis/17_fixed_window_features.py` の既存モデルを学習・保存してAPIに接続する。

ローカルで起動した後、`/api/status` が `model_ready: false` と返すのは正常です。

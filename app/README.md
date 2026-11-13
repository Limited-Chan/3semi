# すぐ埋まる？ナビ

利用者提供のHTMLデザインをベースにした、FastAPI接続の研究デモです。

## 起動

```bash
cd app
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

http://127.0.0.1:8000/ を開く。

## 研究モデルを使う方法

`Ohori/analysis/07_final_test.py` が学習・出力するモデルを**そのまま使います**。以下のどちらかに、ローカルで生成されたファイルを置いてください。

- `Ohori/results/07_final_test/final_model2_age_layout.joblib`
- `app/models/final_model2_age_layout.joblib`

※ 任意の外部ソースから入手した不審なjoblibファイルを読み込まないでください。ファイル配置後はサーバーを再起動します。

ファイルがある場合：
- `POST /api/predict` で既存 `Ohori/src/discrete_model.py` の `predict_horizon_probability` を使い、物件別に7日以内の掲載終了確率を計算。
- 15%以上＝高、15%未満＝低として画面に表示。

ファイルがない場合：
- `POST /api/predict` は503を返す。
- 画面は `POST /api/historical-risks` へ自動切替。国分寺市の過去実績で、高・低を区分する（**個別物件の学習モデル予測ではない**）。
- `GET /api/historical` で類似条件の過去実績も取得できる。

`GET /api/status` の `model_ready` で、学習済みモデルファイルの存在を確認できます。

## フォルダ

- `static/index.html` — 指定されたオリジナルのレスポンシブHTML。物件一覧・絞り込み・詳細・キープ比較。
- `main.py` — FastAPI、学習済みModel 2の読み込み、研究データによる予測／集計。
- `historical.csv` — `Yamamoto/room_level/summary_tables/24_same_condition_duration.csv` と同じ過去実績40条件。
- `requirements.txt` — アプリ依存ライブラリ。

## 注意

- Web画面に出す11件は架空物件。動作確認用の賃料・駅・間取りなどが設定されています。
- 過去実績による区分は**全物件を掲載12日目**とみなした比較で、個別予測とは異なります。
- 学習済みモデルを使う場合、例示物件の掲載開始月は仮に6月として送信しています。実データ入力機能が整うまでは学習モデルの結果も**デモ上の予測**です。
- 掲載終了は成約を意味しません。
- GitHubには学習コードはありますが、現在のmainブランチにはjoblibバイナリを含めていません。

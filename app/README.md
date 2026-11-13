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


## 気になる物件を分析する（新機能）
ブラウザで `http://127.0.0.1:8000/#/analyze` を開く、または下の「物件を分析」を選択。

- ユーザーが間取り、家賃、築年数、面積、駅、駅徒歩、掲載開始月、掲載からの日数を入力する
- `POST /api/analyze` へ送信
- 学習済み `final_model2_age_layout.joblib` があれば元の離散時間ハザードモデルの予測確率で「問い合わせを優先して検討」等の表示を行う
- モデルがなければ**個別予測を捏造せず**、過去の同条件物件の掲載終了実績を示す。12日目を基準にした集計であり、他の掲載日数への個別予測とは区別する
- アプリ画面には研究の実績に基づいた参考情報と注意事項を併記する

**このアプリは問い合わせ先へ実際に問い合わせを送る機能ではありません。** また、スライドで検証したのは掲載終了であり、実際の成約・問い合わせ率ではありません。問い合わせの優先度は判断材料であって、問い合わせの効果を証明するものではありません。

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

# すぐ埋まる？ナビ — Webアプリ試作版

**画面はユーザーが提供したレスポンシブHTMLをそのまま基準に使用**。研究用の別モック `Yamamoto/room_level/presentation/app_mockup.html` はアプリ画面に使っていません。

## 起動
```bash
cd app
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
http://127.0.0.1:8000/ を開く。

## 構成
- `static/index.html`: ユーザー提供のHTML/CSS/JavaScriptを基本にしたPC・スマホ対応画面。物件一覧、絞り込み、詳細、キープ比較、localStorage。
- `main.py`: FastAPI。HTML配信、研究集計API、モデル状態API。
- `historical.csv`: 既存研究の集計（40条件）。出典 `Yamamoto/room_level/summary_tables/24_same_condition_duration.csv`。
- `requirements.txt`: Python依存関係。

## API
- `GET /api/status`: モデル未配置の状態を返す。
- `GET /api/historical?layout=1K&age=7&rent_man=7.2`: 国分寺市の同条件物件に基づく、掲載12日目からの7日間の実績。物件詳細画面はこのAPIを呼び出す。
- `POST /api/predict`: 未実装（503）。学習済みモデルと元データがGitHubにない。

## 実装済みと未実装を区別
検索、詳細、キープとキープ比較は、提供されたHTMLのJavaScriptを活用して画面上で動作します。画面の物件11件と高低ラベルは**架空のデモデータ**。研究モデルの結果ではありません。

過去集計はCSVの実際の集計値に基づきます。ただし**過去実績と個別物件の予測は別物**。掲載終了は成約とは限りません。元のNII/LIFULL賃貸データや学習済みモデルはリポジトリに含まれていません。

`Ohori/src/discrete_model.py` と `Yamamoto/room_level/analysis/17_fixed_window_features.py` にある研究コードは今後の学習モデル接続に利用できます。現在はモデル推論は動作しません。

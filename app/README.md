# すぐ埋まる？ナビ（研究の実績データ連携デモ）

## 起動
```bash
git pull origin main
cd app
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
http://127.0.0.1:8000/ を開く。

## フォルダ・構成
- `static/index.html`: ユーザーが指定したスマホ/PC対応HTMLがデザインの原本。検索、並び替え、詳細、キープ、比較などをJavaScriptで提供。
- `main.py`: FastAPI。画面配信と掲載終了実績APIを提供。
- `historical.csv`: 国分寺市・2020〜2023年掲載開始の同条件物件集計（40条件）。GitHubの `Yamamoto/room_level/summary_tables/24_same_condition_duration.csv` に由来。
- `requirements.txt`: FastAPI・Uvicornなど。

## 実装済み機能
- 11件のサンプル物件（架空）の検索、絞り込み、並び替え。
- 物件カードと詳細ページ。
- キープと比較（localStorageで保存）。
- 研究の過去実績から、**掲載12日目を基準にした次の7日以内の掲載終了割合**を実際のAPIで取得して表示。
- 上記の実績割合が15%以上なら「高」、未満なら「低」を**APIが自動判定**し、一覧・詳細に表示。
- `POST /api/historical-risks` 一括取得。
- `GET /api/historical?layout=1K&age=7&rent_man=7.2` 単件取得。
- `GET /api/status` 状態確認。

## 重要
これは **過去の同条件の集計からリスクを区分する動作版** です。卒研で使った学習済み離散時間ハザードモデルを読み込んだ個別推論ではありません。元データTSVと学習済みモデルはGitHubにないため、`POST /api/predict`（個別モデル推論）は503を返します。

比較条件を揃えるためデモ11件の掲載日数は12日目に統一しました。割合は過去の集計に基づくもので、各物件の成約・実際の掲載終了を保証しません。掲載終了は成約とは限りません。元データの再配布は行いません。

### モデルへの切り替え
元データ `bukken_rent_v1.0.1.tsv` をローカルで用意し、既存の `Yamamoto/room_level/analysis/17_fixed_window_features.py` と `Ohori/src/discrete_model.py` の条件に合わせて学習・保存したモデルを接続する必要があります。

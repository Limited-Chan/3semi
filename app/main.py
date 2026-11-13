"""すぐ埋まる？ナビ: 発表用 Web デモ。個別予測は学習済みモデルの配置後に有効化。"""
from pathlib import Path
import csv
import logging
import os
import math
import sys
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OHORI = REPO / "Ohori"
MODEL_CANDIDATES = [
    HERE / "models" / "final_model2_age_layout.joblib",
    OHORI / "results" / "07_final_test" / "final_model2_age_layout.joblib",
]
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="すぐ埋まる？ナビ", version="0.1.0")


class Property(BaseModel):
    id: int
    age: float = Field(ge=0)
    rent: float = Field(ge=0, description="家賃・円")
    area: float = Field(gt=0)
    walk_distance: float = Field(ge=0)
    layout: str
    station: str
    start_month: int = Field(ge=1, le=12)
    city: str
    elapsed_day: int = Field(ge=0)


class HistoricalRiskItem(BaseModel):
    id: int
    layout: str
    age: float = Field(ge=0)
    rent_man: float = Field(ge=0)


class HistoricalRiskRequest(BaseModel):
    properties: list[HistoricalRiskItem] = Field(max_length=500)


def _find_historical(layout: str, age: float, rent_man: float):
    bands = ["10年未満", "10〜20年", "20〜30年", "30〜40年", "40年以上"]
    age_band = bands[min(int(age // 10), 4)]
    rent_low = int(rent_man // 2) * 2
    with (HERE / "historical.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return next((r for r in rows if r["間取り"] == layout
                and r["築年数"] == age_band
                and r["家賃(万円)"] == f"{rent_low}〜{rent_low + 2}"), None)




# The local extracted dataset remains on the user's machine; it is never committed
# or uploaded by the application.
DATA_CANDIDATES = [
    REPO / "data" / "prepared" / "kokubunji_raw.parquet",
    OHORI / "prepared" / "kokubunji.csv.gz",
    REPO / "data" / "prepared" / "kokubunji.csv.gz",
]


def _data_source():
    override = os.environ.get("KOKUBUNJI_DATA_PATH", "").strip().strip('"')
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"KOKUBUNJI_DATA_PATH に指定したファイルがありません: {candidate}")
        return candidate.resolve()
    return next((path for path in DATA_CANDIDATES if path.is_file()), None)


@lru_cache(maxsize=2)
def _load_listings_cached(path_str: str, modified_ns: int):
    """Read one local extracted file and reuse the cleaned 2020-2023 data."""
    import pandas as pd
    from src.data import load_prepared
    if str(OHORI) not in sys.path:
        sys.path.insert(0, str(OHORI))
    data = load_prepared(path_str)
    if "addr1_2_name" in data.columns:
        data = data[data["addr1_2_name"].astype(str).str.normalize("NFKC").str.strip() == "国分寺市"]
    data = data[data["start_year"].isin([2020, 2021, 2022, 2023])].copy()
    data = data.reset_index(drop=True)
    data["display_id"] = data.index + 1
    return data


def _local_listings():
    path = _data_source()
    if path is None:
        return None
    return _load_listings_cached(str(path), path.stat().st_mtime_ns)


def _number_or_none(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _listing_record(row):
    def val(key):
        v = row.get(key)
        if v is None:
            return ""
        import pandas as pd
        return "" if pd.isna(v) else str(v)
    return {
        "id": int(row["display_id"]),
        "layout": val("layout"),
        "age": _number_or_none(row.get("age")),
        "rent_man": (None if _number_or_none(row.get("rent")) is None
                     else round(float(row["rent"]) / 10000, 2)),
        "area": _number_or_none(row.get("area")),
        "walk_distance": _number_or_none(row.get("walk_distance")),
        "station": val("station"),
        "start_month": (None if _number_or_none(row.get("start_month")) is None
                        else int(row["start_month"])),
        "start_date": val("pub_start_date"),
        "end_date": val("pub_end_date"),
        "duration": _number_or_none(row.get("duration")),
        "city": val("addr1_2_name"),
    }


@app.get("/api/listings")
def list_local_listings(
    page: int = 1, limit: int = 24, layout: str = "",
    rent_max: float | None = None, age_max: float | None = None,
    query: str = "",
):
    """Paginated 2020-2023 historical records, not currently available listings."""
    if page < 1 or limit < 1 or limit > 100 or (rent_max is not None and rent_max < 0) or (age_max is not None and age_max < 0):
        raise HTTPException(422, "page >= 1, 1 <= limit <= 100, filters >= 0")
    try:
        data = _local_listings()
    except Exception as e:
        logger.exception("Unable to load local data")
        raise HTTPException(422, f"抽出済みデータの読み込みに失敗: {str(e)[:220]}") from e
    if data is None:
        return {"ready": False, "total": 0, "page": page, "limit": limit, "items": [],
                "message": "国分寺市の個別データファイルが未設定です。KOKUBUNJI_DATA_PATH または自動検出先に配置してください。"}
    import pandas as pd
    if layout:
        data = data[data["layout"].astype("string") == layout]
    if rent_max is not None:
        data = data[data["rent"] <= rent_max * 10000]
    if age_max is not None:
        data = data[data["age"] <= age_max]
    if query.strip():
        q = query.strip()
        mask = (data["layout"].astype("string").str.contains(q, case=False, regex=False, na=False) |
                data["station"].astype("string").str.contains(q, case=False, regex=False, na=False))
        data = data[mask]
    total = len(data)
    subset = data.iloc[(page - 1) * limit:page * limit]
    return {"ready": True, "total": total, "page": page, "limit": limit,
            "items": [_listing_record(row) for _, row in subset.iterrows()],
            "message": "2020〜2023年の国分寺市の過去掲載記録（成約・現況ではありません）"}


@app.get("/api/listings/{listing_id}")
def get_local_listing(listing_id: int):
    data = _local_listings()
    if data is None:
        raise HTTPException(404, "個別物件データが未設定です")
    if listing_id < 1 or listing_id > len(data):
        raise HTTPException(404, "物件記録が見つかりません")
    return _listing_record(data.iloc[listing_id - 1])


@app.get("/api/groups")
def list_historical_groups():
    """All available research summary groups, not reconstructed individual listings."""
    with (HERE / "historical.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    result = []
    for i, row in enumerate(rows):
        risk = float(row["12日載った物件が、あと7日以内に終わる割合"])
        result.append({
            "id": i + 1,
            "layout": row["間取り"],
            "age_band": row["築年数"],
            "rent_band": row["家賃(万円)"],
            "count": int(row["件数"]),
            "quarter_days": float(row["25%が終わる日"]),
            "median_days": float(row["半分が終わる日"]),
            "three_quarter_days": float(row["75%が終わる日"]),
            "day12_next7_rate": round(risk, 8),
            "level": "hi" if risk >= 0.15 else "lo",
        })
    return {
        "mode": "historical-12-day",
        "threshold": 0.15,
        "total_groups": len(result),
        "groups": result,
        "notice": "国分寺市の過去物件を条件別にまとめた集計です。現存する個別物件の一覧ではありません。"
    }


@app.post("/api/historical-risks")
def historical_risks(request: HistoricalRiskRequest):
    """実データから、掲載12日目時点の条件別7日掲載終了割合で高低を区分する。

    注意: 研究の学習済み個別予測モデルではない。掲載経過12日目でのみ比較可能。
    """
    result = []
    for p in request.properties:
        row = _find_historical(p.layout, p.age, p.rent_man)
        if row:
            value = float(row["12日載った物件が、あと7日以内に終わる割合"])
            result.append({"id": p.id, "status": "ok", "risk": value,
                           "level": "hi" if value >= 0.15 else "lo",
                           "sample_size": int(row["件数"]),
                           "median_listing_days": float(row["半分が終わる日"])})
        else:
            result.append({"id": p.id, "status": "insufficient_data", "risk": None,
                           "level": "unknown", "sample_size": 0})
    return {"mode": "historical-12-day", "threshold": 0.15, "results": result}


class PredictionRequest(BaseModel):
    properties: list[Property] = Field(max_length=500)



@lru_cache(maxsize=1)
def _loaded_model():
    """Only load trusted local model artifacts; joblib files must not come from untrusted sources."""
    model_path = next((p for p in MODEL_CANDIDATES if p.is_file()), None)
    if model_path is None:
        return None
    import joblib
    return joblib.load(model_path)


def _predict_probabilities(properties):
    import pandas as pd
    if str(OHORI) not in sys.path:
        sys.path.insert(0, str(OHORI))
    from src.discrete_model import predict_horizon_probability

    model = _loaded_model()
    if model is None:
        return None
    frame = pd.DataFrame([{
        "age": p.age, "rent": p.rent, "area": p.area,
        "walk_distance": p.walk_distance, "layout": p.layout,
        "station": p.station, "start_month": p.start_month
    } for p in properties])
    return [float(predict_horizon_probability(
        model, frame.iloc[[i]], landmark=p.elapsed_day, horizon=7,
        use_age_layout_combo=True,
    )[0]) for i, p in enumerate(properties)]


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    import sklearn
    selected = next((p for p in MODEL_CANDIDATES if p.is_file()), None)
    local_source = _data_source() if not os.environ.get("KOKUBUNJI_DATA_PATH") else os.environ.get("KOKUBUNJI_DATA_PATH")
    return {"model_ready": selected is not None, "model_path": str(selected) if selected else None,
            "sklearn_version": sklearn.__version__,
            "local_data_configured": bool(local_source),
            "historical_risk_ready": True, "mode": "model-if-available-else-historical",
            "threshold": 0.15}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        values = _predict_probabilities(request.properties)
    except Exception as error:
        logger.exception("Prediction error (model loading or inference)")
        raise HTTPException(status_code=500, detail=f"モデル推論に失敗: {type(error).__name__}: {str(error)[:240]}") from error
    if values is None:
        raise HTTPException(status_code=503, detail="学習済みModel 2がありません。Ohori/results/07_final_test/final_model2_age_layout.joblib に配置してください。")
    return {
        "mode": "final-model2",
        "threshold": 0.15,
        "results": [{"id": p.id, "risk": value, "level": "hi" if value >= .15 else "lo"}
                    for p, value in zip(request.properties, values)],
    }



class AnalyzeRequest(BaseModel):
    layout: str
    age: float = Field(ge=0, le=150)
    rent_man: float = Field(gt=0, le=300)
    area: float = Field(gt=0, le=1000)
    walk_distance: float = Field(ge=0, le=120)
    station: str
    start_month: int = Field(ge=1, le=12)
    elapsed_day: int = Field(ge=0, le=365)


@app.post("/api/analyze")
def analyze_property(request: AnalyzeRequest):
    """Analyze a user-supplied listing using existing model if locally available.

    If no model: show only past conditional statistics for day 12, not a
    personalized current-day prediction.
    """
    row = _find_historical(request.layout, request.age, request.rent_man)
    history = None
    if row:
        history = {
            "count": int(row["件数"]),
            "median_days_from_start": float(row["半分が終わる日"]),
            "q75_days_from_start": float(row["75%が終わる日"]),
            "day12_next7_rate": round(float(row["12日載った物件が、あと7日以内に終わる割合"]), 6),
            "comparison": "同じ間取り・築年数帯・家賃帯の国分寺市過去物件",
        }
    prop = Property(
        id=1, age=request.age, rent=request.rent_man * 10000,
        area=request.area, walk_distance=request.walk_distance,
        layout=request.layout, station=request.station,
        start_month=request.start_month, city="国分寺市",
        elapsed_day=request.elapsed_day,
    )
    model_error = None
    try:
        values = _predict_probabilities([prop])
    except Exception as error:
        logger.exception("Property analysis model inference error")
        values = None
        model_error = f"{type(error).__name__}: {str(error)[:180]}"

    if values is not None:
        risk = values[0]
        high = risk >= 0.15
        return {
            "mode": "model", "risk": risk, "level": "hi" if high else "lo",
            "heading": "問い合わせを優先して検討" if high else "ほかの物件とも比較して検討",
            "reason": "研究モデルの7日以内の掲載終了予測が15%以上です。" if high
                      else "研究モデルの7日以内の掲載終了予測が15%未満です。",
            "historical": history,
            "caution": "掲載終了は成約ではありません。研究の予測は問い合わせの成果や空室状況を保証しません。",
        }
    return {
        "mode": "historical", "risk": None, "level": None,
        "heading": "過去の実績を参考に検討",
        "reason": (("学習済みモデルの推論でエラーが発生したため、過去実績のみを表示します。"
                    if model_error else "個別予測モデルが未配置です。")
                   + ("掲載12日目時点の過去実績は参照できますが、現在の掲載経過日数に対応した予測ではありません。"
                      if history else "一致する条件の十分な過去実績もありません。")),
        "historical": history,
        "model_error": model_error,
        "caution": "モデルが利用できない場合、過去の条件別割合を個別物件の予測や問い合わせ推奨に置き換えません。",
    }


@app.get("/api/historical")
def historical(layout: str, age: float, rent_man: float):
    """12日間掲載中だった同条件物件の、その後7日以内の過去の掲載終了割合。"""
    if age < 0 or rent_man < 0:
        raise HTTPException(status_code=422, detail="築年数と家賃は0以上")
    row = _find_historical(layout, age, rent_man)
    if not row:
        return {"found": False, "reason": "対象100件未満、または集計対象外"}
    rate = float(row["12日載った物件が、あと7日以内に終わる割合"])
    return {
        "found": True, "source": "Yamamoto/room_level/summary_tables/24_same_condition_duration.csv",
        "not_model_prediction": True, "sample_size": int(row["件数"]),
        "risk_after_12_days": round(rate, 6),
        "reference_tier": "hi" if rate >= .15 else "lo",
        "median_listing_days": float(row["半分が終わる日"]),
        "note": "国分寺市・2020〜2023年掲載開始の過去実績。個別の予測ではありません。",
    }

"""すぐ埋まる？ナビ: 発表用 Web デモ。個別予測は学習済みモデルの配置後に有効化。"""
from pathlib import Path
import csv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
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


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"model_ready": False, "historical_risk_ready": True, "mode": "historical-12-day", "threshold": 0.15}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    raise HTTPException(status_code=503, detail="学習済みモデルが未配置です。デモ画面の高低ラベルは架空の表示です。")


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

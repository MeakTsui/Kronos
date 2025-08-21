import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is in path for `from model import ...`
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
sys.path.append(PROJ_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

BINANCE_BASE = "https://api.binance.com"
INTERVAL_TO_PANDAS = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}


class PredictRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Binance symbol, e.g., BTCUSDT")
    interval: str = Field("5m", description="Kline interval, e.g., 1m,5m,1h")
    lookback: int = Field(400, ge=50, le=1024)
    pred_len: int = Field(15, ge=1, le=256)
    device: str = Field("cpu")
    T: float = Field(1.0, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(0, ge=0, le=1024)
    samples: int = Field(30, ge=1, le=200)
    show_band: bool = Field(True)
    band_low: float = Field(10.0, ge=0.0, le=50.0)
    band_high: float = Field(90.0, ge=50.0, le=100.0)


class PredictResponse(BaseModel):
    symbol: str
    interval: str
    history: List[dict]
    prediction: List[dict]
    band_low: Optional[List[dict]] = None
    band_high: Optional[List[dict]] = None


def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance error: {r.text}")
    data = r.json()
    rows = []
    for k in data:
        open_time_ms = k[0]
        open_, high, low, close = map(float, (k[1], k[2], k[3], k[4]))
        volume = float(k[5])
        amount = ((open_ + high + low + close) / 4.0) * volume
        rows.append(
            {
                "timestamps": pd.to_datetime(open_time_ms, unit="ms", utc=True).tz_convert(None),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        )
    return pd.DataFrame(rows)


def make_future_timestamps(last_ts: pd.Timestamp, interval: str, steps: int) -> pd.Series:
    if interval not in INTERVAL_TO_PANDAS:
        raise HTTPException(status_code=400, detail=f"Unsupported interval: {interval}")
    freq = INTERVAL_TO_PANDAS[interval]
    start = last_ts + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=steps, freq=freq)


# Load models at startup
TOKENIZER_HF = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MODEL_HF = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")

tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_HF)
model = Kronos.from_pretrained(MODEL_HF)
predictor = KronosPredictor(model, tokenizer, device=os.getenv("KRONOS_DEVICE", "cpu"), max_context=512)

app = FastAPI(title="Kronos Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
static_dir = os.path.join(CUR_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Fetch history
    limit = max(req.lookback + 5, req.lookback)
    df = fetch_binance_klines(req.symbol, req.interval, limit)
    if df.shape[0] < req.lookback:
        raise HTTPException(status_code=400, detail="Not enough data from Binance")
    df = df.tail(req.lookback).reset_index(drop=True)

    # 2) Prepare inputs
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df["timestamps"].copy()

    last_ts = df["timestamps"].iloc[-1]
    y_ts = pd.Series(make_future_timestamps(last_ts, req.interval, req.pred_len))

    # 3) Predict
    if req.show_band and req.samples >= 2:
        mean_pred_df, samples = predictor.predict_with_samples(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_ts,
            pred_len=req.pred_len,
            T=req.T,
            top_k=req.top_k,
            top_p=req.top_p,
            sample_count=req.samples,
            verbose=False,
        )
        pred_df = mean_pred_df
        # compute percentiles for close
        close_idx = 3
        low_q = float(req.band_low)
        high_q = float(req.band_high)
        p_low = np.percentile(samples[:, :, close_idx], low_q, axis=0)
        p_high = np.percentile(samples[:, :, close_idx], high_q, axis=0)
        band_low = p_low
        band_high = p_high
    else:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_ts,
            pred_len=req.pred_len,
            T=req.T,
            top_k=req.top_k,
            top_p=req.top_p,
            sample_count=req.samples,
            verbose=False,
        )
        band_low = band_high = None

    # 4) Build response payload
    def candles_payload(ts_index: pd.DatetimeIndex, df_like: pd.DataFrame):
        out = []
        for ts, row in zip(ts_index, df_like.itertuples(index=False)):
            out.append(
                {
                    "time": int(pd.Timestamp(ts).timestamp()),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
            )
        return out

    history = candles_payload(df["timestamps"], df)
    pred = candles_payload(pd.DatetimeIndex(y_ts), pred_df)

    def line_payload(ts_index: pd.DatetimeIndex, values: np.ndarray):
        return [{"time": int(pd.Timestamp(t).timestamp()), "value": float(v)} for t, v in zip(ts_index, values)]

    band_low_payload = band_high_payload = None
    if band_low is not None and band_high is not None:
        y_idx = pd.DatetimeIndex(y_ts)
        band_low_payload = line_payload(y_idx, band_low)
        band_high_payload = line_payload(y_idx, band_high)

    return PredictResponse(
        symbol=req.symbol,
        interval=req.interval,
        history=history,
        prediction=pred,
        band_low=band_low_payload,
        band_high=band_high_payload,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.web.server:app", host="0.0.0.0", port=8000, reload=True)

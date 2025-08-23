import os
import sys
import asyncio
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
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
    device: str = Field("cuda:0")
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


# -------- Probability Matrix API (for heatmap/contour/fan chart) --------
class ProbRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Binance symbol, e.g., BTCUSDT")
    interval: str = Field("5m", description="Kline interval, e.g., 1m,5m,1h")
    lookback: int = Field(400, ge=50, le=1024)
    pred_len: int = Field(15, ge=1, le=256)
    device: str = Field("cuda:0")
    T: float = Field(1.0, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(0, ge=0, le=1024)
    samples: int = Field(200, ge=10, le=2000, description="number of sample paths for distribution")
    bins: int = Field(80, ge=20, le=300, description="price grid size for histogram")
    normalize: str = Field("column", description="normalization method: 'column' or 'global'")
    q_low: float = Field(10.0, ge=0.0, le=50.0)
    q_high: float = Field(90.0, ge=50.0, le=100.0)


class ProbResponse(BaseModel):
    symbol: str
    interval: str
    times: List[int]  # epoch seconds for each future step
    priceGrid: List[float]  # bin centers
    probMatrix: List[List[float]]  # shape [len(priceGrid)][len(times)]
    ridge: List[float]  # argmax price per time
    quantiles: Dict[str, List[float]]  # {"p10":[],"p50":[],"p90":[]}


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


@app.post("/predict_prob", response_model=ProbResponse)
def predict_prob(req: ProbRequest):
    """Return probability matrix over (time, price) based on sampled Close values.
    This endpoint draws N sample paths, builds a price grid, computes per-time
    histogram-based probabilities, finds ridge (argmax) and quantiles.
    """
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

    # 3) Sample predictions
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict sampling error: {e}")

    # samples shape: [S, L, D]; we focus on Close at idx 3
    close_idx = 3
    S, L = samples.shape[0], samples.shape[1]
    closes = samples[:, :, close_idx]  # (S, L)

    # 4) Build price grid based on pooled closes
    pooled = closes.reshape(-1)
    vmin, vmax = float(np.min(pooled)), float(np.max(pooled))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise HTTPException(status_code=500, detail="Invalid sample values")
    if vmin == vmax:
        # Expand a tiny range to avoid degenerate bins
        vmin -= 1e-6
        vmax += 1e-6
    bins = int(req.bins)
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0  # priceGrid

    # 5) Histogram per time (column)
    Z = np.zeros((bins, L), dtype=np.float64)
    for j in range(L):
        col = closes[:, j]
        hist, _ = np.histogram(col, bins=edges)
        Z[:, j] = hist.astype(np.float64)

    # Normalize
    if req.normalize == "column":
        for j in range(L):
            s = Z[:, j].sum()
            if s > 0:
                Z[:, j] /= s
    elif req.normalize == "global":
        s = Z.sum()
        if s > 0:
            Z /= s
    else:
        raise HTTPException(status_code=400, detail="normalize must be 'column' or 'global'")

    # 6) Ridge (argmax) and quantiles
    ridge_idx = np.argmax(Z, axis=0)  # (L,)
    ridge_prices = centers[ridge_idx].astype(float).tolist()

    p10 = np.percentile(closes, 10.0, axis=0).astype(float).tolist()
    p50 = np.percentile(closes, 50.0, axis=0).astype(float).tolist()
    p90 = np.percentile(closes, 90.0, axis=0).astype(float).tolist()

    times = [int(pd.Timestamp(t).timestamp()) for t in pd.DatetimeIndex(y_ts)]

    return ProbResponse(
        symbol=req.symbol,
        interval=req.interval,
        times=times,
        priceGrid=centers.astype(float).tolist(),
        probMatrix=Z.tolist(),
        ridge=ridge_prices,
        quantiles={"p10": p10, "p50": p50, "p90": p90},
    )


# -------------------------- Beam approximation endpoint --------------------------
class ProbBeamRequest(BaseModel):
    symbol: str = Field("BTCUSDT")
    interval: str = Field("5m")
    lookback: int = Field(400, ge=50, le=1024)
    pred_len: int = Field(15, ge=1, le=256)
    device: str = Field("cuda:0")
    T: float = Field(1.0, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k1: int = Field(16, ge=1, le=512, description="top-k for s1 candidates per beam")
    top_k2: int = Field(8, ge=1, le=512, description="top-k for s2 candidates per s1")
    beam_width: int = Field(32, ge=2, le=256)
    bins: int = Field(80, ge=20, le=300)
    normalize: str = Field("column")


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(v[int(len(v) * q / 100.0)])
    cutoff = q / 100.0 * cw[-1]
    idx = np.searchsorted(cw, cutoff, side="left")
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


@app.post("/predict_prob_beam", response_model=ProbResponse)
def predict_prob_beam(req: ProbBeamRequest):
    """Beam-search style approximation of multi-step distribution without pure sampling.
    Maintains top-N token paths per step and builds weighted histograms of Close.
    """
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

    # Build normalized tensors similar to KronosPredictor
    x = x_df.values.astype(np.float32)
    x_time_df = pd.DataFrame({
        'minute': x_timestamp.dt.minute,
        'hour': x_timestamp.dt.hour,
        'weekday': x_timestamp.dt.weekday,
        'day': x_timestamp.dt.day,
        'month': x_timestamp.dt.month,
    }).values.astype(np.float32)
    y_time_df = pd.DataFrame({
        'minute': y_ts.dt.minute,
        'hour': y_ts.dt.hour,
        'weekday': y_ts.dt.weekday,
        'day': y_ts.dt.day,
        'month': y_ts.dt.month,
    }).values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -predictor.clip, predictor.clip)

    device = predictor.device
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
        x_stamp = torch.from_numpy(x_time_df).unsqueeze(0).to(device)
        y_stamp = torch.from_numpy(y_time_df).unsqueeze(0).to(device)

        # Initial tokens from tokenizer
        x_token = predictor.tokenizer.encode(x_tensor, half=True)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):
            if current_seq_len <= predictor.max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = predictor.max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        # Beam entries hold (s1_ids, s2_ids, logprob)
        class Beam:
            __slots__ = ("s1", "s2", "logp")
            def __init__(self, s1, s2, logp):
                self.s1 = s1  # torch.LongTensor [1, t]
                self.s2 = s2
                self.logp = float(logp)

        beams: List[Beam] = [Beam(x_token[0].clone(), x_token[1].clone(), 0.0)]

        # For per-step weighted price collection
        step_values: List[List[Tuple[float, float]]] = [[] for _ in range(req.pred_len)]

        for i in range(req.pred_len):
            # Batch decode_s1 for all beams
            Bn = len(beams)
            # Prepare inputs (truncate to max_context window)
            s1_batch, s2_batch = [], []
            for b in beams:
                cur_len = b.s1.size(1)
                if cur_len <= predictor.max_context:
                    s1_batch.append(b.s1)
                    s2_batch.append(b.s2)
                else:
                    s1_batch.append(b.s1[:, -predictor.max_context:])
                    s2_batch.append(b.s2[:, -predictor.max_context:])
            s1_batch = torch.cat(s1_batch, dim=0)
            s2_batch = torch.cat(s2_batch, dim=0)

            current_seq_len = s1_batch.size(1)
            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)
            # Repeat stamp for Bn
            current_stamp = current_stamp.repeat(Bn, 1, 1)

            s1_logits, context = predictor.model.decode_s1(s1_batch, s2_batch, current_stamp)
            s1_logits_last = s1_logits[:, -1, :] / req.T
            s1_probs = torch.softmax(s1_logits_last, dim=-1)

            # top-k for s1
            k1 = min(req.top_k1, s1_probs.size(-1))
            s1_prob_vals, s1_idx = torch.topk(s1_probs, k1, dim=-1)

            # For each beam element, expand s2 candidates
            children: List[Beam] = []
            for b_ix in range(Bn):
                # context for this beam is context[b_ix:b_ix+1]
                ctx_b = context[b_ix:b_ix+1]
                base_logp = beams[b_ix].logp

                for c_ix in range(k1):
                    s1_id = s1_idx[b_ix, c_ix:c_ix+1]  # [1]
                    p1 = float(s1_prob_vals[b_ix, c_ix].item())
                    # decode_s2 expects the new s1 token (shape [1,1])
                    s1_token = s1_id.view(1, 1)
                    s2_logits = predictor.model.decode_s2(ctx_b, s1_token)
                    s2_logits_last = s2_logits[:, -1, :] / req.T
                    s2_probs = torch.softmax(s2_logits_last, dim=-1)
                    k2 = min(req.top_k2, s2_probs.size(-1))
                    s2_prob_vals, s2_idx = torch.topk(s2_probs, k2, dim=-1)

                    for d_ix in range(k2):
                        s2_id = s2_idx[0, d_ix:d_ix+1]
                        p2 = float(s2_prob_vals[0, d_ix].item())
                        # Append tokens
                        new_s1 = torch.cat([beams[b_ix].s1, s1_token.to(device)], dim=1)
                        new_s2 = torch.cat([beams[b_ix].s2, s2_id.view(1, 1).to(device)], dim=1)
                        logp = base_logp + np.log(max(p1 * p2, 1e-12))
                        children.append(Beam(new_s1, new_s2, logp))

            # Prune to beam_width
            if not children:
                raise HTTPException(status_code=500, detail="Beam search produced no children")
            children.sort(key=lambda b: b.logp, reverse=True)
            beams = children[: int(req.beam_width)]

            # Decode current step price for pruned beams and collect weighted values
            # Batch decode tokenizer for efficiency
            s1_dec = torch.cat([b.s1 for b in beams], dim=0)
            s2_dec = torch.cat([b.s2 for b in beams], dim=0)
            z = predictor.tokenizer.decode([s1_dec, s2_dec], half=True)  # [B, T, D]
            last_close = z[:, -1, 3].detach().cpu().numpy()  # normalized space
            # inverse norm
            close_vals = last_close * (x_std[3] + 1e-5) + x_mean[3]
            weights = np.array([np.exp(b.logp) for b in beams], dtype=np.float64)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            for v, w in zip(close_vals.tolist(), weights.tolist()):
                step_values[i].append((float(v), float(w)))

        # Build global price grid
        pooled_vals = np.array([v for lst in step_values for (v, _) in lst], dtype=np.float64)
        if pooled_vals.size == 0:
            raise HTTPException(status_code=500, detail="No values collected from beam")
        vmin, vmax = float(np.min(pooled_vals)), float(np.max(pooled_vals))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        bins = int(req.bins)
        edges = np.linspace(vmin, vmax, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0

        # Histogram per step with weights
        Z = np.zeros((bins, req.pred_len), dtype=np.float64)
        p10, p50, p90 = [], [], []
        for j in range(req.pred_len):
            if not step_values[j]:
                continue
            vals = np.array([v for (v, w) in step_values[j]], dtype=np.float64)
            ws = np.array([w for (v, w) in step_values[j]], dtype=np.float64)
            hist, _ = np.histogram(vals, bins=edges, weights=ws)
            Z[:, j] = hist
            # weighted quantiles
            p10.append(_weighted_percentile(vals, ws, 10.0))
            p50.append(_weighted_percentile(vals, ws, 50.0))
            p90.append(_weighted_percentile(vals, ws, 90.0))

        # Normalize
        if req.normalize == "column":
            for j in range(req.pred_len):
                s = Z[:, j].sum()
                if s > 0:
                    Z[:, j] /= s
        elif req.normalize == "global":
            s = Z.sum()
            if s > 0:
                Z /= s
        else:
            raise HTTPException(status_code=400, detail="normalize must be 'column' or 'global'")

        ridge_idx = np.argmax(Z, axis=0)
        ridge_prices = centers[ridge_idx].astype(float).tolist()

        times = [int(pd.Timestamp(t).timestamp()) for t in pd.DatetimeIndex(y_ts)]

        return ProbResponse(
            symbol=req.symbol,
            interval=req.interval,
            times=times,
            priceGrid=centers.astype(float).tolist(),
            probMatrix=Z.tolist(),
            ridge=ridge_prices,
            quantiles={"p10": p10, "p50": p50, "p90": p90},
        )

# ---------------------- Scheduler for periodic prediction push ----------------------

# Remote service config (e.g., Cloudflare Workers)
CRICKET_API_BASE = os.getenv("CRICKET_API_BASE", "https://app.cricket-ai.xyz")
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY", "")  # Bearer token for auth

# Scheduler controls
SCHED_ENABLED = os.getenv("SCHED_ENABLED", "false").lower() == "true"
SCHED_INTERVAL_SEC = int(os.getenv("SCHED_INTERVAL_SEC", "180"))

# Default prediction params when remote job omits fields
DEFAULT_LOOKBACK = int(os.getenv("DEFAULT_LOOKBACK", "400"))
DEFAULT_PRED_LEN = int(os.getenv("DEFAULT_PRED_LEN", "15"))
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "5m")
DEFAULT_SAMPLES = int(os.getenv("DEFAULT_SAMPLES", "30"))
DEFAULT_SHOW_BAND = os.getenv("DEFAULT_SHOW_BAND", "true").lower() == "true"
DEFAULT_BAND_LOW = float(os.getenv("DEFAULT_BAND_LOW", "10"))
DEFAULT_BAND_HIGH = float(os.getenv("DEFAULT_BAND_HIGH", "90"))


def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if CRICKET_API_KEY:
        h["Authorization"] = f"Bearer {CRICKET_API_KEY}"
    return h


def _fetch_symbols_to_predict() -> List[Dict[str, Any]]:
    url = f"{CRICKET_API_BASE}/api/symbols"
    try:
        r = requests.get(url, headers=_auth_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[scheduler] fetch symbols error: {e}")
        return []


def _post_prediction(payload: Dict[str, Any]) -> None:
    url = f"{CRICKET_API_BASE}/api/predictions"
    try:
        r = requests.post(url, headers=_auth_headers(), json=payload, timeout=20)
        if r.status_code >= 300:
            print(f"[scheduler] push prediction failed {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[scheduler] push prediction error: {e}")


def _group_jobs(jobs: List[Dict[str, Any]]) -> Dict[Tuple[str, int, int, int, bool, float, float], List[Dict[str, Any]]]:
    """Group by interval, lookback, pred_len, samples, show_band, band_low, band_high"""
    groups: Dict[Tuple[str, int, int, int, bool, float, float], List[Dict[str, Any]]] = {}
    for j in jobs:
        interval = j.get("interval", DEFAULT_INTERVAL)
        lookback = int(j.get("lookback", DEFAULT_LOOKBACK))
        pred_len = int(j.get("pred_len", DEFAULT_PRED_LEN))
        samples = int(j.get("samples", DEFAULT_SAMPLES))
        show_band = bool(j.get("show_band", DEFAULT_SHOW_BAND))
        band_low = float(j.get("band_low", DEFAULT_BAND_LOW))
        band_high = float(j.get("band_high", DEFAULT_BAND_HIGH))
        key = (interval, lookback, pred_len, samples, show_band, band_low, band_high)
        groups.setdefault(key, []).append(j)
    return groups


async def _scheduler_loop():
    if not SCHED_ENABLED:
        print("[scheduler] disabled (SCHED_ENABLED=false)")
        return
    print(f"[scheduler] started, every {SCHED_INTERVAL_SEC}s, target={CRICKET_API_BASE}")
    while True:
        try:
            jobs = _fetch_symbols_to_predict()
            if not jobs:
                await asyncio.sleep(SCHED_INTERVAL_SEC)
                continue

            groups = _group_jobs(jobs)
            for (interval, lookback, pred_len, samples, show_band, band_low, band_high), items in groups.items():
                # Prepare per-symbol data
                dfs: List[pd.DataFrame] = []
                x_ts_list: List[pd.Series] = []
                y_ts_list: List[pd.Series] = []
                symbols: List[str] = []

                for job in items:
                    symbol = str(job.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    try:
                        limit = max(lookback + 5, lookback)
                        df = fetch_binance_klines(symbol, interval, limit)
                        if df.shape[0] < lookback:
                            print(f"[scheduler] insufficient data for {symbol}-{interval}")
                            continue
                        df = df.tail(lookback).reset_index(drop=True)
                        last_ts = df["timestamps"].iloc[-1]
                        y_ts = pd.Series(make_future_timestamps(last_ts, interval, pred_len))
                        x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
                        dfs.append(x_df)
                        x_ts_list.append(df["timestamps"].copy())
                        y_ts_list.append(y_ts)
                        symbols.append(symbol)
                    except Exception as e:
                        print(f"[scheduler] fetch/prep error for {symbol}: {e}")

                if not dfs:
                    continue

                # Predict in batch
                try:
                    if show_band and samples >= 2:
                        mean_dfs, samples_arr = predictor.predict_with_samples_batch(
                            dfs=dfs,
                            x_timestamps=x_ts_list,
                            y_timestamps=y_ts_list,
                            pred_len=pred_len,
                            T=1.0,
                            top_k=0,
                            top_p=0.9,
                            sample_count=samples,
                            verbose=False,
                        )
                    else:
                        mean_dfs = predictor.predict_batch(
                            dfs=dfs,
                            x_timestamps=x_ts_list,
                            y_timestamps=y_ts_list,
                            pred_len=pred_len,
                            T=1.0,
                            top_k=0,
                            top_p=0.9,
                            sample_count=samples,
                            verbose=False,
                        )
                        samples_arr = None
                except Exception as e:
                    print(f"[scheduler] predict batch error: {e}")
                    continue

                # Build and push per-symbol payloads
                def to_epoch(ts):
                    return int(pd.Timestamp(ts).timestamp())

                for idx, symbol in enumerate(symbols):
                    df_hist = dfs[idx]
                    x_ts = x_ts_list[idx]
                    pred_df = mean_dfs[idx]
                    y_ts = y_ts_list[idx]

                    history = [
                        {
                            "time": to_epoch(t),
                            "open": float(r.open),
                            "high": float(r.high),
                            "low": float(r.low),
                            "close": float(r.close),
                            "volume": float(r.volume),
                        }
                        for t, r in zip(x_ts, df_hist.itertuples(index=False))
                    ]

                    prediction = [
                        {
                            "time": to_epoch(t),
                            "open": float(r.open),
                            "high": float(r.high),
                            "low": float(r.low),
                            "close": float(r.close),
                            "volume": float(r.volume),
                        }
                        for t, r in zip(pd.DatetimeIndex(y_ts), pred_df.itertuples(index=False))
                    ]

                    band_low_payload = band_high_payload = None
                    if samples_arr is not None:
                        close_idx = 3
                        low_vals = np.percentile(samples_arr[idx, :, :, close_idx], band_low, axis=0)
                        high_vals = np.percentile(samples_arr[idx, :, :, close_idx], band_high, axis=0)
                        band_low_payload = [
                            {"time": to_epoch(t), "value": float(v)}
                            for t, v in zip(pd.DatetimeIndex(y_ts), low_vals)
                        ]
                        band_high_payload = [
                            {"time": to_epoch(t), "value": float(v)}
                            for t, v in zip(pd.DatetimeIndex(y_ts), high_vals)
                        ]

                    payload = {
                        "symbol": symbol,
                        "interval": interval,
                        "lookback": lookback,
                        "pred_len": pred_len,
                        "samples": samples,
                        "created_at": int(pd.Timestamp.utcnow().timestamp()),
                        "history": history,
                        "prediction": prediction,
                        "band_low": band_low_payload,
                        "band_high": band_high_payload,
                    }
                    _post_prediction(payload)

        except Exception as e:
            print(f"[scheduler] loop error: {e}")

        await asyncio.sleep(SCHED_INTERVAL_SEC)


@app.on_event("startup")
async def _start_scheduler():
    # Fire-and-forget task
    asyncio.create_task(_scheduler_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.web.server:app", host="0.0.0.0", port=8000, reload=True)

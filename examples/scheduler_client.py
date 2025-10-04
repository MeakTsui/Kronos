#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone scheduler client.
- Runs on a separate machine from the FastAPI server.
- Pulls symbols from cloud API.
- Fetches Binance history locally for each symbol (for payload history only).
- Calls remote server's /explain_predict_path to get predictions.
- Pushes {history, predictions} to cloud.

Environment variables:
- SERVER_BASE: base URL of the FastAPI server providing /explain_predict_path (default: http://127.0.0.1:8000)
- CRICKET_API_BASE: cloud API base to push predictions (default: https://app.cricket-ai.xyz)
- CRICKET_API_KEY: bearer token for both pulling symbols and pushing predictions
- DEFAULT_INTERVAL: e.g., 5m
- DEFAULT_LOOKBACK: e.g., 400
- DEFAULT_PRED_LEN: e.g., 15
- DEFAULT_SAMPLES: e.g., 30 (not required by explain but kept for metadata)
- EXPLAIN_BINS: e.g., 128
- EXPLAIN_NORMALIZE: column|global
- EXPLAIN_TOP_K1: e.g., 8
- EXPLAIN_TOP_K2: e.g., 8
- EXPLAIN_SAMPLE_PATH: true|false
- EXPLAIN_INCLUDE_MODE: true|false
- EXPLAIN_INCLUDE_EXPECTED: true|false
- EXPLAIN_T: float, default 1.0
- EXPLAIN_TOP_P: float, default 0.9
- EXPLAIN_TOP_K: int, default 0
- BATCH_SIZE: int, default 4 (process symbols in mini-batches)
- TIMEOUT_SEC: int, default 15

CLI overrides are available; run with -h for help.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime

# -------------------------- Config helpers --------------------------

def _get_env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _auth_headers_cloud() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = os.getenv("CRICKET_API_KEY", "Nbb@123")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


# -------------------------- Binance utils --------------------------

BINANCE_API = "https://api.binance.com/api/v3/klines"

_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


def fetch_binance_klines(symbol: str, interval: str, limit: int, timeout: int = 15) -> List[Dict[str, Any]]:
    if interval not in _INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit),
    }
    r = requests.get(BINANCE_API, params=params, timeout=timeout)
    r.raise_for_status()
    arr = r.json()
    out: List[Dict[str, Any]] = []
    for it in arr:
        # https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        open_time = int(it[0]) // 1000
        o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
        v = float(it[5])
        # quote asset volume in it[7], not required for history payload
        out.append({
            "time": open_time,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
    return out


# -------------------------- Cloud symbol list --------------------------

def pull_symbols() -> List[Dict[str, Any]]:
    base = os.getenv("CRICKET_API_BASE", "https://app.cricket-ai.xyz")
    url = f"{base}/api/v1/ai/symbols"
    try:
        r = requests.get(url, headers=_auth_headers_cloud(), timeout=int(os.getenv("TIMEOUT_SEC", "15")))
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[client] fetch symbols error: {e}")
        return []


# -------------------------- Explain call --------------------------

def call_explain(server_base: str, payload: Dict[str, Any], timeout_sec: int = 20) -> Dict[str, Any]:
    url = server_base.rstrip("/") + "/explain_predict_path"
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()


# -------------------------- Push to cloud --------------------------

def push_prediction(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # base = os.getenv("CRICKET_API_BASE", "https://app.cricket-ai.xyz")
    base = os.getenv("CRICKET_API_BASE", "http://localhost:8787")
    url = f"{base}/api/v1/ai/predictions"
    try:
        r = requests.post(url, headers=_auth_headers_cloud(), json=payload, timeout=int(os.getenv("TIMEOUT_SEC", "15")))
        if r.status_code >= 300:
            return False, f"{r.status_code}: {r.text}"
        return True, None
    except Exception as e:
        return False, str(e)


def push_prediction_r2(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Upload the prediction payload directly to Cloudflare R2 (S3-compatible).

    Required environment variables:
    - R2_ACCESS_KEY_ID
    - R2_SECRET_ACCESS_KEY
    - R2_ACCOUNT_ID
    - R2_BUCKET

    Optional environment variables:
    - R2_ENDPOINT (default: https://{ACCOUNT_ID}.r2.cloudflarestorage.com)
    - R2_REGION (default: auto)
    - R2_PREFIX (object key prefix, default: predictions)

    Returns (ok, error_message_or_None)
    """
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")
    region = os.getenv("R2_REGION", "auto")
    endpoint = os.getenv("R2_ENDPOINT") or (f"https://{account_id}.r2.cloudflarestorage.com" if account_id else None)
    prefix = os.getenv("R2_PREFIX", "predictions")

    if not all([account_id, access_key, secret_key, bucket]):
        return False, "missing one of R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET"
    if not endpoint:
        return False, "R2_ENDPOINT could not be derived; set R2_ACCOUNT_ID or R2_ENDPOINT"

    # Build an S3 client for Cloudflare R2
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
    except Exception as e:
        return False, f"boto3 client init error: {e}"

    # Compose object keys
    symbol = str(payload.get("symbol", "unknown")).upper()
    interval = str(payload.get("interval", "unknown"))
    created_at = int(payload.get("created_at", int(time.time())))
    ts = datetime.utcfromtimestamp(created_at).strftime('%Y%m%d_%H%M%S')
    key_dated = f"{prefix}/{symbol}/{interval}/{ts}.json"
    key_latest = f"{prefix}/{symbol}/{interval}/latest.json"

    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # Upload dated file
        s3.put_object(Bucket=bucket, Key=key_dated, Body=body, ContentType="application/json")
        # Upload/update latest pointer
        s3.put_object(Bucket=bucket, Key=key_latest, Body=body, ContentType="application/json")
        return True, None
    except (ClientError, BotoCoreError) as e:
        return False, f"r2 upload error: {e}"
    except Exception as e:
        return False, str(e)


# -------------------------- Main run-once logic --------------------------

def run_once(args: argparse.Namespace) -> List[Dict[str, Any]]:
    server_base = args.server_base
    interval = args.interval
    lookback = args.lookback
    pred_len = args.pred_len
    samples = args.samples

    # explain params
    bins = args.bins
    normalize = args.normalize
    top_k1 = args.top_k1
    top_k2 = args.top_k2
    sample_path = args.sample_path
    include_mode = args.include_mode
    include_expected = args.include_expected
    T = args.T
    top_p = args.top_p
    top_k = args.top_k

    timeout_sec = args.timeout
    batch_size = args.batch_size

    jobs = pull_symbols()
    if not jobs:
        print("[client] no symbols to process.")
        return []

    def to_created_at() -> int:
        return int(time.time())

    results: List[Dict[str, Any]] = []

    # process in mini-batches
    N = len(jobs)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = jobs[start:end]
        print(f"[client] batch {start}-{end} of {N}")

        for job in batch:
            sym = str(job.get("symbol", "")).upper()
            if not sym:
                continue
            t0 = time.time()
            ok = True
            err = None
            try:
                # 1) history for payload only
                limit = max(lookback + 5, lookback)
                hist = fetch_binance_klines(sym, interval, limit, timeout=timeout_sec)
                if len(hist) < lookback:
                    raise RuntimeError("insufficient binance data")
                # cut to lookback tail
                hist = hist[-lookback:]

                # 2) call server explain
                explain_body = {
                    "symbol": sym,
                    "interval": interval,
                    "lookback": lookback,
                    "pred_len": pred_len,
                    "T": T,
                    "top_p": top_p,
                    "top_k": top_k,
                    "bins": bins,
                    "normalize": normalize,
                    "top_k1": top_k1,
                    "top_k2": top_k2,
                    "sample_path": sample_path,
                    "include_mode": include_mode,
                    "include_expected": include_expected,
                }
                explain_resp = call_explain(server_base, explain_body, timeout_sec)

                # 3) push to cloud
                payload = {
                    "symbol": sym,
                    "interval": interval,
                    "lookback": lookback,
                    "pred_len": pred_len,
                    "samples": samples,
                    "created_at": to_created_at(),
                    "history": hist,
                    "prediction": json.dumps(explain_resp),
                }
                ok, err = push_prediction(payload)
            except Exception as e:
                ok = False
                err = str(e)

            elapsed = time.time() - t0
            result = {"symbol": sym, "ok": ok, "error": err, "elapsed_sec": round(elapsed, 3)}
            results.append(result)
            status = "OK" if ok else f"FAIL: {err}"
            print(f"[client] {sym} => {status} ({elapsed:.2f}s)")

        # small pause between batches to be polite
        time.sleep(0.5)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone scheduler client for explain predictions push")
    # p.add_argument("--server-base", dest="server_base", default=os.getenv("SERVER_BASE", "http://127.0.0.1:8000"))
    p.add_argument("--server-base", dest="server_base", default=os.getenv("SERVER_BASE", "http://192.168.50.200:8000"))
    p.add_argument("--interval", default=os.getenv("DEFAULT_INTERVAL", "5m"))
    p.add_argument("--lookback", type=int, default=int(os.getenv("DEFAULT_LOOKBACK", "512")))
    p.add_argument("--pred-len", dest="pred_len", type=int, default=int(os.getenv("DEFAULT_PRED_LEN", "16")))
    p.add_argument("--samples", type=int, default=int(os.getenv("DEFAULT_SAMPLES", "1")))

    # explain params
    p.add_argument("--bins", type=int, default=int(os.getenv("EXPLAIN_BINS", "32")))
    p.add_argument("--normalize", choices=["column", "global"], default=os.getenv("EXPLAIN_NORMALIZE", "column"))
    p.add_argument("--top-k1", dest="top_k1", type=int, default=int(os.getenv("EXPLAIN_TOP_K1", "8")))
    p.add_argument("--top-k2", dest="top_k2", type=int, default=int(os.getenv("EXPLAIN_TOP_K2", "8")))
    p.add_argument("--sample-path", dest="sample_path", default=True, action="store_true" if _get_env_bool("EXPLAIN_SAMPLE_PATH", True) else "store_false")
    p.add_argument("--include-mode", dest="include_mode", default=True, action="store_true" if _get_env_bool("EXPLAIN_INCLUDE_MODE", True) else "store_false")
    p.add_argument("--include-expected", dest="include_expected", default=True, action="store_true" if _get_env_bool("EXPLAIN_INCLUDE_EXPECTED", True) else "store_false")
    p.add_argument("--T", type=float, default=float(os.getenv("EXPLAIN_T", "1.0")))
    p.add_argument("--top-p", dest="top_p", type=float, default=float(os.getenv("EXPLAIN_TOP_P", "0.9")))
    p.add_argument("--top-k", dest="top_k", type=int, default=int(os.getenv("EXPLAIN_TOP_K", "0")))

    p.add_argument("--batch-size", dest="batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "4")))
    p.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT_SEC", "30")))

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_once(args)
    # Pretty print summary
    print("\n[client] summary:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    # exit code: 0 if all ok, 1 if any failed
    if any((not r.get("ok", False)) for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

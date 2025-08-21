import argparse
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import mplfinance as mpf

# Ensure project root is in path for `from model import ...`
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


BINANCE_BASE = "https://api.binance.com"

# Map Binance interval to pandas offset alias
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


def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch recent klines from Binance REST API.

    Returns a DataFrame with columns: ['timestamps','open','high','low','close','volume','amount']
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Kline array fields per Binance docs:
    # [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, trades,
    #   takerBuyBase, takerBuyQuote, ignore ]
    rows = []
    for k in data:
        open_time_ms = k[0]
        open_, high, low, close = map(float, (k[1], k[2], k[3], k[4]))
        volume = float(k[5])
        # Use average price * volume as amount proxy
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
    df = pd.DataFrame(rows)
    return df


def make_future_timestamps(last_ts: pd.Timestamp, interval: str, steps: int) -> pd.Series:
    if interval not in INTERVAL_TO_PANDAS:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_TO_PANDAS.keys())}")
    freq = INTERVAL_TO_PANDAS[interval]
    start = last_ts + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=steps, freq=freq)


def plot_prediction_line(kline_df: pd.DataFrame, pred_df: pd.DataFrame, symbol: str, interval: str, band: Tuple[pd.Series, pd.Series] | None = None):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]

    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(close_df["Prediction"], label="Prediction (mean)", color="red", linewidth=1.5)
    # Uncertainty band if provided
    if band is not None:
        lower, upper = band
        ax1.fill_between(lower.index, lower.values, upper.values, color="red", alpha=0.15, label="Uncertainty (P10-P90)")
    ax1.set_ylabel("Close", fontsize=12)
    ax1.set_title(f"{symbol} {interval} Next Prediction", fontsize=13)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True)

    ax2.plot(volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.2)
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.2)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True)

    # Avoid tight_layout with mplfinance axes; adjust manually to prevent warnings
    try:
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.12, hspace=0.2)
    except Exception:
        pass
    plt.show()


def plot_prediction_candle(kline_df: pd.DataFrame, pred_len: int, symbol: str, interval: str, pad_steps: int = 0):
    """Plot candlesticks with a vertical separator at prediction start and volume panel."""
    df_plot = kline_df.copy()
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)

    # mplfinance expects specific column names
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df_plot = df_plot.rename(columns=rename_map)

    # Determine prediction start timestamp
    pred_start_idx = len(df_plot) - pred_len
    pred_start_ts = df_plot.index[pred_start_idx]

    # Add right-side padding to avoid candles squeezed at left
    # infer frequency from index
    try:
        inferred = pd.infer_freq(df_plot.index)
        if inferred is None:
            # Fallback to interval mapping
            inferred = INTERVAL_TO_PANDAS.get(interval, '1min')
        freq = pd.tseries.frequencies.to_offset(inferred)
    except Exception:
        freq = pd.tseries.frequencies.to_offset(INTERVAL_TO_PANDAS.get(interval, '1min'))

    if pad_steps > 0:
        last_ts = df_plot.index[-1]
        pad_index = pd.date_range(start=last_ts + freq, periods=pad_steps, freq=freq)
        pad_df = pd.DataFrame(index=pad_index, columns=df_plot.columns, dtype=float)
        df_plot = pd.concat([df_plot, pad_df])

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        volume=True,
        figsize=(10, 6),
        returnfig=True,
        title=f"{symbol} {interval} Next Prediction (Candles)",
        style='yahoo',
    )

    # Add a vertical line where prediction starts
    ax_price = axes[0]
    ax_price.axvline(pred_start_ts, color='orange', linestyle='--', linewidth=1.2, label='Prediction Start')
    # Shade the prediction region (from start to the last index, excluding padding part for better clarity)
    try:
        last_real_ts = df_plot.index[-1 - pad_steps] if pad_steps > 0 else df_plot.index[-1]
        ax_price.axvspan(pred_start_ts, last_real_ts, color='orange', alpha=0.08)
    except Exception:
        pass
    ax_price.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict next K-lines for a Binance symbol using Kronos")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Binance trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--interval", type=str, default="5m", help="Binance kline interval, e.g., 1m")
    parser.add_argument("--lookback", type=int, default=400, help="Number of past bars as context (<= 512 recommended)")
    parser.add_argument("--pred_len", type=int, default=15, help="Number of future steps to predict")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device, e.g., cuda:0 or cpu")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base", help="HF path of tokenizer")
    parser.add_argument("--model", type=str, default="NeoQuasar/Kronos-small", help="HF path of predictor model")
    parser.add_argument("--max_context", type=int, default=512, help="Model max context")
    parser.add_argument("--T", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 to disable)")
    parser.add_argument("--samples", type=int, default=1, help="Sample count for ensemble")
    parser.add_argument("--plot_style", type=str, default="line", choices=["line", "candle"], help="Plot style: line or candle")
    parser.add_argument("--pad_ratio", type=float, default=0.3, help="Right padding as ratio of pred_len for candle plot")
    parser.add_argument("--pad_min_steps", type=int, default=3, help="Minimum right padding steps for candle plot")
    args = parser.parse_args()

    # 1) Fetch recent klines
    limit = max(args.lookback + 5, args.lookback)  # small buffer to ensure enough rows
    df = fetch_binance_klines(args.symbol, args.interval, limit)
    if df.shape[0] < args.lookback:
        raise RuntimeError(f"Fetched rows {df.shape[0]} < lookback {args.lookback}. Try increasing limit or check symbol/interval.")

    # Align to exactly lookback rows
    df = df.tail(args.lookback).reset_index(drop=True)

    # 2) Prepare inputs
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df["timestamps"].copy()

    # 3) Construct future timestamps for prediction horizon
    last_ts = df["timestamps"].iloc[-1]
    # predictor.predict expects a pandas Series with .dt accessor
    y_timestamp_series = pd.Series(make_future_timestamps(last_ts, args.interval, args.pred_len))

    # 4) Load model and predictor
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=args.max_context)

    # 5) Predict
    if args.plot_style == "line":
        # For line plot, get samples for uncertainty band
        sample_count = max(args.samples, 10)  # ensure enough samples for a stable band
        mean_pred_df, samples = predictor.predict_with_samples(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp_series,
            pred_len=args.pred_len,
            T=args.T,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=sample_count,
            verbose=True,
        )
        pred_df = mean_pred_df
    else:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp_series,
            pred_len=args.pred_len,
            T=args.T,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.samples,
            verbose=True,
        )

    # 6) Combine for visualization: last lookback + future horizon
    # Use a DatetimeIndex for predicted part when concatenating
    y_idx = pd.DatetimeIndex(y_timestamp_series)
    kline_df = pd.concat([
        df.set_index("timestamps"),
        pred_df.set_index(y_idx),
    ])

    # 7) Plot
    if args.plot_style == "candle":
        pad_steps = max(args.pad_min_steps, int(round(args.pred_len * args.pad_ratio)))
        plot_prediction_candle(kline_df, args.pred_len, args.symbol, args.interval, pad_steps=pad_steps)
    else:
        # Build uncertainty band from samples (close price P10-P90)
        # samples shape: [S, pred_len, D], column order: [open, high, low, close, volume, amount]
        close_idx = 3
        y_idx = pd.DatetimeIndex(y_timestamp_series)
        p10 = np.percentile(samples[:, :, close_idx], 10, axis=0)
        p90 = np.percentile(samples[:, :, close_idx], 90, axis=0)
        band = (pd.Series(p10, index=y_idx), pd.Series(p90, index=y_idx))
        plot_prediction_line(kline_df, pred_df, args.symbol, args.interval, band=band)

    # 8) Print head
    print("Forecasted Data Head:")
    print(pred_df.head())


if __name__ == "__main__":
    main()

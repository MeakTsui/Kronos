const el = (id) => document.getElementById(id);

if (typeof window.LightweightCharts === 'undefined') {
  throw new Error('LightweightCharts not loaded. Ensure the standalone build script tag is before app.js.');
}

const chartContainer = el('chart');
const chart = LightweightCharts.createChart(chartContainer, {
  layout: { textColor: '#222', background: { type: 'Solid', color: '#fff' } },
  rightPriceScale: { borderVisible: true },
  timeScale: { borderVisible: true },
  grid: { vertLines: { color: '#eee' }, horzLines: { color: '#eee' } },
});

if (typeof chart.addCandlestickSeries !== 'function') {
  console.error('Chart object:', chart);
  throw new Error('addCandlestickSeries is not a function. Check Lightweight Charts version (use v4 standalone).');
}

const candleSeries = chart.addCandlestickSeries();
const predLineSeries = chart.addLineSeries({ color: 'red', lineWidth: 2 });
const bandLowSeries = chart.addLineSeries({ color: 'rgba(255,0,0,0.6)', lineWidth: 1, lineStyle: 2 });
const bandHighSeries = chart.addLineSeries({ color: 'rgba(255,0,0,0.6)', lineWidth: 1, lineStyle: 2 });

async function runPrediction() {
  const payload = {
    symbol: el('symbol').value.trim().toUpperCase(),
    interval: el('interval').value,
    lookback: parseInt(el('lookback').value, 10),
    pred_len: parseInt(el('pred_len').value, 10),
    samples: parseInt(el('samples').value, 10),
    T: parseFloat(el('T').value),
    top_p: parseFloat(el('top_p').value),
    top_k: parseInt(el('top_k').value, 10),
    show_band: el('show_band').checked,
    band_low: parseFloat(el('band_low').value),
    band_high: parseFloat(el('band_high').value),
  };

  el('run').disabled = true;
  el('run').textContent = 'Running...';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    const data = await res.json();

    // Render candles (history + predicted candles)
    const candles = [...data.history, ...data.prediction];
    candleSeries.setData(candles);

    // Render predicted mean close line
    const predLine = data.prediction.map(c => ({ time: c.time, value: c.close }));
    predLineSeries.setData(predLine);

    // Render bands if provided
    if (payload.show_band && data.band_low && data.band_high) {
      bandLowSeries.setData(data.band_low);
      bandHighSeries.setData(data.band_high);
    } else {
      bandLowSeries.setData([]);
      bandHighSeries.setData([]);
    }

    chart.timeScale().fitContent();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    el('run').disabled = false;
    el('run').textContent = 'Run Prediction';
  }
}

el('run').addEventListener('click', runPrediction);

// Initial run
runPrediction();

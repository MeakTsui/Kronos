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

function epochToDateStr(sec){
  const d = new Date(sec * 1000);
  return d.toISOString().replace('T',' ').slice(0,19);
}

function renderHeatmap(prob){
  // prob: { times:number[], priceGrid:number[], probMatrix:number[][], ridge:number[] }
  const heat = {
    x: prob.times.map((t, i) => i), // use index for better spacing; we show time in hover
    y: prob.priceGrid,
    z: prob.probMatrix,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: { title: 'Prob' },
    hovertemplate: 'time=%{customdata}<br>price=%{y}<br>prob=%{z:.4f}<extra></extra>',
    customdata: prob.times.map(epochToDateStr),
  };
  const ridge = {
    x: prob.times.map((_, i) => i),
    y: prob.ridge,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'white', width: 2 },
    name: 'Ridge',
    hovertemplate: 'time=%{text}<br>ridge=%{y:.4f}<extra></extra>',
    text: prob.times.map(epochToDateStr),
  };
  Plotly.newPlot('heatmap', [heat, ridge], {
    margin: {l:40,r:10,t:10,b:30},
    xaxis: { title: 'Step', zeroline: false },
    yaxis: { title: 'Price', zeroline: false },
  }, {responsive:true});
}

function renderContour(prob){
  const contour = {
    x: prob.times.map((_, i) => i),
    y: prob.priceGrid,
    z: prob.probMatrix,
    type: 'contour',
    colorscale: 'Viridis',
    contours: { coloring: 'heatmap' },
    colorbar: { title: 'Prob' },
  };
  const ridge = {
    x: prob.times.map((_, i) => i),
    y: prob.ridge,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'white', width: 2 },
    name: 'Ridge',
  };
  Plotly.newPlot('contour', [contour, ridge], {
    margin: {l:40,r:10,t:10,b:30},
    xaxis: { title: 'Step', zeroline: false },
    yaxis: { title: 'Price', zeroline: false },
  }, {responsive:true});
}

function renderFan(prob){
  const x = prob.times;
  const mid = prob.quantiles.p50.map((v,i)=>({ x: i, y: v }));
  const p10 = prob.quantiles.p10.map((v,i)=>({ x: i, y: v }));
  const p90 = prob.quantiles.p90.map((v,i)=>({ x: i, y: v }));

  const midTrace = { x: mid.map(d=>d.x), y: mid.map(d=>d.y), type:'scatter', mode:'lines', name:'Median', line:{color:'#224488', width:2} };
  const p10Trace = { x: p10.map(d=>d.x), y: p10.map(d=>d.y), type:'scatter', mode:'lines', name:'p10', line:{color:'orange', width:1, dash:'dot'} };
  const p90Trace = { x: p90.map(d=>d.x), y: p90.map(d=>d.y), type:'scatter', mode:'lines', name:'p90', line:{color:'orange', width:1, dash:'dot'}, fill:'tonexty', fillcolor:'rgba(255,165,0,0.15)' };

  Plotly.newPlot('fan', [p10Trace, p90Trace, midTrace], {
    margin: {l:40,r:10,t:10,b:30},
    xaxis: { title: 'Step' },
    yaxis: { title: 'Close' },
    showlegend: true,
    legend: { orientation: 'h' }
  }, {responsive:true});
}

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

    // Fetch probability matrix for heatmap/contour/fan
    try {
      const mode = el('prob_mode').value;
      let url = '/predict_prob';
      let body = {
        symbol: payload.symbol,
        interval: payload.interval,
        lookback: payload.lookback,
        pred_len: payload.pred_len,
        T: payload.T,
        top_p: payload.top_p,
        bins: 80,
        normalize: 'column'
      };
      if (mode === 'sampling') {
        body.samples = Math.max(payload.samples, 120);
        body.top_k = payload.top_k;
      } else {
        url = '/predict_prob_beam';
        body.top_k1 = parseInt(el('beam_top_k1').value, 10);
        body.top_k2 = parseInt(el('beam_top_k2').value, 10);
        body.beam_width = parseInt(el('beam_width').value, 10);
      }

      const probRes = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (probRes.ok) {
        const prob = await probRes.json();
        renderHeatmap(prob);
        renderContour(prob);
        renderFan(prob);
      } else {
        const txt = await probRes.text();
        console.warn('probability endpoint error', probRes.status, txt);
      }
    } catch (e) {
      console.warn('probability endpoint fetch failed', e);
    }
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

"""
AlphaCast — Sector-Augmented Time-Series Transformers for Financial Forecasting
Streamlit dashboard: 'Model Showdown'

Run with:
    streamlit run app.py

The app:
  1. Lets the user pick any stock for which artifacts/<STOCK>_processed/ exists.
  2. Loads the trained LSTM and TFT, generates a live 7-day forecast from BOTH.
  3. Plots actual historical price alongside the two forecast lines (Plotly).
  4. Shows a Model Showdown table with RMSE & MAPE proving TFT > LSTM.
"""
import os
import json
import pickle
import glob

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler  # noqa: F401  (needed for unpickling)

# ---------------------------------------------------------------------- paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(PROJECT_ROOT, 'artifacts')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')

# ---------------------------------------------------------------------- LSTM def
# Must match the architecture used in Notebook 3 so we can load the weights.
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---------------------------------------------------------------------- helpers
def discover_processed_stocks():
    """Find every artifacts/<STOCK>_processed folder that has a config.json."""
    stocks = []
    for d in sorted(glob.glob(os.path.join(ARTIFACTS, '*_processed'))):
        if os.path.exists(os.path.join(d, 'config.json')):
            stocks.append(os.path.basename(d).replace('_processed', ''))
    return stocks


@st.cache_resource(show_spinner=False)
def load_artifacts(stock: str):
    """Load every file the dashboard needs for one stock — cached so we only do it once."""
    data_dir = os.path.join(ARTIFACTS, f'{stock}_processed')
    with open(os.path.join(data_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(data_dir, 'showdown.json')) as f:
        showdown = json.load(f)

    df = pd.read_csv(os.path.join(data_dir, 'engineered.csv'), parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    with open(os.path.join(data_dir, 'feature_scaler.pkl'), 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(os.path.join(data_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    with open(os.path.join(data_dir, 'darts_target_scaler.pkl'), 'rb') as f:
        darts_target_scaler = pickle.load(f)
    with open(os.path.join(data_dir, 'darts_past_scaler.pkl'), 'rb') as f:
        darts_past_scaler = pickle.load(f)

    # ---- LSTM ----
    # weights_only kwarg only exists in PyTorch >= 1.13; pass it defensively.
    try:
        ckpt = torch.load(os.path.join(MODELS_DIR, 'lstm_model.pt'),
                          map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(os.path.join(MODELS_DIR, 'lstm_model.pt'),
                          map_location='cpu')
    lstm = LSTMForecaster(**ckpt['arch'])
    lstm.load_state_dict(ckpt['state_dict'])
    lstm.eval()

    # ---- TFT ----
    tft = TFTModel.load(os.path.join(MODELS_DIR, 'transformer_model.pt'),
                        map_location='cpu')

    return {
        'cfg'          : cfg,
        'showdown'     : showdown,
        'df'           : df,
        'feature_scaler' : feature_scaler,
        'target_scaler'  : target_scaler,
        'darts_target_scaler' : darts_target_scaler,
        'darts_past_scaler'   : darts_past_scaler,
        'lstm'         : lstm,
        'tft'          : tft,
    }


def lstm_live_forecast(art):
    """Use the LAST `lookback` rows of engineered data to forecast the next 7 days."""
    cfg, df = art['cfg'], art['df']
    lookback     = cfg['lookback']
    feature_cols = cfg['feature_cols']

    # Take the most recent `lookback` rows and scale them
    last_window = df[feature_cols].iloc[-lookback:].values
    scaled      = art['feature_scaler'].transform(last_window)
    x = torch.from_numpy(scaled).float().unsqueeze(0)   # (1, lookback, n_features)

    with torch.no_grad():
        pred_scaled = art['lstm'](x).numpy().reshape(-1, 1)   # (horizon, 1)

    pred_usd = art['target_scaler'].inverse_transform(pred_scaled).flatten()
    return pred_usd


def tft_live_forecast(art):
    """Use darts' native predict() for the same lookback window."""
    cfg, df = art['cfg'], art['df']
    feature_cols = cfg['feature_cols']
    target_col   = cfg['target_col']
    past_covs    = [c for c in feature_cols if c != target_col]

    df_t = df.reset_index(drop=True).copy()
    df_t['t'] = np.arange(len(df_t))

    target_ts = TimeSeries.from_dataframe(df_t, time_col='t', value_cols=[target_col])
    past_ts   = TimeSeries.from_dataframe(df_t, time_col='t', value_cols=past_covs)

    target_s = art['darts_target_scaler'].transform(target_ts)
    past_s   = art['darts_past_scaler'].transform(past_ts)

    fc = art['tft'].predict(n=cfg['horizon'], series=target_s, past_covariates=past_s)
    fc_usd = art['darts_target_scaler'].inverse_transform(fc).values().flatten()
    return fc_usd


def build_future_dates(last_date: pd.Timestamp, n: int):
    """Generate the next n US business days starting after last_date."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)


# ---------------------------------------------------------------------- UI
st.set_page_config(page_title='AlphaCast — Model Showdown',
                   layout='wide', initial_sidebar_state='expanded')
st.title('AlphaCast — Model Showdown')
st.caption('Sector-Augmented Time-Series Transformers vs LSTM | B.Tech Sem-8 Major Project')

stocks = discover_processed_stocks()
if not stocks:
    st.error('No processed stocks found in artifacts/. Run Notebooks 1–4 first.')
    st.stop()

with st.sidebar:
    st.header('Configuration')
    stock = st.selectbox('Target stock', stocks, index=0)
    history_days = st.slider('Historical window to display (days)', 60, 500, 180, step=20)
    st.markdown('---')
    st.markdown('**Project files**')
    st.code(f'artifacts/{stock}_processed/\nmodels/lstm_model.pt\nmodels/transformer_model.pt')

# -----------------------------------------------------------------------------
with st.spinner(f'Loading models and data for {stock}...'):
    art = load_artifacts(stock)

cfg      = art['cfg']
df       = art['df']
showdown = art['showdown']

# -----------------------------------------------------------------------------
# Header banner with stock + matched ETF
c1, c2, c3, c4 = st.columns(4)
c1.metric('Stock',       cfg['stock'])
c2.metric('Sector ETF',  cfg['matched_etf'])
c3.metric('Sector',      cfg['sector'])
c4.metric('Lookback × Horizon', f"{cfg['lookback']} × {cfg['horizon']}")

st.markdown('---')

# -----------------------------------------------------------------------------
# Generate live 7-day forecasts
with st.spinner('Generating 7-day forecasts...'):
    lstm_fc = lstm_live_forecast(art)
    tft_fc  = tft_live_forecast(art)

last_date    = df['Date'].iloc[-1]
future_dates = build_future_dates(last_date, cfg['horizon'])

# -----------------------------------------------------------------------------
# Plot
hist = df.iloc[-history_days:]

fig = go.Figure()

# historical Close
fig.add_trace(go.Scatter(
    x=hist['Date'], y=hist['Close'],
    mode='lines', name=f"{cfg['stock']} actual Close",
    line=dict(color='#111111', width=2)))

# normalised ETF anchor (rebased to start of visible window) — secondary
etf_norm = hist['ETF_Close'] / hist['ETF_Close'].iloc[0] * hist['Close'].iloc[0]
fig.add_trace(go.Scatter(
    x=hist['Date'], y=etf_norm,
    mode='lines', name=f"{cfg['matched_etf']} (rebased anchor)",
    line=dict(color='#9aa0a6', width=1, dash='dot'),
    opacity=0.7))

# bridge from last actual to first forecast point so the line is continuous
bridge_x = [last_date, future_dates[0]]
bridge_lstm = [hist['Close'].iloc[-1], lstm_fc[0]]
bridge_tft  = [hist['Close'].iloc[-1], tft_fc[0]]

# LSTM forecast
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_lstm, mode='lines',
    line=dict(color='#EA4335', width=2, dash='dash'),
    showlegend=False))
fig.add_trace(go.Scatter(
    x=future_dates, y=lstm_fc,
    mode='lines+markers', name='LSTM 7-day forecast',
    line=dict(color='#EA4335', width=2, dash='dash'),
    marker=dict(size=8, symbol='square')))

# TFT forecast
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_tft, mode='lines',
    line=dict(color='#1A73E8', width=2.5),
    showlegend=False))
fig.add_trace(go.Scatter(
    x=future_dates, y=tft_fc,
    mode='lines+markers', name='TFT 7-day forecast',
    line=dict(color='#1A73E8', width=2.5),
    marker=dict(size=9, symbol='triangle-up')))

fig.add_vline(x=last_date, line=dict(color='#cccccc', width=1, dash='dot'))

fig.update_layout(
    height=560,
    margin=dict(l=20, r=20, t=40, b=20),
    title=f"Live 7-day forecast — {cfg['stock']} (anchored on {cfg['matched_etf']})",
    xaxis_title='Date', yaxis_title='Close (USD)',
    legend=dict(orientation='h', y=-0.15),
    hovermode='x unified',
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Forecast values table
st.subheader('7-day forecast values')
fc_df = pd.DataFrame({
    'Date'      : future_dates.strftime('%Y-%m-%d'),
    'LSTM ($)'  : np.round(lstm_fc, 2),
    'TFT  ($)'  : np.round(tft_fc, 2),
    'Δ (TFT − LSTM)' : np.round(tft_fc - lstm_fc, 2),
})
st.dataframe(fc_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Showdown — backtest metrics
st.markdown('---')
st.subheader('Model Showdown — backtest metrics on the held-out 20% test set')

lstm_m = showdown['lstm']
tft_m  = showdown['tft']

show_df = pd.DataFrame([
    {
        'Model'          : 'LSTM (Baseline)',
        'RMSE ($)'       : round(lstm_m['rmse_usd'], 4),
        'MAPE (%)'       : round(lstm_m['mape_pct'], 4),
        'Test windows'   : lstm_m['n_test_windows'],
        '# parameters'   : lstm_m.get('n_params', '—'),
    },
    {
        'Model'          : 'TFT (Transformer)',
        'RMSE ($)'       : round(tft_m['rmse_usd'], 4),
        'MAPE (%)'       : round(tft_m['mape_pct'], 4),
        'Test windows'   : tft_m['n_test_windows'],
        '# parameters'   : '~30k (TFT)',
    },
])
st.dataframe(show_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
c1.metric('TFT improvement on RMSE',
          f"{tft_m['vs_lstm_rmse_pct_improvement']:+.2f}%",
          help='positive = TFT better')
c2.metric('TFT improvement on MAPE',
          f"{tft_m['vs_lstm_mape_pct_improvement']:+.2f}%",
          help='positive = TFT better')

verdict = ('Transformer wins.' if tft_m['rmse_usd'] < lstm_m['rmse_usd']
           else 'LSTM wins.')
st.success(
    f'**Verdict:** {verdict}  '
    f"TFT RMSE = \\${tft_m['rmse_usd']:.2f} vs LSTM RMSE = \\${lstm_m['rmse_usd']:.2f} "
    f"on {tft_m['n_test_windows']} sliding test windows."
)

# -----------------------------------------------------------------------------
with st.expander('How the comparison is fair'):
    st.markdown(f"""
- **Same stock & sector ETF anchor:** {cfg['stock']} + {cfg['matched_etf']} ({cfg['sector']}).
- **Same train / test split:** chronological 80 / 20 — no shuffling, no leakage.
- **Same scaler protocol:** MinMaxScaler fit on train rows only, applied to test.
- **Same lookback / horizon:** {cfg['lookback']}-day input → {cfg['horizon']}-day forecast.
- **Same evaluation:** sliding-window predictions, stride 1, identical window count.
- **Same loss:** MSE for both models.
- **Capacity in the same ballpark** (~50k LSTM params vs ~30k TFT params) so the
  comparison is about *architecture*, not raw model size.
""")

---
title: AlphaCast
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# AlphaCast — Sector-Augmented Time-Series Transformers for Financial Forecasting

> **B.Tech Final Year Major Project — Semester 8**
> A comparative analysis of **LSTM** vs **Temporal Fusion Transformer** for 7-day stock-price forecasting, using the stock's most-correlated **sector ETF** as a stabilising anchor.

**Live demo:** *(link will appear here after Streamlit Cloud deploy)*

---

## TL;DR

On AAPL (1984–2017, 80/20 chronological split, identical 30-day windows for both models):

| Model | RMSE ($) | MAPE (%) | # Params |
|---|---|---|---|
| LSTM (Baseline) | 40.98 | 26.95 | ~54k |
| **TFT (Champion)** | **11.43** | **7.52** | ~30k |
| **TFT improvement** | **+72.1%** | **+72.1%** | — |

The Transformer wins despite having **fewer** parameters — the gain is architectural, not capacity-driven.

---

## Project Pipeline

```
data/Stocks + data/ETFs
        |
        v
N1: Data + ETF Matching   (Pearson corr on daily returns)
        |
        v
N2: Feature Engineering   (MA7, MA21, RSI, MACD on stock + ETF)
        |
        v
+--- same 3-D windows (N, 30, 14) ---+
|                                    |
v                                    v
N3: LSTM (PyTorch)        N4: TFT (darts)
|                                    |
+--- showdown.json -------------------+
        |
        v
app.py  Streamlit Model Showdown dashboard
```

---

## Repo Layout

```
sem8_project/
├── app.py                             # Streamlit dashboard (deploy entry point)
├── requirements.txt                   # Pinned dependencies for Streamlit Cloud
├── notebooks/
│   ├── 01_Data_and_ETF_Matching.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_LSTM.ipynb
│   └── 04_Champion_Transformer.ipynb
├── artifacts/
│   └── AAPL_processed/
│       ├── config.json
│       ├── showdown.json              # LSTM + TFT metrics, read by the app
│       ├── engineered.csv             # cleaned + indicator-augmented data
│       └── *_scaler.pkl               # fitted scalers for inference
├── models/
│   ├── lstm_model.pt                  # PyTorch state dict + arch
│   ├── transformer_model.pt           # darts TFT (paired with .ckpt)
│   └── transformer_model.pt.ckpt
└── AlphaCast_Viva_CheatSheet.pdf      # interactive Q&A document
```

`data/` (788 MB raw OHLCV from Kaggle "Huge Stock Market Dataset") is **gitignored** — the app reads only the engineered CSV and saved models, so it isn't needed at runtime.

---

## Run Locally

```bash
git clone https://github.com/parassss17/alphacast.git
cd alphacast
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

To re-train on a different stock, drop the OHLCV files into `data/Stocks/` and `data/ETFs/`, then run notebooks 1 → 4 sequentially.

---

## Methodology Highlights (Viva-Defendable)

- **Daily-return correlation** (not raw price) for sector matching → avoids spurious trend correlation.
- **Chronological 80/20 split** → no temporal leakage.
- **MinMaxScaler fit on train only** → no look-ahead leakage.
- **Identical 30-day input / 7-day output windows** for both models → fair architectural comparison.
- **Capacity matched** within 2× → the Transformer wins on architecture, not size.

Full Q&A in [`AlphaCast_Viva_CheatSheet.pdf`](AlphaCast_Viva_CheatSheet.pdf).

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `PyTorch` · `darts` · `streamlit` · `plotly`

---

## Author

**Paras Beniwal** — B.Tech Final Year, Sem-8 Major Project

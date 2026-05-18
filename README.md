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

# AlphaCast

B.Tech Final Year Major Project (Semester 8).

AlphaCast compares an LSTM against a Temporal Fusion Transformer (TFT) for
forecasting a stock's price 7 days ahead. The idea we test is whether anchoring
a stock to its most-correlated sector ETF gives the models useful extra context.
We use AAPL, and its matched sector ETF turns out to be XLK (Technology).

## Results

Backtest on AAPL (1984–2017 data, chronological 80/20 split, 601 test windows):

| Model | RMSE ($) | MAPE (%) | Params |
|---|---|---|---|
| LSTM (baseline) | 40.98 | 26.95 | ~54k |
| TFT (champion) | 11.43 | 7.52 | ~30k |

The TFT is about 72% better on both metrics, even though it has fewer
parameters — so the improvement comes from the architecture, not model size.

We also ran an ablation (Notebook 5) to check if the sector-ETF anchor actually
helps. Removing the ETF features and retraining the same TFT gave RMSE 12.08
(vs 11.43 with the ETF). So the anchor helps, but the effect is modest (~5% on
RMSE) — we report this honestly rather than overstating it.

## How it works

```
data (Stocks + ETFs)
  -> N1  match AAPL to its sector ETF (correlation on daily returns)
  -> N2  feature engineering (MA7, MA21, RSI, MACD on stock + ETF)
  -> N3  train LSTM        }  same windows, same split, same scaling
  -> N4  train TFT         }  so only the model differs
  -> N5  ETF ablation (TFT without the ETF features)
  -> app.py  Streamlit dashboard
```

## Files

```
sem8_project/
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── make_paper_figures.py        # regenerates the paper figures from saved artifacts
├── notebooks/
│   ├── 01_Data_and_ETF_Matching.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_LSTM.ipynb
│   ├── 04_Champion_Transformer.ipynb
│   └── 05_ETF_Ablation.ipynb
├── artifacts/AAPL_processed/    # engineered data, scalers, metrics, predictions
└── models/                      # trained LSTM + TFT (+ TFT-no-ETF from the ablation)
```

The raw 788 MB Kaggle dataset lives in `data/` and is gitignored. The app only
needs the engineered CSV and the saved models, so it isn't required to run.

## Run it

```bash
git clone https://github.com/parassss17/alphacast.git
cd alphacast
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

To train on a different stock, put its OHLCV files in `data/Stocks/` and the
ETF files in `data/ETFs/`, then run notebooks 1 to 4 in order (5 is optional).

## A few things we were careful about

- Sector matching uses daily-return correlation, not raw price, so a shared
  long-term uptrend doesn't create a fake correlation.
- The train/test split is chronological (no shuffling) so there's no leakage.
- Scalers are fit on the training data only.
- Both models use the exact same windows, split and loss, so the comparison
  is fair and about the architecture.

## Stack

pandas, numpy, scikit-learn, PyTorch, darts, Streamlit, plotly

## Author

Paras Beniwal — B.Tech Final Year, Semester 8 Major Project.

"""
Generate publication figures for the AlphaCast LNCS paper.
All figures are produced from the REAL saved artifacts / raw data —
nothing is mocked.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
ART  = os.path.join(PROJ, "artifacts")
PROC = os.path.join(ART, "AAPL_processed")
FIG  = "/Users/parasbeniwal/Desktop/reports sem8/alphacast_paper/figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities",
}

def load_ticker(ticker, folder):
    fp = os.path.join(folder, f"{ticker.lower()}.us.txt")
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date")
    if "OpenInt" in df.columns:
        df = df.drop(columns=["OpenInt"])
    return df.set_index("Date")

# ---------------------------------------------------------------- Fig: correlation bar
def fig_correlation():
    stock = load_ticker("AAPL", os.path.join(DATA, "Stocks"))
    s_ret = stock["Close"].pct_change().dropna()
    rows = []
    for tk, sec in SECTOR_ETFS.items():
        try:
            e = load_ticker(tk, os.path.join(DATA, "ETFs"))
        except FileNotFoundError:
            continue
        e_ret = e["Close"].pct_change().dropna()
        j = pd.concat([s_ret, e_ret], axis=1, join="inner").dropna()
        if len(j) < 252:
            continue
        j.columns = ["s", "e"]
        rows.append((tk, sec, j["s"].corr(j["e"])))
    rows.sort(key=lambda r: r[2], reverse=True)
    tks  = [r[0] for r in rows]
    cors = [r[2] for r in rows]
    colors = ["#1A73E8" if i == 0 else "#9aa0a6" for i in range(len(tks))]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.bar(tks, cors, color=colors)
    for b, c in zip(bars, cors):
        ax.text(b.get_x()+b.get_width()/2, c+0.012, f"{c:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Pearson correlation\n(daily returns)")
    ax.set_xlabel("SPDR sector ETF")
    ax.set_ylim(0, max(cors)*1.18)
    ax.set_title("AAPL daily-return correlation against each sector ETF")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_correlation.pdf"))
    fig.savefig(os.path.join(FIG, "fig_correlation.png"))
    plt.close(fig)
    print("correlation:", rows[0])
    return rows

# ---------------------------------------------------------------- Fig: AAPL vs XLK rebased
def fig_rebased():
    df = pd.read_csv(os.path.join(ART, "AAPL_with_XLK.csv"), parse_dates=["Date"])
    df = df.sort_values("Date")
    v = df.set_index("Date")[["Close", "ETF_Close"]].copy()
    v = v / v.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(v.index, v["Close"], label="AAPL", color="#111111", lw=1.3)
    ax.plot(v.index, v["ETF_Close"], label="XLK (Technology)", color="#1A73E8", lw=1.3)
    ax.set_ylabel("Normalised price\n(rebased to 100)")
    ax.set_xlabel("Date")
    ax.set_title("AAPL and its sector anchor XLK over the study period")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rebased.pdf"))
    fig.savefig(os.path.join(FIG, "fig_rebased.png"))
    plt.close(fig)

# ---------------------------------------------------------------- Fig: 1-day-ahead test
def fig_oneday():
    lp = np.load(os.path.join(PROC, "lstm_predictions.npy"), allow_pickle=True)
    tp = np.load(os.path.join(PROC, "tft_predictions.npy"), allow_pickle=True)
    ta = np.load(os.path.join(PROC, "tft_actuals.npy"), allow_pickle=True)
    n = min(len(lp), len(tp), len(ta))
    lp, tp, ta = lp[:n], tp[:n], ta[:n]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ta[:, 0], label="Actual", color="black", lw=1.5)
    ax.plot(lp[:, 0], label="LSTM", color="#EA4335", lw=1.0, alpha=0.85)
    ax.plot(tp[:, 0], label="TFT",  color="#1A73E8", lw=1.0, alpha=0.85)
    ax.set_xlabel("Test window index (chronological)")
    ax.set_ylabel("AAPL Close (USD)")
    ax.set_title("One-day-ahead forecast on the held-out test set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_oneday.pdf"))
    fig.savefig(os.path.join(FIG, "fig_oneday.png"))
    plt.close(fig)

# ---------------------------------------------------------------- Fig: sample 7-day window
def fig_window():
    lp = np.load(os.path.join(PROC, "lstm_predictions.npy"), allow_pickle=True)
    tp = np.load(os.path.join(PROC, "tft_predictions.npy"), allow_pickle=True)
    ta = np.load(os.path.join(PROC, "tft_actuals.npy"), allow_pickle=True)
    n = min(len(lp), len(tp), len(ta))
    i = n // 2
    d = np.arange(1, 8)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(d, ta[i], "o-", color="black", label="Actual")
    ax.plot(d, lp[i], "s--", color="#EA4335", label="LSTM")
    ax.plot(d, tp[i], "^--", color="#1A73E8", label="TFT")
    ax.set_xlabel("Forecast day ahead (t+1 … t+7)")
    ax.set_ylabel("AAPL Close (USD)")
    ax.set_title(f"Representative 7-day forecast (test window #{i})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_window.pdf"))
    fig.savefig(os.path.join(FIG, "fig_window.png"))
    plt.close(fig)

# ---------------------------------------------------------------- Fig: per-horizon RMSE
def fig_horizon_rmse():
    lp = np.load(os.path.join(PROC, "lstm_predictions.npy"), allow_pickle=True)
    la = np.load(os.path.join(PROC, "lstm_actuals.npy"), allow_pickle=True)
    tp = np.load(os.path.join(PROC, "tft_predictions.npy"), allow_pickle=True)
    ta = np.load(os.path.join(PROC, "tft_actuals.npy"), allow_pickle=True)
    n = min(len(lp), len(la), len(tp), len(ta))
    lp, la, tp, ta = lp[:n], la[:n], tp[:n], ta[:n]
    lstm_rmse = np.sqrt(np.mean((lp - la) ** 2, axis=0))
    tft_rmse  = np.sqrt(np.mean((tp - ta) ** 2, axis=0))
    x = np.arange(1, 8)
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w/2, lstm_rmse, w, label="LSTM", color="#EA4335")
    ax.bar(x + w/2, tft_rmse,  w, label="TFT",  color="#1A73E8")
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title("Error growth across the 7-day forecast horizon")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_horizon_rmse.pdf"))
    fig.savefig(os.path.join(FIG, "fig_horizon_rmse.png"))
    plt.close(fig)
    print("per-horizon LSTM RMSE:", np.round(lstm_rmse, 2))
    print("per-horizon TFT  RMSE:", np.round(tft_rmse, 2))

# ---------------------------------------------------------------- Fig: ETF ablation
def fig_ablation():
    """TFT with ETF vs TFT without ETF (1-day-ahead) — the ablation result."""
    ap = os.path.join(PROC, "tft_noetf_predictions.npy")
    if not os.path.exists(ap):
        print("ablation: tft_noetf_predictions.npy not found, skipping "
              "(run Notebook 5 first)")
        return
    tp = np.load(os.path.join(PROC, "tft_predictions.npy"), allow_pickle=True)
    npred = np.load(ap, allow_pickle=True)
    ta = np.load(os.path.join(PROC, "tft_actuals.npy"), allow_pickle=True)
    n = min(len(tp), len(npred), len(ta))
    tp, npred, ta = tp[:n], npred[:n], ta[:n]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ta[:, 0], label="Actual", color="black", lw=1.5)
    ax.plot(tp[:, 0], label="TFT with ETF", color="#1A73E8", lw=1.0, alpha=0.85)
    ax.plot(npred[:, 0], label="TFT without ETF", color="#F9AB00", lw=1.0, alpha=0.85)
    ax.set_xlabel("Test window index (chronological)")
    ax.set_ylabel("AAPL Close (USD)")
    ax.set_title("ETF ablation: does the sector anchor help? (1-day-ahead)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf"))
    fig.savefig(os.path.join(FIG, "fig_ablation.png"))
    plt.close(fig)
    with open(os.path.join(PROC, "ablation.json")) as f:
        ab = json.load(f)
    print("ablation: RMSE no-ETF %.2f vs with-ETF %.2f  (ETF cut RMSE by %.2f%%)"
          % (ab["rmse_usd"], ab["with_etf_rmse_usd"], ab["etf_rmse_reduction_pct"]))


if __name__ == "__main__":
    rows = fig_correlation()
    fig_rebased()
    fig_oneday()
    fig_window()
    fig_horizon_rmse()
    fig_ablation()
    print("\nAll figures written to:", FIG)
    print(os.listdir(FIG))

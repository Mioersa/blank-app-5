import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Intraday Option‑Chain Analyzer", layout="wide")
st.title("🚀 Intraday Option‑Chain Correlation & Buy/Sell Signal Analyzer")

# ---------- File Upload ----------
st.sidebar.header("Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (pattern *_ddmmyyyy_hhmmss.csv):",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("⬆️ Upload 1 or more intraday Option Chain CSV files to continue.")
    st.stop()

st.write(f"Loaded {len(uploaded_files)} file(s)")

# ---------- Calculation Function ----------
def compute_features(df):
    # Diffs and pct change
    for s in ["CE", "PE"]:
        df[f"{s}_ΔPrice"] = df[f"{s}_lastPrice"].diff()
        df[f"{s}_%ret"] = df[f"{s}_lastPrice"].pct_change() * 100
        df[f"{s}_ΔOI"] = df[f"{s}_openInterest"].diff()
        df[f"{s}_ΔVol"] = df[f"{s}_totalTradedVolume"].diff()
        df[f"{s}_ΔIV"] = df[f"{s}_impliedVolatility"].diff()

    # Rolling correlations
    df["r_price_OI_CE"] = df["CE_ΔPrice"].rolling(20).corr(df["CE_ΔOI"])
    df["r_price_vol_CE"] = df["CE_ΔPrice"].rolling(20).corr(df["CE_ΔVol"])
    df["r_price_OI_PE"] = df["PE_ΔPrice"].rolling(20).corr(df["PE_ΔOI"])
    df["r_price_vol_PE"] = df["PE_ΔPrice"].rolling(20).corr(df["PE_ΔVol"])

    # OI imbalance
    df["OIimb"] = (df["CE_openInterest"] - df["PE_openInterest"]) / (
        df["CE_openInterest"] + df["PE_openInterest"]
    )

    # Composite strength score
    df["strength"] = (
        0.4 * df["r_price_OI_CE"]
        + 0.3 * df["r_price_vol_CE"]
        + 0.3 * df["OIimb"]
    )

    # Lead–lag correlation
    lags = range(-3, 4)
    corrs = [df["CE_lastPrice"].corr(df["CE_openInterest"].shift(l)) for l in lags]
    lag_df = pd.DataFrame({"lag": lags, "corr": corrs})
    best_lag = int(lag_df.loc[lag_df["corr"].idxmax(), "lag"])

    # Regime detection
    rollcorr = df["CE_lastPrice"].rolling(20).corr(df["CE_openInterest"])
    regime = "Bullish" if rollcorr.dropna().iloc[-1] > 0 else "Bearish"

    # Latest snapshot
    latest = df.iloc[-1]
    res = dict(
        r_price_OI_CE=round(df["r_price_OI_CE"].dropna().iloc[-1], 3)
        if df["r_price_OI_CE"].dropna().size
        else 0,
        r_price_vol_CE=round(df["r_price_vol_CE"].dropna().iloc[-1], 3)
        if df["r_price_vol_CE"].dropna().size
        else 0,
        OIimb=round(latest["OIimb"], 3),
        strength=round(df["strength"].dropna().iloc[-1], 3)
        if df["strength"].dropna().size
        else 0,
        best_lag=best_lag,
        regime=regime,
    )

    # Signal decision
    if res["strength"] > 0.2:
        res["Signal"] = "📈 Buy CE"
    elif res["strength"] < -0.2:
        res["Signal"] = "📉 Buy PE"
    else:
        res["Signal"] = "⚖️ Neutral"

    return res

# ---------- Process Uploaded Files ----------
results = []
for f in uploaded_files:
    try:
        df = pd.read_csv(f)
        res = compute_features(df)
        res["file"] = f.name
        results.append(res)
    except Exception as e:
        st.warning(f"❌ Error in {f.name}: {e}")

if not results:
    st.error("No valid results generated.")
    st.stop()

summary = pd.DataFrame(results).set_index("file")
st.success("✅ Analysis complete — Summary below.")
st.dataframe(summary.style.background_gradient(cmap="RdYlGn"))

# ---------- Download Button ----------
st.download_button(
    "📥 Download Summary CSV",
    summary.to_csv().encode(),
    "OptionChain_Summary.csv",
    "text/csv",
)

# ---------- Quick Overview ----------
st.subheader("📊 Aggregate Stats")
bull = (summary["Signal"] == "📈 Buy CE").sum()
bear = (summary["Signal"] == "📉 Buy PE").sum()
neu = (summary["Signal"] == "⚖️ Neutral").sum()
st.write(f"➡️ {bull} Bullish files | {bear} Bearish | {neu} Neutral")
st.bar_chart(summary[["r_price_OI_CE", "r_price_vol_CE", "OIimb", "strength"]])

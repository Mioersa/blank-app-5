import streamlit as st
import pandas as pd
import numpy as np
import io, os

st.set_page_config(page_title="Intraday Option‑Chain Analyzer", layout="wide")
st.title("🚀 Intraday Option‑Chain Correlation & Buy/Sell Signal Analyzer")

# ---------- File Upload ----------
st.sidebar.header("Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (*_ddmmyyyy_hhmmss.csv):",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("⬆️ Upload 1 or more intraday Option Chain CSV files to proceed.")
    st.stop()

st.write(f"Loaded {len(uploaded_files)} file(s)")

# ---------- Core Computation ----------
def compute_features(df):
    # Compute diffs and pct returns
    for side in ["CE", "PE"]:
        df[f"{side}_ΔPrice"] = df[f"{side}_lastPrice"].diff()
        df[f"{side}_%ret"] = df[f"{side}_lastPrice"].pct_change() * 100
        df[f"{side}_ΔOI"] = df[f"{side}_openInterest"].diff()
        df[f"{side}_ΔVol"] = df[f"{side}_totalTradedVolume"].diff()
        df[f"{side}_ΔIV"] = df[f"{side}_impliedVolatility"].diff()

    # Rolling correlations
    df["r_price_vol_CE"] = df["CE_ΔPrice"].rolling(20).corr(df["CE_ΔVol"])
    df["r_price_OI_CE"] = df["CE_ΔPrice"].rolling(20).corr(df["CE_ΔOI"])
    df["r_price_vol_PE"] = df["PE_ΔPrice"].rolling(20).corr(df["PE_ΔVol"])
    df["r_price_OI_PE"] = df["PE_ΔPrice"].rolling(20).corr(df["PE_ΔOI"])

    # OI imbalance
    df["OIimb"] = (df["CE_openInterest"] - df["PE_openInterest"]) / (
        df["CE_openInterest"] + df["PE_openInterest"]
    )

    # Trend strength = weighted composite
    df["strength"] = (
        0.4 * df["r_price_OI_CE"]
        + 0.3 * df["r_price_vol_CE"]
        + 0.3 * df["OIimb"]
    )

    # Lead–lag correlation: OI lead test
    lags = range(-3, 4)
    corrs = [df["CE_lastPrice"].corr(df["CE_openInterest"].shift(l)) for l in lags]
    lag_df = pd.DataFrame({"lag": lags, "corr": corrs})
    best_lag = lag_df.loc[lag_df["corr"].idxmax(), "lag"]

    # Regime (based on rolling Price–OI corr)
    rollcorr = df["CE_lastPrice"].rolling(20).corr(df["CE_openInterest"])
    regime = "Bullish" if rollcorr.dropna().iloc[-1] > 0 else "Bearish"

    # Collect summary metrics
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
        best_lag=int(best_lag),
        regime=regime,
    )

    # Simple directional signal rules
    if res["strength"] > 0.2:
        res["Signal"] = "📈 Buy CE"
    elif res["strength"] < -0.2:
        res["Signal"] = "📉 Buy PE"
    else:
        res["Signal"] = "⚖️ Neutral"

    return res


# ---------- Run Over Uploaded CSVs ----------
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
    st.error("No valid data parsed.")
    st.stop()

summary = pd.DataFrame(results).set_index("file")
st.success("✅ Computation complete — see summary below.")
st.dataframe(summary.style.background_gradient(cmap="RdYlGn"))

# ---------- Download ----------
st.download_button(
    "📥 Download Summary CSV",
    summary.to_csv().encode(),
    "OptionChain_Summary.csv",
    "text/csv",
)

# ---------- Quick insight view ----------
st.subheader("📊 Quick Stats")
bull = (summary["Signal"] == "📈 Buy CE").sum()
bear = (summary["Signal"] == "📉 Buy PE").sum()
neu = (summary["Signal"] == "⚖️ Neutral").sum()
st.write(f"➡️ {bull} Bullish files | {bear} Bearish | {neu} Neutral")

st.bar_chart(summary[["r_price_OI_CE", "r_price_vol_CE", "OIimb", "strength"]])

# ---------- Footer ----------
st.caption(
    "📘 Metrics derived from ΔPrice, ΔOI, ΔVol, OI imbalance, lead–lag, and regime logic."
)
st.caption("Deploy via Streamlit Cloud or GitHub → add:")
st.code("streamlit\npandas\nnumpy", language="text")

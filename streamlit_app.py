
import streamlit as st
import pandas as pd
import numpy as np
import glob, os

st.title("Intraday Option Chain Correlation Analysis")

# === Config ===
path = st.text_input("📁 Folder path of CSVs", "./data")
if not os.path.exists(path):
    st.stop()

files = sorted(glob.glob(os.path.join(path, "*.csv")))
st.write(f"Found {len(files)} CSVs")

def analyze(df):
    res = {}
    for side in ["CE", "PE"]:
        df[f'{side}_ΔPrice'] = df[f'{side}_lastPrice'].diff()
        df[f'{side}_%ret'] = df[f'{side}_lastPrice'].pct_change() * 100
        df[f'{side}_ΔOI'] = df[f'{side}_openInterest'].diff()
        df[f'{side}_ΔVol'] = df[f'{side}_totalTradedVolume'].diff()
        df[f'{side}_ΔIV'] = df[f'{side}_impliedVolatility'].diff()

    # Correlations
    res['corr_price_OI_CE'] = np.corrcoef(df['CE_ΔPrice'].dropna(), df['CE_ΔOI'].dropna())[0, 1]
    res['corr_price_OI_PE'] = np.corrcoef(df['PE_ΔPrice'].dropna(), df['PE_ΔOI'].dropna())[0, 1]
    res['corr_price_vol_CE'] = np.corrcoef(df['CE_ΔPrice'].dropna(), df['CE_ΔVol'].dropna())[0, 1]
    res['corr_price_vol_PE'] = np.corrcoef(df['PE_ΔPrice'].dropna(), df['PE_ΔVol'].dropna())[0, 1]
    res['OIimb'] = (df['CE_openInterest'].iloc[-1] - df['PE_openInterest'].iloc[-1]) / (
                   df['CE_openInterest'].iloc[-1] + df['PE_openInterest'].iloc[-1])

    # Composite score
    r_price_OI = (res['corr_price_OI_CE'] - res['corr_price_OI_PE'])
    r_price_vol = (res['corr_price_vol_CE'] - res['corr_price_vol_PE'])
    res['Composite'] = 0.4*r_price_OI + 0.3*r_price_vol + 0.3*res['OIimb']

    # Simple recommendation
    if res['Composite'] > 0.2:
        res['Signal'] = "📈 Buy CE"
    elif res['Composite'] < -0.2:
        res['Signal'] = "📉 Buy PE"
    else:
        res['Signal'] = "⚖️ Neutral"
    return res

results = []
for f in files:
    try:
        df = pd.read_csv(f)
        res = analyze(df)
        res["file"] = os.path.basename(f)
        results.append(res)
    except Exception as e:
        st.warning(f"Error {f}: {e}")

if results:
    summary = pd.DataFrame(results).set_index("file")
    st.dataframe(summary.style.background_gradient(cmap="coolwarm"))
    st.download_button("📥 Download results CSV", summary.to_csv().encode(), "summary.csv", "text/csv")
    st.write("✅ Done. Each row shows directional bias per 5‑min snapshot file.")


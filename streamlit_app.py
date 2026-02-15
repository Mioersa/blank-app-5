import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

# ---------- Page Setup ----------
st.set_page_config(page_title="Strike‑wise Intraday Analyzer", layout="wide")
st.title("🎯 Strike‑wise Intraday Option Analysis")

# ---------- Upload & Load ----------
@st.cache_data
def load_files(files):
    dfs=[]
    for f in files:
        try:
            name=f.name.split('.')[0]
            dd,hh=name.split('_')[-2],name.split('_')[-1]
            ts=datetime.strptime(dd+hh,"%d%m%Y%H%M%S")
        except: ts=pd.Timestamp.utcnow()
        df=pd.read_csv(f)
        if "timestamp" not in df.columns:
            df["timestamp"]=ts
        else:
            df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce").fillna(ts)
        dfs.append(df)
    data=pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    return data

files=st.file_uploader("Upload multiple 5‑min CSVs",type="csv",accept_multiple_files=True)

if files:
    data=load_files(files)
    st.success(f"✅ Merged {len(files)} files → {len(data)} rows")

    # ---------- Strike selector ----------
    strikes=sorted(data["CE_strikePrice"].unique())
    sel_strike=st.selectbox("Select Strike Price",strikes)

    # ---------- Side selector ----------
    side_opt=st.radio("Choose side(s) to analyze",["CE","PE","Both"],horizontal=True)

    # ---------- Chart type ----------
    chart_type=st.radio("Chart type",["Line","Bar"],horizontal=True)

    if st.button("Analyze Strike"):
        df=data[data["CE_strikePrice"]==sel_strike].reset_index(drop=True)
        st.info(f"Analyzing strike {sel_strike}")

        for s in ["CE","PE"]:
            if f"{s}_lastPrice" not in df.columns: continue
            df[f"{s}_price_change"]=df[f"{s}_lastPrice"].diff()
            df[f"{s}_vol_change"]=df[f"{s}_totalTradedVolume"].diff()
            df[f"{s}_oi_change"]=df[f"{s}_openInterest"].diff()

        # Choose which series to show
        plot_sides=[]
        if side_opt in ("CE","Both"): plot_sides.append("CE")
        if side_opt in ("PE","Both"): plot_sides.append("PE")

        metrics=[]
        for side in plot_sides:
            metrics.append({
                "Side": side,
                "Price Δ": df[f"{side}_lastPrice"].iloc[-1]-df[f"{side}_lastPrice"].iloc[0],
                "OI Δ": df[f"{side}_openInterest"].iloc[-1]-df[f"{side}_openInterest"].iloc[0],
                "Vol Δ": df[f"{side}_totalTradedVolume"].iloc[-1]-df[f"{side}_totalTradedVolume"].iloc[0],
            })
        st.subheader("📊 Metrics Snapshot")
        st.dataframe(pd.DataFrame(metrics).round(2))

        # ---------- Plot section ----------
        plots=["lastPrice","vol_change","oi_change"]
        titles=["Price Over Time","Volume Change","Open Interest Change"]
        colors={"CE":"green","PE":"red"}

        for i,base in enumerate(plots):
            fig,ax=plt.subplots(figsize=(10,3))
            for s in plot_sides:
                data_series=df[f"{s}_{base}"]
                if chart_type=="Line":
                    ax.plot(df["timestamp"],data_series,color=colors[s],label=s)
                else:
                    ax.bar(df["timestamp"],data_series,color=colors[s],label=s,alpha=0.6)
            ax.legend(); ax.set_title(titles[i])
            st.pyplot(fig)

        # ---------- Simple Direction Indication ----------
        st.subheader("📈 Directional Summary")
        def direction_text(price_d, oi_d, side):
            if price_d>0 and oi_d>0:
                return f"{side}: Price↑ & OI↑ → **Long / Bullish build-up**"
            elif price_d<0 and oi_d<0:
                return f"{side}: Price↓ & OI↓ → **Unwinding / Profit booking**"
            elif price_d<0 and oi_d>0:
                return f"{side}: Price↓ & OI↑ → **Short build-up**"
            else:
                return f"{side}: Mixed signals"
        for m in metrics:
            st.write(direction_text(np.sign(m["Price Δ"]),np.sign(m["OI Δ"]),m["Side"]))

else:
    st.write("👆 Upload intraday 5‑min CSV files to start.")

st.caption("Tips: use Both‑side view for pair understanding; try bar chart to spot spikes quickly.")



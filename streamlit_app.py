import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Intraday CE–PE Signal Dashboard", layout="wide")
st.title("📊 Intraday CE–PE Signal Dashboard (Full Feature Edition)")

files = st.file_uploader(
    "Upload multiple 5‑min CSVs (*_ddmmyyyy_hhmmss.csv format)", 
    type="csv", 
    accept_multiple_files=True
)

if files:
    dfs=[]
    for f in files:
        try:
            name=f.name.split('.')[0]
            ddmmyy,hhmmss=name.split('_')[-2],name.split('_')[-1]
            ts=datetime.strptime(ddmmyy+hhmmss,"%d%m%Y%H%M%S")
        except Exception:
            ts=pd.Timestamp.utcnow()
        df=pd.read_csv(f)
        df["timestamp"]=ts
        dfs.append(df)

    data=pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    st.success(f"✅ {len(files)} files merged → {len(data)} total rows")

    window=st.sidebar.slider("Rolling window size",5,50,20)

    # ---------- Step 1: Price & derived columns ----------
    st.subheader("1️⃣ Price, Volume, and OI Changes")

    for s in ["CE","PE"]:
        data[f"{s}_price_change"]=data[f"{s}_lastPrice"].diff()
        data[f"{s}_pct_return"]=data[f"{s}_lastPrice"].pct_change()*100
        data[f"{s}_vol_change"]=data[f"{s}_totalTradedVolume"].diff()
        data[f"{s}_oi_change"]=data[f"{s}_openInterest"].diff()

    cols_plot=[["CE_lastPrice","PE_lastPrice"],
               ["CE_vol_change","PE_vol_change"],
               ["CE_oi_change","PE_oi_change"]]
    titles=["Price Over Time","Volume Change","Open Interest Change"]
    colors=[("green","red"),("blue","orange"),("purple","brown")]

    for (pair,title,c) in zip(cols_plot,titles,colors):
        fig,ax=plt.subplots(figsize=(10,3))
        ax.plot(data["timestamp"],data[pair[0]],color=c[0],label=pair[0])
        ax.plot(data["timestamp"],data[pair[1]],color=c[1],label=pair[1])
        ax.legend(); ax.set_title(title); st.pyplot(fig)

    # ---------- Step 2: Volume confirmation ----------
    st.subheader("2️⃣ Volume Confirmation")
    for s in ["CE","PE"]:
        r=data[f"{s}_price_change"].corr(data[f"{s}_vol_change"])
        st.info(f"{s}: corr(price∆,vol∆)={r:.2f} {'→ strong move' if abs(r)>0.4 else ''}")

    # ---------- Step 3: Price–OI correlation ----------
    st.subheader("3️⃣ Price–OI Correlation")
    corr_results={}
    for s in ["CE","PE"]:
        corr_results[s]=data[f"{s}_price_change"].corr(data[f"{s}_oi_change"])
        note="long build‑up" if corr_results[s]>0 else "short build‑up"
        st.write(f"{s}: {corr_results[s]:.2f} → {note}")

    # ---------- Step 4: Rolling correlations (CE + PE) ----------
    st.subheader("4️⃣ Rolling Correlations + Crossovers")

    def plot_rollcorr(side,color):
        roll=data[f"{side}_price_change"].rolling(window).corr(data[f"{side}_oi_change"])
        fig,ax=plt.subplots(figsize=(10,3))
        ax.plot(data["timestamp"],roll,color=color,label=f"{side} roll corr")
        ax.axhline(0,color='gray',ls='--')

        # crossover markers
        sign=np.sign(roll)
        cross=np.where(np.sign(roll)!=np.sign(roll.shift(1)))[0]
        ax.scatter(data["timestamp"].iloc[cross],
                   roll.iloc[cross],
                   color=np.where(roll.iloc[cross]>0,'green','red'),
                   s=50,alpha=0.8,marker='o',label='Crossover')

        ax.fill_between(data["timestamp"],0,roll,where=roll>0,
                        color='green',alpha=0.2)
        ax.fill_between(data["timestamp"],0,roll,where=roll<0,
                        color='red',alpha=0.2)
        ax.legend(); ax.set_title(f"{side}: Rolling Price–OI Correlation")
        st.pyplot(fig)
        return roll

    ce_roll=plot_rollcorr("CE","teal")
    pe_roll=plot_rollcorr("PE","orange")
    st.caption("Green dots = bullish build‑up crossover; Red = bearish reversal.")

    # ---------- Step 5: Lag test ----------
    st.subheader("5️⃣ Lead/Lag (OI leads Price?)")
    maxlag=5
    for s in ["CE","PE"]:
        lags=[]
        for lag in range(1,maxlag+1):
            val=data[f"{s}_price_change"].corr(data[f"{s}_openInterest"].shift(lag))
            lags.append((lag,val))
        lagdf=pd.DataFrame(lags,columns=["Lag(5‑min)","Corr"])
        best=lagdf.loc[lagdf["Corr"].idxmax()]
        st.write(f"{s}: highest corr {best['Corr']:.2f} at lag {best['Lag(5‑min)']} → OI leads by {best['Lag(5‑min)']*5} min")
        st.dataframe(lagdf)

    # ---------- Step 6: IV correlation ----------
    st.subheader("6️⃣ Price–IV Correlation")
    for s in ["CE","PE"]:
        if f"{s}_impliedVolatility" in data.columns:
            r=data[f"{s}_price_change"].corr(data[f"{s}_impliedVolatility"].diff())
            st.write(f"{s}: {r:.2f} → {'nervous breakout' if r>0 else 'calm trend'}")

    # ---------- Step 7 & 8: Same‑side Sentiment ----------
    st.subheader("7️⃣‑8️⃣ Same‑Side Sentiment (Price ↔ OI)")
    ce_sent=data["CE_lastPrice"].corr(data["CE_openInterest"])
    pe_sent=data["PE_lastPrice"].corr(data["PE_openInterest"])
    st.info(f"CE sentiment={ce_sent:.2f} {'(bullish build‑up)' if ce_sent>0.3 else ''}")
    st.info(f"PE sentiment={pe_sent:.2f} {'(bearish build‑up)' if pe_sent>0.3 else ''}")

    # ---------- Step 9: OI Imbalance ----------
    st.subheader("9️⃣ OI Imbalance")
    data["OI_imb"]=(data["CE_openInterest"]-data["PE_openInterest"])/(data["CE_openInterest"]+data["PE_openInterest"])
    fig,ax=plt.subplots(figsize=(10,3))
    ax.plot(data["timestamp"],data["OI_imb"],color='black')
    ax.axhline(0,color='gray',ls='--'); st.pyplot(fig)
    st.write(f"Latest Imbalance={data['OI_imb'].iloc[-1]:.2f} (>0 calls dominate, <0 puts dominate)")

    # ---------- Step 10: Sentiment + strength ----------
    st.subheader("🔟 Sentiment Classification & Strength Score")
    latest=data.iloc[-1]
    ce_up=latest["CE_price_change"]>0 and latest["CE_oi_change"]>0
    pe_up=latest["PE_price_change"]>0 and latest["PE_oi_change"]>0
    if ce_up and not pe_up:
        mood="Bullish"
    elif pe_up and not ce_up:
        mood="Bearish"
    else:
        mood="Neutral"
    st.success(f"Current sentiment: **{mood}**")

    strength=(abs(corr_results["CE"])+abs(corr_results["PE"])
              +abs(ce_sent)+abs(pe_sent))/4
    bb_metric=ce_sent - pe_sent + data["OI_imb"].iloc[-1]
    st.write(f"Strength Score={strength:.2f}, Bull/Bear Imbalance={bb_metric:.2f}")

    # ---------- Summary table ----------
    st.subheader("📋 Summary Snapshot")
    summary=pd.DataFrame({
        "Metric":["CE_price_OI","PE_price_OI","CE_sent","PE_sent","OI_imbalance",
                  "Strength","BullBear_metric"],
        "Value":[corr_results["CE"],corr_results["PE"],ce_sent,pe_sent,
                 data["OI_imb"].iloc[-1],strength,bb_metric]
    })
    st.dataframe(summary.round(3))

    # ---------- ADDITIONAL STRIKE‑SPECIFIC FUNCTION ----------
    st.markdown("---")
    st.header("🎯 Strike‑Specific Quick Visuals")
    strikes=sorted(data["CE_strikePrice"].unique())
    sel=st.selectbox("Select Strike Price", strikes)
    if st.button("Analyze Selected Strike"):
        df_strike=data[data["CE_strikePrice"]==sel].reset_index(drop=True)
        st.info(f"Analyzing Strike: {sel}")

        # compute changes for that strike
        for s in ["CE","PE"]:
            df_strike[f"{s}_price_change"]=df_strike[f"{s}_lastPrice"].diff()
            df_strike[f"{s}_vol_change"]=df_strike[f"{s}_totalTradedVolume"].diff()
            df_strike[f"{s}_oi_change"]=df_strike[f"{s}_openInterest"].diff()

        plots=[["CE_lastPrice","PE_lastPrice"],
               ["CE_vol_change","PE_vol_change"],
               ["CE_oi_change","PE_oi_change"]]
        titles=["Price Over Time","Volume Change","Open Interest Change"]
        colors=[("green","red"),("blue","orange"),("purple","brown")]

        for (pair,title,c) in zip(plots,titles,colors):
            fig,ax=plt.subplots(figsize=(10,3))
            ax.plot(df_strike["timestamp"],df_strike[pair[0]],color=c[0],label=pair[0])
            ax.plot(df_strike["timestamp"],df_strike[pair[1]],color=c[1],label=pair[1])
            ax.legend(); ax.set_title(title); st.pyplot(fig)
        st.caption("Green = CE (Call); Red = PE (Put)")

else:
    st.write("👆 Upload 5‑min CSVs (e.g. *4500.csv*) to begin.")

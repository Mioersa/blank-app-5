import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Strike‑wise Intraday Analyzer", layout="wide")
st.title("🎯 Individual Strike Analysis – Price, Volume & OI Moves")

# ---------- Upload & Load ----------
files=st.file_uploader("Upload multiple 5‑min CSVs (*_ddmmyyyy_hhmmss.csv)",type="csv",accept_multiple_files=True)

if files:
    dfs=[]
    for f in files:
        try:
            name=f.name.split('.')[0]
            dd,hh=name.split('_')[-2],name.split('_')[-1]
            ts=datetime.strptime(dd+hh,"%d%m%Y%H%M%S")
        except: ts=pd.Timestamp.utcnow()
        df=pd.read_csv(f)
        df["timestamp"]=ts
        dfs.append(df)
    data=pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    st.success(f"✅ Merged {len(files)} files → {len(data)} rows")

    # ---------- Strike selector ----------
    strikes=sorted(data["CE_strikePrice"].unique())
    sel=st.selectbox("Select Strike Price",strikes)
    if st.button("Analyze Strike"):
        df=data[data["CE_strikePrice"]==sel].reset_index(drop=True)
        st.info(f"Analyzing strike {sel}")

        # ---------- Compute quick deltas ----------
        for s in ["CE","PE"]:
            df[f"{s}_price_change"]=df[f"{s}_lastPrice"].diff()
            df[f"{s}_vol_change"]=df[f"{s}_totalTradedVolume"].diff()
            df[f"{s}_oi_change"]=df[f"{s}_openInterest"].diff()

        # ---------- Plot changes ----------
        plots=[["CE_lastPrice","PE_lastPrice"],
               ["CE_vol_change","PE_vol_change"],
               ["CE_oi_change","PE_oi_change"]]
        titles=["Price Over Time","Volume Change","Open Interest Change"]
        colors=[("green","red"),("blue","orange"),("purple","brown")]

        for (pair,title,c) in zip(plots,titles,colors):
            fig,ax=plt.subplots(figsize=(10,3))
            ax.plot(df["timestamp"],df[pair[0]],color=c[0],label=pair[0])
            ax.plot(df["timestamp"],df[pair[1]],color=c[1],label=pair[1])
            ax.legend(); ax.set_title(title); st.pyplot(fig)
        st.caption("Green = CE (Call)  Red = PE (Put)")

        # ---------- Simple Direction Indicators ----------
        st.subheader("📈 Price‑Direction Indications")
        ce_dir=np.sign(df["CE_price_change"].sum())
        pe_dir=np.sign(df["PE_price_change"].sum())
        ce_oi_dir=np.sign(df["CE_oi_change"].sum())
        pe_oi_dir=np.sign(df["PE_oi_change"].sum())

        # CE
        if ce_dir>0 and ce_oi_dir>0:
            st.success("CE Call side: Price↑ & OI↑ → **Bullish build‑up** calls being bought.")
        elif ce_dir<0 and ce_oi_dir<0:
            st.warning("CE Price↓ & OI↓ → **Long unwinding** calls being exited.")
        elif ce_dir<0 and ce_oi_dir>0:
            st.error("CE Price↓ & OI↑ → **Short build‑up** calls being sold.")
        else:
            st.info("CE mixed → no clear direction.")

        # PE
        if pe_dir>0 and pe_oi_dir>0:
            st.error("PE Put side: Price↑ & OI↑ → **Bearish build‑up** puts being bought.")
        elif pe_dir<0 and pe_oi_dir<0:
            st.warning("PE Price↓ & OI↓ → **Short covering** puts being closed.")
        elif pe_dir<0 and pe_oi_dir>0:
            st.error("PE Price↓ & OI↑ → **Bearish continuation** new short puts.")
        else:
            st.info("PE mixed → no clear direction.")

        # ---------- Mini summary ----------
        st.subheader("🧾 Summary Snapshot")
        summary={
            "Strike":sel,
            "CE_price_change_total":df["CE_price_change"].sum(),
            "PE_price_change_total":df["PE_price_change"].sum(),
            "CE_oi_change_total":df["CE_oi_change"].sum(),
            "PE_oi_change_total":df["PE_oi_change"].sum()
        }
        st.dataframe(pd.DataFrame([summary]).T.rename(columns={0:"Value"}).round(2))

else:
    st.write("👆 Upload intraday 5‑min CSV files to start.")

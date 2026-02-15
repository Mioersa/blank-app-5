# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob, os

st.set_page_config(page_title="Strike‑Wise Intraday Analytics", layout="wide")
st.title("🚀 Strike‑Wise Intraday Analytics Dashboard")

# ----------  LOAD & PREP  ----------
files = sorted(glob.glob("*_[0-9]*.csv"))
if not files:
    st.warning("No CSVs found. Expected filenames like XYZ_05032024_093500.csv")
    st.stop()

multi_df = []
for f in files:
    ts = os.path.splitext(f)[0].split("_")[-1]
    try:
        dt = pd.to_datetime(ts, format="%d%m%Y_%H%M%S")
    except:
        dt = pd.to_datetime("now")
    df = pd.read_csv(f)
    df["timestamp"] = dt
    multi_df.append(df)

df = pd.concat(multi_df).sort_values("timestamp").reset_index(drop=True)
df['CE_strikePrice'] = df['CE_strikePrice'].round()
df['PE_strikePrice'] = df['PE_strikePrice'].round()

# ----------  SIDEBAR SETTINGS  ----------
st.sidebar.header("Filters")
strike = st.sidebar.selectbox("Strike Price", sorted(df['CE_strikePrice'].unique()))
view = st.sidebar.radio("View", ["CE", "PE", "Both"])
ma_opt = st.sidebar.multiselect("Moving Avg", [5,10,20])
agg_opt = st.sidebar.checkbox("Aggregate All Strikes", False)

# ----------  BASIC SECTION ----------
st.header("📊 Price & Volume Trends")

def draw_price(df,v,label):
    fig = px.line(df, x='timestamp', y=v, title=label)
    for w in ma_opt:
        df[f"MA_{w}"] = df[v].rolling(w).mean()
        fig.add_scatter(x=df['timestamp'], y=df[f"MA_{w}"], mode='lines', name=f"MA{w}")
    st.plotly_chart(fig,use_container_width=True)

data = df if agg_opt else df[df['CE_strikePrice']==strike]

if view in ["CE","Both"]:
    draw_price(data,'CE_lastPrice', f"CE Price @ {strike}")
if view in ["PE","Both"]:
    draw_price(data,'PE_lastPrice', f"PE Price @ {strike}")

# ----------  PRICE CHANGE / MOMENTUM ----------
st.subheader("📈 Price Change & Momentum")
data['CE_delta'] = data['CE_lastPrice'].diff()
data['CE_pct'] = data['CE_lastPrice'].pct_change()*100
fig = go.Figure()
fig.add_trace(go.Bar(x=data['timestamp'], y=data['CE_pct'],
                     marker_color=np.where(data['CE_pct']>0,'green','red')))
fig.update_layout(title="CE % Change", showlegend=False)
st.plotly_chart(fig,use_container_width=True)

# ----------  CORRELATION & CONVICTION ----------
st.header("💪 Correlation & Volume Conviction")
data['vol_diff'] = data['CE_totalTradedVolume'].diff()
data['r_price_vol'] = data['CE_delta'].rolling(20).corr(data['vol_diff'])
st.line_chart(data[['r_price_vol']])
r_last = data['r_price_vol'].iloc[-1]
st.metric("r (Price, Volume)", round(r_last,3))

# ----------  SIDE DOMINANCE ----------
st.header("⚖️ Call vs Put Sentiment")
r_call = data['CE_lastPrice'].rolling(20).corr(data['CE_openInterest'])
r_put = data['PE_lastPrice'].rolling(20).corr(data['PE_openInterest'])
oi_imb = (data['CE_openInterest']-data['PE_openInterest'])/(data['CE_openInterest']+data['PE_openInterest'])
pc_ratio = data['PE_totalTradedVolume']/data['CE_totalTradedVolume']
bias = np.where(oi_imb>0,"🟩 Bullish","🟥 Bearish")
st.metric("Current Bias", bias[-1])
fig = px.line(x=data['timestamp'], y=oi_imb, title="OI Imbalance (+CE pressure)")
st.plotly_chart(fig,use_container_width=True)

# ----------  STRENGTH SCORE ----------
st.header("🧮 Overall Strength Score")
r_price_OI = data['CE_lastPrice'].rolling(20).corr(data['CE_openInterest'])
strength = 0.4*r_price_OI + 0.3*r_last + 0.3*r_call
data['strength'] = strength
st.line_chart(data[['strength']])
st.metric("Trend Health", round(data['strength'].dropna().iloc[-1],3))

# ----------  LEAD‑LAG ANALYSIS ----------
st.header("🧠 Lead–Lag Correlation (OI→Price)")
lags = range(-3,4)
results=[]
for l in lags:
    results.append(data['CE_lastPrice'].corr(data['CE_openInterest'].shift(l)))
leadlag = pd.DataFrame({'lag':lags,'corr':results})
fig = px.bar(leadlag,x='lag',y='corr',title="CE Price ↔ OI Lag Corr")
st.plotly_chart(fig,use_container_width=True)
bestlag = leadlag.iloc[leadlag['corr'].idxmax()]['lag']
st.metric("Leading Lag", bestlag)

# ----------  ROLLING REGIME / ANOMALY ----------
st.header("🔮 Regime & Anomaly Detection")
rollcorr = data['CE_lastPrice'].rolling(20).corr(data['CE_openInterest'])
data['signflip'] = np.sign(rollcorr).diff()
flips = data[data['signflip']!=0]
regime = "Bullish" if rollcorr.iloc[-1]>0 else "Bearish"
st.metric("Current Regime", regime)
fig = px.line(data, x='timestamp', y=rollcorr, title="Rolling Price–OI Correlation")
fig.add_scatter(x=flips['timestamp'], y=flips['signflip']*0, mode='markers', name='Sign flip', marker_color='orange')
st.plotly_chart(fig,use_container_width=True)

# ----------  VISUAL DASHBOARD ----------
st.header("🧭 Visualization Dashboard")
cols=['CE_lastPrice','CE_openInterest','CE_totalTradedVolume','PE_lastPrice','PE_openInterest']
corr=df[cols].corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
st.plotly_chart(fig,use_container_width=True)
scatter = px.scatter(data, x=data['CE_changeinOpenInterest'], y=data['CE_delta'],
                     color=np.where(data['CE_delta']>0,'green','red'),
                     title="ΔPrice vs ΔOI Quadrant (Strength map)")
st.plotly_chart(scatter,use_container_width=True)

# ----------  ML / ADVANCED PLACEHOLDER ----------
st.header("🤖 Advanced Stat / ML Modules (Placeholders)")
st.write("""
- Dynamic Conditional Correlation, cointegration, Bayesian Impact – add via `arch`, `statsmodels`, `bayesian_changepoint_detection`.
- Feature importances / forecasts – fit XGBoost or LSTM on (price, OI, vol) features.
""")

st.success("✅ All core / intermediate / advanced features scaffolded successfully")

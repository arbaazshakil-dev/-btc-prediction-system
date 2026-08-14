import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KRAKEN = "https://api.kraken.com/0/public"
FEATURES = ["r1","r3","r5","ema_gap","rsi","vol_z","range_pct","body_pct"]

st.set_page_config(page_title="BTC Signal Pro", page_icon="₿", layout="centered", initial_sidebar_state="collapsed")

def api(endpoint, params=None):
    r = requests.get(f"{KRAKEN}/{endpoint}", params=params or {}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data["result"]

@st.cache_data(ttl=10)
def get_ohlc(pair="XBTUSD"):
    result = api("OHLC", {"pair": pair, "interval": 1})
    key = [k for k in result if k != "last"][0]
    df = pd.DataFrame(result[key], columns=["time","open","high","low","close","vwap","volume","count"])
    for c in ["open","high","low","close","vwap","volume","count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna()

@st.cache_data(ttl=5)
def get_ticker(pair="XBTUSD"):
    result = api("Ticker", {"pair": pair})
    key = list(result.keys())[0]
    return float(result[key]["c"][0])

@st.cache_data(ttl=5)
def get_book(pair="XBTUSD"):
    result = api("Depth", {"pair": pair, "count": 100})
    key = list(result.keys())[0]
    book = result[key]
    bids = sum(float(x[0]) * float(x[1]) for x in book["bids"])
    asks = sum(float(x[0]) * float(x[1]) for x in book["asks"])
    imb = (bids - asks) / (bids + asks) if bids + asks else 0
    return imb, bids, asks

def features(df):
    x = df.copy()
    x["r1"] = x.close.pct_change()
    x["r3"] = x.close.pct_change(3)
    x["r5"] = x.close.pct_change(5)
    e5 = x.close.ewm(span=5).mean()
    e20 = x.close.ewm(span=20).mean()
    x["ema_gap"] = e5 / e20 - 1
    d = x.close.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = -d.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    x["vol_z"] = (x.volume - x.volume.rolling(30).mean()) / x.volume.rolling(30).std()
    return x.replace([np.inf, -np.inf], np.nan)

def train(x, horizon):
    future = x.close.shift(-horizon) / x.close - 1
    mask = x[FEATURES].notna().all(axis=1) & future.notna()
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(x.loc[mask, FEATURES], (future.loc[mask] > 0).astype(int))
    return model

st.title("₿ BTC Short-Term Signal Pro")
st.caption("Phone dashboard • research signal, not a guaranteed prediction")

with st.sidebar:
    pair = st.text_input("Kraken pair", "XBTUSD").upper()
    horizon = st.slider("Prediction window (minutes)", 1, 30, 8)
    refresh = st.slider("Refresh seconds", 10, 60, 20)

try:
    df = get_ohlc(pair)
    x = features(df)
    model = train(x, horizon)
    price = get_ticker(pair)
    imb, bids, asks = get_book(pair)

    p_up = float(model.predict_proba(x[FEATURES].iloc[[-1]])[0, 1])
    p_up = float(np.clip(p_up + np.clip(imb * 0.08, -0.08, 0.08), 0.01, 0.99))
    direction = "UP" if p_up >= 0.5 else "DOWN"
    confidence = max(p_up, 1 - p_up)

    st.metric("CURRENT CALL", f"{direction} — {confidence*100:.1f}%")
    a, b = st.columns(2)
    a.metric("BTC", f"${price:,.2f}")
    b.metric("Order Book", f"{imb*100:+.2f}%")

    st.write(f"**Horizon:** {horizon} minutes")
    st.write(f"**Momentum:** {'Bullish' if x.ema_gap.iloc[-1] > 0 else 'Bearish'}")
    st.write(f"**RSI:** {x.rsi.iloc[-1]:.1f}")
    st.write(f"**Volume Z-score:** {x.vol_z.iloc[-1]:+.2f}")
    st.write(f"**Bid depth:** ${bids:,.0f}")
    st.write(f"**Ask depth:** ${asks:,.0f}")

    st.line_chart(df.tail(180).set_index("time")["close"], height=260)
    st.caption("Data source: Kraken public REST API")
    st.caption("Confidence is model output, not a guaranteed chance of profit.")

except Exception as e:
    st.error(f"Market data error: {e}")

time.sleep(refresh)
st.rerun()

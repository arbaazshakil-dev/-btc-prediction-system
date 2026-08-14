
import time, requests, numpy as np, pandas as pd, streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://fapi.binance.com"
FEATURES = ["r1","r3","r5","ema_gap","rsi","vol_z","taker_ratio"]

st.set_page_config(page_title="BTC Signal", page_icon="₿", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding:1rem .8rem 2rem;max-width:700px}
h1{font-size:1.7rem}
.signal{border-radius:22px;padding:22px;text-align:center;margin:8px 0 14px}
.up{background:#0d2b1b;border:2px solid #35c46a}
.down{background:#321414;border:2px solid #ef6262}
.call{font-size:2.8rem;font-weight:800;margin:0}
.conf{font-size:1.4rem;font-weight:700;margin-top:5px}
.card{padding:14px;border-radius:16px;background:#17243b;margin:8px 0}
.small{opacity:.75;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=10)
def klines(symbol="BTCUSDT", limit=1500):
    r = requests.get(
        f"{BASE}/fapi/v1/klines",
        params={"symbol":symbol,"interval":"1m","limit":limit},
        timeout=10)
    r.raise_for_status()
    cols=["open_time","open","high","low","close","volume","close_time",
          "quote_volume","trades","taker_base","taker_quote","ignore"]
    d=pd.DataFrame(r.json(),columns=cols)
    for c in ["open","high","low","close","volume","taker_base"]:
        d[c]=pd.to_numeric(d[c])
    d["time"]=pd.to_datetime(d.open_time,unit="ms",utc=True)
    return d

@st.cache_data(ttl=5)
def market(symbol="BTCUSDT"):
    book=requests.get(
        f"{BASE}/fapi/v1/depth",
        params={"symbol":symbol,"limit":100},timeout=10).json()
    bids=sum(float(p)*float(q) for p,q in book["bids"])
    asks=sum(float(p)*float(q) for p,q in book["asks"])
    imbalance=(bids-asks)/(bids+asks) if bids+asks else 0
    oi=float(requests.get(
        f"{BASE}/fapi/v1/openInterest",
        params={"symbol":symbol},timeout=10).json()["openInterest"])
    funding=float(requests.get(
        f"{BASE}/fapi/v1/premiumIndex",
        params={"symbol":symbol},timeout=10).json()["lastFundingRate"])
    return imbalance,bids,asks,oi,funding

def make_features(d):
    x=d.copy()
    x["r1"]=x.close.pct_change()
    x["r3"]=x.close.pct_change(3)
    x["r5"]=x.close.pct_change(5)
    e5=x.close.ewm(span=5).mean()
    e20=x.close.ewm(span=20).mean()
    x["ema_gap"]=e5/e20-1
    delta=x.close.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=-delta.clip(upper=0).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-100/(1+rs)
    x["vol_z"]=(x.volume-x.volume.rolling(30).mean())/x.volume.rolling(30).std()
    x["taker_ratio"]=x.taker_base/x.volume.replace(0,np.nan)
    return x.replace([np.inf,-np.inf],np.nan)

def train(x,horizon):
    future=x.close.shift(-horizon)/x.close-1
    mask=x[FEATURES].notna().all(axis=1) & future.notna()
    model=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=1000,class_weight="balanced"))
    ])
    model.fit(x.loc[mask,FEATURES],(future.loc[mask]>0).astype(int))
    return model

st.title("₿ BTC Short-Term Signal")
st.caption("Phone dashboard • research signal, not a guaranteed prediction")

with st.sidebar:
    symbol=st.text_input("Symbol","BTCUSDT").upper()
    horizon=st.slider("Prediction window",1,60,8)
    refresh=st.slider("Refresh seconds",5,60,15)
    st.write("Data: Binance Futures public market feed")

try:
    d=klines(symbol)
    x=make_features(d)
    model=train(x,horizon)
    imb,bids,asks,oi,funding=market(symbol)

    p=float(model.predict_proba(x[FEATURES].iloc[[-1]])[0,1])
    p=float(np.clip(p+np.clip(imb*0.10,-0.10,0.10),0.01,0.99))
    direction="UP" if p>=0.5 else "DOWN"
    conf=max(p,1-p)
    css="up" if direction=="UP" else "down"
    emoji="🟢" if direction=="UP" else "🔴"

    signal_html = """
    <div class="signal {css}">
      <div class="small">CURRENT CALL</div>
      <div class="call">{emoji} {direction}</div>
      <div class="conf">{conf:.1f}% model confidence</div>
      <div class="small">{horizon}-minute horizon</div>
    </div>
    """.format(css=css,emoji=emoji,direction=direction,
               conf=conf*100,horizon=horizon)
    st.markdown(signal_html,unsafe_allow_html=True)

    a,b=st.columns(2)
    a.metric("BTC",f"${d.close.iloc[-1]:,.2f}")
    b.metric("Order Book",f"{imb*100:+.2f}%")

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Market structure")
    st.write(f"**Momentum:** {'Bullish' if x.ema_gap.iloc[-1]>0 else 'Bearish'}")
    st.write(f"**RSI:** {x.rsi.iloc[-1]:.1f}")
    st.write(f"**Volume Z:** {x.vol_z.iloc[-1]:+.2f}")
    st.write(f"**Taker ratio:** {x.taker_ratio.iloc[-1]:.3f}")
    st.write(f"**Open interest:** {oi:,.2f} BTC")
    st.write(f"**Funding:** {funding*100:.5f}%")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Order-book depth")
    st.write(f"Bid depth: **${bids:,.0f}**")
    st.write(f"Ask depth: **${asks:,.0f}**")
    st.markdown('</div>',unsafe_allow_html=True)

    st.line_chart(d.tail(180).set_index("time")["close"],height=260)
    st.caption("Updated " + pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S UTC"))

except Exception as e:
    st.error(f"Market data error: {e}")

time.sleep(refresh)
st.rerun()

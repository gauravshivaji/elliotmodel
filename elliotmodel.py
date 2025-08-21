import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ML imports (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Nifty500 Buy/Sell Predictor + Elliott Wave", layout="wide")
st.title("📊 Nifty500 Buy/Sell Predictor — Rules + Elliott Wave + ML")

# ---------------- TICKERS ----------------
NIFTY500_TICKERS = [
    "360ONE.NS","3MINDIA.NS","ABB.NS","TIPSMUSIC.NS","ACC.NS","ACMESOLAR.NS","AIAENG.NS","APLAPOLLO.NS","AUBANK.NS","AWL.NS","AADHARHFC.NS",
    "AARTIIND.NS","AAVAS.NS","ABBOTINDIA.NS","ACE.NS","ADANIENSOL.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ATGL.NS",
    "ABCAPITAL.NS","ABFRL.NS","ABREL.NS","ABSLAMC.NS","AEGISLOG.NS","AFCONS.NS","AFFLE.NS","AJANTPHARM.NS","AKUMS.NS","APLLTD.NS",
    "ALIVUS.NS","ALKEM.NS","ALKYLAMINE.NS","ALOKINDS.NS","ARE&M.NS","AMBER.NS","AMBUJACEM.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANGELONE.NS",
    "APARINDS.NS","APOLLOHOSP.NS","APOLLOTYRE.NS","APTUS.NS","ASAHIINDIA.NS","ASHOKLEY.NS","ASIANPAINT.NS","ASTERDM.NS","ASTRAZEN.NS","ASTRAL.NS",
    "ATUL.NS","AUROPHARMA.NS","AIIL.NS","DMART.NS","AXISBANK.NS","BASF.NS","BEML.NS","BLS.NS","BSE.NS","BAJAJ-AUTO.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","BAJAJHLDNG.NS","BAJAJHFL.NS","BALKRISIND.NS","BALRAMCHIN.NS","BANDHANBNK.NS","BANKBARODA.NS","BANKINDIA.NS","MAHABANK.NS",
    "BATAINDIA.NS","BAYERCROP.NS","BERGEPAINT.NS","BDL.NS","BEL.NS","BHARATFORG.NS","BHEL.NS","BPCL.NS","BHARTIARTL.NS","BHARTIHEXA.NS",
    "BIKAJI.NS","BIOCON.NS","BSOFT.NS","BLUEDART.NS","BLUESTARCO.NS","BBTC.NS","BOSCHLTD.NS","FIRSTCRY.NS","BRIGADE.NS","BRITANNIA.NS",
    "MAPMYINDIA.NS","CCL.NS","CESC.NS","CGPOWER.NS","CRISIL.NS","CAMPUS.NS","CANFINHOME.NS","CANBK.NS","CAPLIPOINT.NS","CGCL.NS",
    "CARBORUNIV.NS","CASTROLIND.NS","CEATLTD.NS","CENTRALBK.NS","CDSL.NS","CENTURYPLY.NS","CERA.NS","CHALET.NS","CHAMBLFERT.NS","CHENNPETRO.NS",
    "CHOLAHLDNG.NS","CHOLAFIN.NS","CIPLA.NS","CUB.NS","CLEAN.NS","COALINDIA.NS","COCHINSHIP.NS","COFORGE.NS","COHANCE.NS","COLPAL.NS",
    "CAMS.NS","CONCORDBIO.NS","CONCOR.NS","COROMANDEL.NS","CRAFTSMAN.NS","CREDITACC.NS","CROMPTON.NS","CUMMINSIND.NS","CYIENT.NS","DCMSHRIRAM.NS",
    "DLF.NS","DOMS.NS","DABUR.NS","DALBHARAT.NS","DATAPATTNS.NS","DEEPAKFERT.NS","DEEPAKNTR.NS","DELHIVERY.NS","DEVYANI.NS","DIVISLAB.NS",
    "DIXON.NS","LALPATHLAB.NS","DRREDDY.NS","DUMMYDBRLT.NS","EIDPARRY.NS","EIHOTEL.NS","EICHERMOT.NS","ELECON.NS","ELGIEQUIP.NS","EMAMILTD.NS",
    "EMCURE.NS","ENDURANCE.NS","ENGINERSIN.NS","ERIS.NS","ESCORTS.NS","ETERNAL.NS","EXIDEIND.NS","NYKAA.NS","FEDERALBNK.NS","FACT.NS",
    "FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FORTIS.NS","GAIL.NS","GVT&D.NS","GMRAIRPORT.NS","GRSE.NS","GICRE.NS",
    "GILLETTE.NS","GLAND.NS","GLAXO.NS","GLENMARK.NS","MEDANTA.NS","GODIGIT.NS","GPIL.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GODREJCP.NS",
    "GODREJIND.NS","GODREJPROP.NS","GRANULES.NS","GRAPHITE.NS","GRASIM.NS","GRAVITA.NS","GESHIP.NS","FLUOROCHEM.NS","GUJGASLTD.NS","GMDCLTD.NS",
    "GNFC.NS","GPPL.NS","GSPL.NS","HEG.NS","HBLENGINE.NS","HCLTECH.NS","HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HFCL.NS",
    "HAPPSTMNDS.NS","HAVELLS.NS","HEROMOTOCO.NS","HSCL.NS","HINDALCO.NS","HAL.NS","HINDCOPPER.NS","HINDPETRO.NS","HINDUNILVR.NS","HINDZINC.NS",
    "POWERINDIA.NS","HOMEFIRST.NS","HONASA.NS","HONAUT.NS","HUDCO.NS","HYUNDAI.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","IDBI.NS",
    "IDFCFIRSTB.NS","IFCI.NS","IIFL.NS","INOXINDIA.NS","IRB.NS","IRCON.NS","ITC.NS","ITI.NS","INDGN.NS","INDIACEM.NS",
    "INDIAMART.NS","INDIANB.NS","IEX.NS","INDHOTEL.NS","IOC.NS","IOB.NS","IRCTC.NS","IRFC.NS","IREDA.NS","IGL.NS",
    "INDUSTOWER.NS","INDUSINDBK.NS","NAUKRI.NS","INFY.NS","INOXWIND.NS","INTELLECT.NS","INDIGO.NS","IGIL.NS","IKS.NS","IPCALAB.NS",
    "JBCHEPHARM.NS","JKCEMENT.NS","JBMA.NS","JKTYRE.NS","JMFINANCIL.NS","JSWENERGY.NS","JSWHL.NS","JSWINFRA.NS","JSWSTEEL.NS","JPPOWER.NS",
    "J&KBANK.NS","JINDALSAW.NS","JSL.NS","JINDALSTEL.NS","JIOFIN.NS","JUBLFOOD.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JWL.NS","JUSTDIAL.NS",
    "JYOTHYLAB.NS","JYOTICNC.NS","KPRMILL.NS","KEI.NS","KNRCON.NS","KPITTECH.NS","KAJARIACER.NS","KPIL.NS","KALYANKJIL.NS","KANSAINER.NS",
    "KARURVYSYA.NS","KAYNES.NS","KEC.NS","KFINTECH.NS","KIRLOSBROS.NS","KIRLOSENG.NS","KOTAKBANK.NS","KIMS.NS","LTF.NS","LTTS.NS",
    "LICHSGFIN.NS","LTFOODS.NS","LTIM.NS","LT.NS","LATENTVIEW.NS","LAURUSLABS.NS","LEMONTREE.NS","LICI.NS","LINDEINDIA.NS","LLOYDSME.NS",
    "LODHA.NS","LUPIN.NS","MMTC.NS","MRF.NS","MGL.NS","MAHSEAMLES.NS","M&MFIN.NS","M&M.NS","MANAPPURAM.NS","MRPL.NS",
    "MANKIND.NS","MARICO.NS","MARUTI.NS","MASTEK.NS","MFSL.NS","MAXHEALTH.NS","MAZDOCK.NS","METROPOLIS.NS","MINDACORP.NS","MSUMI.NS",
    "MOTILALOFS.NS","MPHASIS.NS","MCX.NS","MUTHOOTFIN.NS","NATCOPHARM.NS","NBCC.NS","NCC.NS","NHPC.NS","NLCINDIA.NS","NMDC.NS",
    "NSLNISP.NS","NTPCGREEN.NS","NTPC.NS","NH.NS","NATIONALUM.NS","NAVA.NS","NAVINFLUOR.NS","NESTLEIND.NS","NETWEB.NS","NETWORK18.NS",
    "NEULANDLAB.NS","NEWGEN.NS","NAM-INDIA.NS","NIVABUPA.NS","NUVAMA.NS","OBEROIRLTY.NS","ONGC.NS","OIL.NS","OLAELEC.NS","OLECTRA.NS",
    "PAYTM.NS","OFSS.NS","POLICYBZR.NS","PCBL.NS","PGEL.NS","PIIND.NS","PNBHOUSING.NS","PNCINFRA.NS","PTCIL.NS","PVRINOX.NS",
    "PAGEIND.NS","PATANJALI.NS","PERSISTENT.NS","PETRONET.NS","PFIZER.NS","PHOENIXLTD.NS","PIDILITIND.NS","PEL.NS","PPLPHARMA.NS","POLYMED.NS",
    "POLYCAB.NS","POONAWALLA.NS","PFC.NS","POWERGRID.NS","PRAJIND.NS","PREMIERENE.NS","PRESTIGE.NS","PNB.NS","RRKABEL.NS","RBLBANK.NS",
    "RECLTD.NS","RHIM.NS","RITES.NS","RADICO.NS","RVNL.NS","RAILTEL.NS","RAINBOW.NS","RKFORGE.NS","RCF.NS","RTNINDIA.NS",
    "RAYMONDLSL.NS","RAYMOND.NS","REDINGTON.NS","RELIANCE.NS","RPOWER.NS","ROUTE.NS","SBFC.NS","SBICARD.NS","SBILIFE.NS","SJVN.NS",
    "SKFINDIA.NS","SRF.NS","SAGILITY.NS","SAILIFE.NS","SAMMAANCAP.NS","MOTHERSON.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SCHAEFFLER.NS",
    "SCHNEIDER.NS","SCI.NS","SHREECEM.NS","RENUKA.NS","SHRIRAMFIN.NS","SHYAMMETL.NS","SIEMENS.NS","SIGNATURE.NS","SOBHA.NS","SOLARINDS.NS",
    "SONACOMS.NS","SONATSOFTW.NS","STARHEALTH.NS","SBIN.NS","SAIL.NS","SWSOLAR.NS","SUMICHEM.NS","SUNPHARMA.NS","SUNTV.NS","SUNDARMFIN.NS",
    "SUNDRMFAST.NS","SUPREMEIND.NS","SUZLON.NS","SWANENERGY.NS","SWIGGY.NS","SYNGENE.NS","SYRMA.NS","TBOTEK.NS","TVSMOTOR.NS","TANLA.NS",
    "TATACHEM.NS","TATACOMM.NS","TCS.NS","TATACONSUM.NS","TATAELXSI.NS","TATAINVEST.NS","TATAMOTORS.NS","TATAPOWER.NS","TATASTEEL.NS","TATATECH.NS",
    "TTML.NS","TECHM.NS","TECHNOE.NS","TEJASNET.NS","NIACL.NS","RAMCOCEM.NS","THERMAX.NS","TIMKEN.NS","TITAGARH.NS","TITAN.NS",
    "TORNTPHARM.NS","TORNTPOWER.NS","TARIL.NS","TRENT.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TIINDIA.NS","UCOBANK.NS","UNOMINDA.NS",
    "UPL.NS","UTIAMC.NS","ULTRACEMCO.NS","UNIONBANK.NS","UBL.NS","UNITDSPR.NS","USHAMART.NS","VGUARD.NS","DBREALTY.NS","VTL.NS",
    "VBL.NS","MANYAVAR.NS","VEDL.NS","VIJAYA.NS","VMM.NS","IDEA.NS","VOLTAS.NS","WAAREEENER.NS","WELCORP.NS","WELSPUNLIV.NS",
    "WESTLIFE.NS","WHIRLPOOL.NS","WIPRO.NS","WOCKPHARMA.NS","YESBANK.NS","ZFCVINDIA.NS","ZEEL.NS","ZENTEC.NS","ZENSARTECH.NS","ZYDUSLIFE.NS",
    "ECLERX.NS",
]


# ---------------- UTIL ----------------
def add_tradingview_links(df):
    df = df.copy()
    if "Ticker" in df.columns:
        df["TradingView"] = df["Ticker"].apply(
            lambda t: f'<a href="https://www.tradingview.com/chart/?symbol=NSE:{t.replace(".NS","")}" target="_blank">📈 Chart</a>'
        )
    return df

class _TQDM:
    def __init__(self, total, desc=""):
        self.pb = st.progress(0, text=desc)
        self.total = max(total, 1)
        self.i = 0
    def update(self):
        self.i += 1
        self.pb.progress(min(self.i / self.total, 1.0), text=f"{self.i}/{self.total}")
    def close(self):
        self.pb.empty()

def stqdm(iterable, total=None, desc=""):
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = 100
    bar = _TQDM(total=total, desc=desc)
    for x in iterable:
        yield x
        bar.update()
    bar.close()

# ---------------- DATA DOWNLOAD ----------------
@st.cache_data(show_spinner=False)
def download_data_multi(tickers, period="2y", interval="1d"):
    if isinstance(tickers, str):
        tickers = [tickers]
    frames = []
    batch_size = 50
    for i in stqdm(range(0, len(tickers), batch_size), desc="Downloading", total=len(tickers)//batch_size + 1):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, period=period, interval=interval, group_by="ticker", progress=False, threads=True)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    if isinstance(out.columns, pd.MultiIndex):
        # drop duplicate multiindex columns that sometimes appear
        idx = pd.MultiIndex.from_tuples(list(dict.fromkeys(out.columns.tolist())))
        out = out.loc[:, idx]
    return out

@st.cache_data(show_spinner=False)
def load_history_for_ticker(ticker, period="5y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- ELLIOTT WAVE (ZigZag + Heuristics) ----------------
def zigzag_pivots(close: pd.Series, pct=0.05, min_bars=5):
    """
    Dependency-free ZigZag: pivots when price reverses by >= pct from last pivot and min_bars elapsed.
    Returns DataFrame with columns ['idx','price','type'] where type in {'H','L'}.
    """
    if close.isna().all() or len(close) < max(50, min_bars*4):
        return pd.DataFrame(columns=["idx", "price", "type"])

    c = close.values.astype(float)
    idxs = close.index

    piv = []  # tuples (index, price, 'H'/'L')

    last_piv_i = 0
    last_piv_p = c[0]
    trend = None  # 'up' or 'down'
    last_extreme_i = 0
    last_extreme_p = c[0]

    for i in range(1, len(c)):
        # track most extreme in current leg
        if trend in (None, 'up'):
            if c[i] > last_extreme_p:
                last_extreme_p = c[i]; last_extreme_i = i
        if trend in (None, 'down'):
            if c[i] < last_extreme_p:
                last_extreme_p = c[i]; last_extreme_i = i

        # reversal checks (percent from extreme and spacing)
        if trend in (None, 'up'):
            # reverse to down if drop from extreme exceeds pct
            if last_extreme_p != 0:
                dd = (c[i] - last_extreme_p) / last_extreme_p
            else:
                dd = 0
            if dd <= -pct and (i - last_piv_i) >= min_bars:
                # mark High pivot at last extreme
                piv.append((idxs[last_extreme_i], float(last_extreme_p), 'H'))
                last_piv_i = last_extreme_i; last_piv_p = last_extreme_p
                trend = 'down'
                last_extreme_i = i; last_extreme_p = c[i]
        if trend in (None, 'down'):
            if last_extreme_p != 0:
                uu = (c[i] - last_extreme_p) / last_extreme_p
            else:
                uu = 0
            if uu >= pct and (i - last_piv_i) >= min_bars:
                # mark Low pivot at last extreme
                piv.append((idxs[last_extreme_i], float(last_extreme_p), 'L'))
                last_piv_i = last_extreme_i; last_piv_p = last_extreme_p
                trend = 'up'
                last_extreme_i = i; last_extreme_p = c[i]

    # Ensure alternating types and keep more extreme if duplicates
    if len(piv) >= 2:
        cleaned = [piv[0]]
        for i in range(1, len(piv)):
            t_i = piv[i][2]
            t_prev = cleaned[-1][2]
            if t_i == t_prev:
                prev = cleaned[-1]
                if t_i == 'H':
                    better = piv[i][1] > prev[1]
                else:
                    better = piv[i][1] < prev[1]
                if better:
                    cleaned[-1] = piv[i]
            else:
                cleaned.append(piv[i])
        piv = cleaned

    if not piv:
        return pd.DataFrame(columns=["idx", "price", "type"])
    idx, price, typ = zip(*piv)
    return pd.DataFrame({"idx": list(idx), "price": list(price), "type": list(typ)})

def fib_okay(a, b, ratio, tol=0.18):
    """Check if |a/b - ratio| within tolerance (basic fib confluence)."""
    if b == 0 or np.isnan(a) or np.isnan(b):
        return False
    return abs((a / b) - ratio) <= tol * ratio

def elliott_phase_from_pivots(pivots: pd.DataFrame):
    """
    Compact heuristic:
    - Try to interpret last 5 alternating pivots as an impulse up/down.
    - Validate HH/HL (up) or LL/LH (down) + rough Fib relations for waves 2 and 4.
    - Fallback: last 3 pivots as ABC correction up/down.
    Returns dict: phase {'ImpulseUp','ImpulseDown','CorrectionUp','CorrectionDown','Unknown'},
                  wave_no (5 or 3), bullish, bearish.
    """
    out = {"phase": "Unknown", "wave_no": 0, "bullish": False, "bearish": False}
    if pivots.empty:
        return out

    # Try last 5
    if len(pivots) >= 5:
        p5 = pivots.iloc[-5:].reset_index(drop=True)
        alt = all(p5.loc[i, "type"] != p5.loc[i-1, "type"] for i in range(1, 5))
        if alt:
            prices = p5["price"].values
            types = p5["type"].values
            up_pattern = (types.tolist() == ['L','H','L','H','L'])
            down_pattern = (types.tolist() == ['H','L','H','L','H'])
            if up_pattern:
                hh_ok = prices[3] > prices[1]
                hl_ok = prices[4] > prices[2]
                w1 = prices[1] - prices[0]
                w2 = prices[1] - prices[2]
                w3 = prices[3] - prices[2]
                w4 = prices[3] - prices[4]
                fib2 = fib_okay(w2, w1, 0.382) or fib_okay(w2, w1, 0.5) or fib_okay(w2, w1, 0.618)
                fib4 = fib_okay(w4, w3, 0.382) or fib_okay(w4, w3, 0.5) or fib_okay(w4, w3, 0.618)
                if hh_ok and hl_ok and (fib2 or fib4):
                    out.update({"phase": "ImpulseUp", "wave_no": 5, "bullish": True})
                    return out
            if down_pattern:
                ll_ok = prices[3] < prices[1]
                lh_ok = prices[4] < prices[2]
                w1 = prices[0] - prices[1]
                w2 = prices[2] - prices[1]
                w3 = prices[2] - prices[3]
                w4 = prices[4] - prices[3]
                fib2 = fib_okay(w2, w1, 0.382) or fib_okay(w2, w1, 0.5) or fib_okay(w2, w1, 0.618)
                fib4 = fib_okay(w4, w3, 0.382) or fib_okay(w4, w3, 0.5) or fib_okay(w4, w3, 0.618)
                if ll_ok and lh_ok and (fib2 or fib4):
                    out.update({"phase": "ImpulseDown", "wave_no": 5, "bearish": True})
                    return out

    # Fallback: last 3 (ABC)
    if len(pivots) >= 3:
        p3 = pivots.iloc[-3:].reset_index(drop=True)
        alt3 = all(p3.loc[i, "type"] != p3.loc[i-1, "type"] for i in range(1, 3))
        if alt3:
            t = p3["type"].tolist()
            if t == ['L','H','L']:
                out.update({"phase": "CorrectionUp", "wave_no": 3, "bullish": True})
            elif t == ['H','L','H']:
                out.update({"phase": "CorrectionDown", "wave_no": 3, "bearish": True})
    return out

def add_elliott_features_core(df_close: pd.Series, pct=0.05, min_bars=5):
    piv = zigzag_pivots(df_close, pct=pct, min_bars=min_bars)
    phase = elliott_phase_from_pivots(piv)
    return phase, piv

# ---------------- FEATURE ENGINEERING ----------------
def compute_features(df, sma_windows=(20, 50, 200), support_window=30, zz_pct=0.05, zz_min_bars=5):
    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.DataFrame()

    df = df.copy()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    # SMAs
    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()

    # Support (simple rolling min)
    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    # Divergences (very lightweight)
    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)

    # Returns and distances
    for w in (1, 3, 5, 10):
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    for win in sma_windows:
        df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]

    # Slopes
    for col in ["RSI"] + [f"SMA{w}" for w in sma_windows]:
        df[f"{col}_slope"] = df[col].diff()

    # --- Elliott features ---
    try:
        phase, piv = add_elliott_features_core(df["Close"], pct=zz_pct, min_bars=zz_min_bars)
        phase_map = {
            "ImpulseUp": 1, "ImpulseDown": -1,
            "CorrectionUp": 2, "CorrectionDown": -2,
            "Unknown": 0
        }
        df["Elliott_Phase_Code"] = phase_map.get(phase["phase"], 0)
        df["Elliott_Wave_No"] = int(phase.get("wave_no", 0))
        df["Elliott_Bullish"] = bool(phase.get("bullish", False))
        df["Elliott_Bearish"] = bool(phase.get("bearish", False))
        # numeric versions for ML selection
        df["Elliott_Bullish_Int"] = df["Elliott_Bullish"].astype(int)
        df["Elliott_Bearish_Int"] = df["Elliott_Bearish"].astype(int)
    except Exception:
        df["Elliott_Phase_Code"] = 0
        df["Elliott_Wave_No"] = 0
        df["Elliott_Bullish"] = False
        df["Elliott_Bearish"] = False
        df["Elliott_Bullish_Int"] = 0
        df["Elliott_Bearish_Int"] = 0

    return df




def get_latest_features_for_ticker(ticker_df, ticker, sma_windows, support_window, zz_pct, zz_min_bars):
    df = compute_features(ticker_df, sma_windows, support_window, zz_pct, zz_min_bars).dropna()
    if df.empty:
        return None
    latest = df.iloc[-1]
    return {
        "Ticker": ticker,
        "Close": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "Support": float(latest["Support"]),
        **{f"SMA{w}": float(latest.get(f"SMA{w}", np.nan)) for w in sma_windows},
        "Bullish_Div": bool(latest["Bullish_Div"]),
        "Bearish_Div": bool(latest["Bearish_Div"]),
        "Elliott_Phase_Code": int(latest["Elliott_Phase_Code"]) if "Elliott_Phase_Code" in latest else 0,
        "Elliott_Wave_No": int(latest["Elliott_Wave_No"]) if "Elliott_Wave_No" in latest else 0,
        "Elliott_Bullish_Int": int(latest["Elliott_Bullish_Int"]) if "Elliott_Bullish_Int" in latest else 0,
        "Elliott_Bearish_Int": int(latest["Elliott_Bearish_Int"]) if "Elliott_Bearish_Int" in latest else 0,

        
    }

def get_features_for_all(tickers, sma_windows, support_window, zz_pct, zz_min_bars):
    multi_df = download_data_multi(tickers)
    if multi_df is None or multi_df.empty:
        return pd.DataFrame()

    features_list = []
    if isinstance(multi_df.columns, pd.MultiIndex):
        available = multi_df.columns.get_level_values(0).unique()
        for ticker in tickers:
            if ticker not in available:
                continue
            tdf = multi_df[ticker].dropna()
            if tdf.empty:
                continue
            feats = get_latest_features_for_ticker(tdf, ticker, sma_windows, support_window, zz_pct, zz_min_bars)
            if feats:
                features_list.append(feats)
    else:
        feats = get_latest_features_for_ticker(multi_df.dropna(), tickers[0], sma_windows, support_window, zz_pct, zz_min_bars)
        if feats:
            features_list.append(feats)
    return pd.DataFrame(features_list)

# ---------------- RULE-BASED STRATEGY (+ Elliott) ----------------
def predict_buy_sell_rule(df, rsi_buy=30, rsi_sell=70):
    if df.empty:
        return df
    results = df.copy()

    # --- Core patterns (relaxed thresholds) ---
    reversal_buy_core = (
        (results["RSI"] < rsi_buy) &
        (results.get("Bullish_Div", True)) &  # default True if missing
        (np.abs(results["Close"] - results.get("Support", results["Close"])) < 0.1 * results["Close"]) &
        (results["Close"] > results["SMA20"])
    )

    trend_buy_core = (
        (results["Close"] > results["SMA20"]) &
        (results["SMA20"] > results["SMA50"]) &
        (results["RSI"] > 40)  # lower threshold
    )

    base_sell_core = (
        ((results["RSI"] > rsi_sell) & (results.get("Bearish_Div", True))) |
        (results["Close"] < results.get("Support", results["Close"])) |
        ((results["SMA20"] < results["SMA50"]) & (results["SMA50"] < results["SMA200"]))
    )

    # --- Elliott confirmations (safe defaults) ---
    ew_bull = (results.get("Elliott_Bullish_Int", 0) == 1) | (results.get("Elliott_Phase_Code", 0) == 1)
    ew_bear = (results.get("Elliott_Bearish_Int", 0) == 1) | (results.get("Elliott_Phase_Code", 0) == -1)

    # --- Buy/Sell refinements ---
    results["Reversal_Buy"] = reversal_buy_core | ew_bull
    results["Trend_Buy"] = trend_buy_core |ew_bull

    ew_only_buy = (
        ew_bull &
        (results["RSI"].between(35, 65)) &  # much wider zone
        (results["Close"] > results["SMA20"])
    )

    ew_only_sell = (
        ew_bear &
        (results["RSI"] > 50)  # less strict
    )

    # --- Final signals ---
    results["Sell_Point"]  = results["Reversal_Buy"] | results["Trend_Buy"] | ew_only_buy
    results["Buy_Point"] = base_sell_core | ew_only_sell

    return results



# ---------------- LABELS FOR ML ----------------
def label_from_rule_based(df, rsi_buy=30, rsi_sell=70):
    rules = predict_buy_sell_rule(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    label = pd.Series(0, index=rules.index, dtype=int)
    label[rules["Buy_Point"]] = 1
    label[rules["Sell_Point"]] = -1
    return label

def label_from_future_returns(df, horizon=60, buy_thr=0.03, sell_thr=-0.03):
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = pd.Series(0, index=df.index, dtype=int)
    label[fut_ret >= buy_thr] = 1
    label[fut_ret <= sell_thr] = -1
    return label

# ---------------- ML DATASET ----------------
def build_ml_dataset_for_tickers(
    tickers, sma_windows, support_window,
    label_mode="rule",
    horizon=60, buy_thr=0.03, sell_thr=-0.03,
    rsi_buy=30, rsi_sell=70,
    min_rows=250,
    zz_pct=0.05, zz_min_bars=5
):
    X_list, y_list, meta_list = [], [], []
    feature_cols = None

    for t in stqdm(tickers, desc="Preparing ML data"):
        hist = load_history_for_ticker(t, period="5y", interval="1d")
        if hist is None or hist.empty or len(hist) < min_rows:
            continue

        feat = compute_features(hist, sma_windows, support_window, zz_pct, zz_min_bars)
        if feat.empty:
            continue

        if label_mode == "rule":
            y = label_from_rule_based(feat, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
        else:
            y = label_from_future_returns(feat, horizon=horizon, buy_thr=buy_thr, sell_thr=sell_thr)

        data = feat.join(y.rename("Label")).dropna()
        if data.empty:
            continue

        # Ensure EW ints are present (they are numeric already)
        drop_cols = set(["Label", "Support", "Bullish_Div", "Bearish_Div"])
        use = data.select_dtypes(include=[np.number]).drop(columns=list(drop_cols & set(data.columns)), errors="ignore")

        if feature_cols is None:
            feature_cols = list(use.columns)

        X_list.append(use[feature_cols])
        y_list.append(data["Label"])
        meta_list.append(pd.Series([t] * len(use), index=use.index, name="Ticker"))

    if not X_list:
        return pd.DataFrame(), pd.Series(dtype=int), [], []

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)
    tickers_series = pd.concat(meta_list, axis=0)
    return X, y, feature_cols, tickers_series

def train_rf_classifier(X, y, random_state=42):
    if X.empty or y.empty:
        return None, None, None
    stratify_opt = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=stratify_opt, random_state=random_state
        )
    except Exception:
        X_train, X_test, y_train, y_test = X.iloc[:-200], X.iloc[-200:], y.iloc[:-200], y.iloc[-200:]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    return clf, acc, report

def latest_feature_row_for_ticker(ticker, sma_windows, support_window, feature_cols, zz_pct, zz_min_bars):
    hist = load_history_for_ticker(ticker, period="3y", interval="1d")
    if hist is None or hist.empty:
        return None
    feat = compute_features(hist, sma_windows, support_window, zz_pct, zz_min_bars).dropna()
    if feat.empty:
        return None
    use = feat.select_dtypes(include=[np.number])
    row = use.iloc[-1:].copy()
    for m in [c for c in feature_cols if c not in row.columns]:
        row[m] = 0.0
    row = row[feature_cols]
    return row

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")

    select_all = st.checkbox("Select all stocks", value=True)
    default_list = NIFTY500_TICKERS if select_all else NIFTY500_TICKERS[:25]
    selected_tickers = st.multiselect("Select stocks", NIFTY500_TICKERS, default=default_list)

    sma_w1 = st.number_input("SMA Window 1", 5, 250, 20)
    sma_w2 = st.number_input("SMA Window 2", 5, 250, 50)
    sma_w3 = st.number_input("SMA Window 3", 5, 250, 200)
    support_window = st.number_input("Support Period (days)", 5, 200, 30)

    st.markdown("---")
    st.subheader("Elliott (ZigZag) Tuning")
    zz_pct = st.slider("ZigZag reversal (%)", 2, 12, 5, help="Sensitivity for swing detection (5% = 0.05).") / 100.0
    zz_min_bars = st.slider("Min bars between pivots", 3, 15, 5)

    st.markdown("---")
    label_mode = st.radio("ML Labeling Mode", ["Rule-based (teach the rules)", "Future Returns"], index=0)

    if label_mode == "Rule-based (teach the rules)":
        st.subheader("Rule thresholds (also used to generate ML labels)")
        rsi_buy_lbl = st.slider("RSI Buy Threshold", 5, 50, 30)
        rsi_sell_lbl = st.slider("RSI Sell Threshold", 50, 95, 70)
        rsi_buy = rsi_buy_lbl
        rsi_sell = rsi_sell_lbl
        ml_horizon, ml_buy_thr, ml_sell_thr = 60, 0.03, -0.03
    else:
        st.subheader("Rule thresholds (for live rule signals only)")
        rsi_buy = st.slider("RSI Buy Threshold", 5, 50, 30)
        rsi_sell = st.slider("RSI Sell Threshold", 50, 95, 70)
        st.subheader("ML labeling (future return)")
        ml_horizon = st.number_input("Horizon (days ahead)", 2, 60, 5)
        ml_buy_thr = st.number_input("Buy threshold (e.g., 0.03 = +3%)", 0.005, 0.20, 0.03, step=0.005, format="%.3f")
        ml_sell_thr = st.number_input("Sell threshold (e.g., -0.03 = -3%)", -0.20, -0.005, -0.03, step=0.005, format="%.3f")

    run_analysis = st.button("Run Analysis")

# ---------------- MAIN ----------------
if run_analysis:
    sma_tuple = (sma_w1, sma_w2, sma_w3)

    with st.spinner("Fetching data & computing rule-based + Elliott features..."):
        feats = get_features_for_all(selected_tickers, sma_tuple, support_window, zz_pct, zz_min_bars)
        if feats is None or feats.empty:
            st.error("No valid data for selected tickers.")
        else:
            preds_rule = predict_buy_sell_rule(feats, rsi_buy, rsi_sell)

    tab1, tab2, tab3, tab4 = st.tabs([
        "✅ Rule Buy (current snapshot)",
        "❌ Rule Sell (current snapshot)",
        "📈 Chart",
        "🤖 ML Signals"
    ])

    with tab1:
        if feats.empty:
            st.info("No rule-based buy signals.")
        else:
            df_buy = preds_rule[preds_rule["Buy_Point"]].copy()
            # add TradingView link next to ticker
            df_buy["TradingView"] = df_buy["Ticker"].apply(lambda x: f'<a href="https://in.tradingview.com/chart/?symbol=NSE%3A{x.replace(".NS","")}" target="_blank">📈 Chart</a>')
            # Recent useful columns
            show_cols = ["Ticker","TradingView","Close","RSI","Reversal_Buy","Trend_Buy"]
            cols = [c for c in show_cols if c in df_buy.columns] + [c for c in df_buy.columns if c not in show_cols]
            st.write(df_buy[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab2:
        if feats.empty:
            st.info("No rule-based sell signals.")
        else:
            df_sell = preds_rule[preds_rule["Sell_Point"]].copy()
            df_sell["TradingView"] = df_sell["Ticker"].apply(lambda x: f'<a href="https://in.tradingview.com/chart/?symbol=NSE%3A{x.replace(".NS","")}" target="_blank">📈 Chart</a>')
            show_cols = ["Ticker","TradingView","Close","RSI"]
            cols = [c for c in show_cols if c in df_sell.columns] + [c for c in df_sell.columns if c not in show_cols]
            st.write(df_sell[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab3:
        ticker_for_chart = st.selectbox("Chart Ticker", selected_tickers)
        chart_df = yf.download(ticker_for_chart, period="6mo", interval="1d", progress=False, threads=True)
        if not chart_df.empty:
            chart_df = compute_features(chart_df, sma_tuple, support_window, zz_pct, zz_min_bars).dropna()
            if not chart_df.empty:
                st.line_chart(chart_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
                st.line_chart(chart_df[["RSI"]])

                latest = chart_df.iloc[-1]
                phase_code = int(latest.get("Elliott_Phase_Code", 0))
                phase_text = {1:"ImpulseUp", -1:"ImpulseDown", 2:"CorrectionUp", -2:"CorrectionDown", 0:"Unknown"}.get(phase_code, "Unknown")
                wave_no = int(latest.get("Elliott_Wave_No", 0))
                st.caption(f"🌀 Elliott Phase: **{phase_text}**  |  Wave#: **{wave_no}**  |  ZigZag: {zz_pct*100:.1f}% / {zz_min_bars} bars")
        else:
            st.warning("No chart data available.")

    with tab4:
        if not SKLEARN_OK:
            st.error("scikit-learn not available. Install with: pip install scikit-learn")
        else:
            with st.spinner("Building ML dataset & training model..."):
                if label_mode == "Rule-based (teach the rules)":
                    X, y, feature_cols, tickers_series = build_ml_dataset_for_tickers(
                        selected_tickers, sma_tuple, support_window,
                        label_mode="rule", rsi_buy=rsi_buy, rsi_sell=rsi_sell,
                        zz_pct=zz_pct, zz_min_bars=zz_min_bars
                    )
                else:
                    X, y, feature_cols, tickers_series = build_ml_dataset_for_tickers(
                        selected_tickers, sma_tuple, support_window,
                        label_mode="future", horizon=ml_horizon, buy_thr=ml_buy_thr, sell_thr=ml_sell_thr,
                        zz_pct=zz_pct, zz_min_bars=zz_min_bars
                    )

                if X.empty or y.empty:
                    st.warning("Not enough historical data to train the ML model for the chosen settings.")
                else:
                    clf, acc, report = train_rf_classifier(X, y)
                    st.caption(f"Validation accuracy (holdout): **{acc:.3f}**")
                    with st.expander("Classification report"):
                        st.text(report)

                    rows = []
                    for t in stqdm(selected_tickers, desc="Scoring", total=len(selected_tickers)):
                        row = latest_feature_row_for_ticker(t, sma_tuple, support_window, feature_cols, zz_pct, zz_min_bars)
                        if row is None:
                            continue
                        proba = clf.predict_proba(row)[0] if hasattr(clf, "predict_proba") else None
                        pred = clf.predict(row)
                        rows.append({
                            "Ticker": t,
                            "ML_Pred": {1: "BUY", 0: "HOLD", -1: "SELL"}.get(int(pred), "HOLD"),
                            "Prob_Buy": float(proba[list(clf.classes_).index(1)]) if proba is not None and 1 in clf.classes_ else np.nan,
                            "Prob_Hold": float(proba[list(clf.classes_).index(0)]) if proba is not None and 0 in clf.classes_ else np.nan,
                            "Prob_Sell": float(proba[list(clf.classes_).index(-1)]) if proba is not None and -1 in clf.classes_ else np.nan,
                        })
                    ml_df = pd.DataFrame(rows).sort_values(["ML_Pred", "Prob_Buy"], ascending=[True, False])

                    def tradingview_link(ticker):
                        return f"https://in.tradingview.com/chart/?symbol=NSE%3A{ticker.replace('.NS','')}"
                    ml_df["TradingView"] = ml_df["Ticker"].apply(tradingview_link)

                    st.dataframe(
                        ml_df,
                        use_container_width=True,
                        column_config={
                            "TradingView": st.column_config.LinkColumn(
                                "TradingView",
                                display_text="📈 Chart"
                            )
                        }
                    )

    if 'preds_rule' in locals() and preds_rule is not None and not preds_rule.empty:
        st.download_button(
            "📥 Download Rule-based Results (snapshot)",
            preds_rule.to_csv(index=False).encode(),
            "nifty500_rule_signals.csv",
            "text/csv",
        )

st.markdown("⚠ Educational use only — not financial advice.")











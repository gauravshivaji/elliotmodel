import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

def add_tradingview_links(df):
    df = df.copy()
    if "Ticker" in df.columns:
        df["TradingView"] = df["Ticker"].apply(
            lambda t: f'<a href="https://www.tradingview.com/chart/?symbol=NSE:{t.replace(".NS","")}" target="_blank">📈 Chart</a>'
        )
    return df

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Nifty500 Buy/Sell Predictor", layout="wide")
st.title("📊 Nifty500 Buy/Sell Predictor — Rules vs ML")

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

# ---------------- HELPERS ----------------

@st.cache_data(show_spinner=False)
def download_data_multi(tickers, period="2y", interval="1d"):
    """Batch download to be kinder to Yahoo and reduce failures."""
    if isinstance(tickers, str):
        tickers = [tickers]
    frames = []
    batch_size = 50  # chunks to avoid request overload
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
    # merge along columns (MultiIndex preserved)
    out = pd.concat(frames, axis=1)
    # de-duplicate top-level tickers if concat overlapped
    if isinstance(out.columns, pd.MultiIndex):
        idx = pd.MultiIndex.from_tuples(list(dict.fromkeys(out.columns.tolist())))
        out = out.loc[:, idx]
    return out

def compute_features(df, sma_windows=(20, 50, 200), support_window=30):
    # Flatten MultiIndex if needed (for single-ticker computations downstream)
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

    # Support (rolling minimum)
    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    # Divergences
    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)

    # Returns for ML
    for w in (1, 3, 5, 10):
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    # Distance from SMAs
    for win in sma_windows:
        df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]

    # Slopes
    for col in ["RSI"] + [f"SMA{w}" for w in sma_windows]:
        df[f"{col}_slope"] = df[col].diff()

    return df

def get_latest_features_for_ticker(ticker_df, ticker, sma_windows, support_window):
    df = compute_features(ticker_df, sma_windows, support_window).dropna()
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
    }

# ----------- Elliott Wave additions ------------

def compute_zigzag(df, threshold=5):
    """
    Simple zigzag implementation:
    Marks pivots in price where % change exceeds threshold.
    Returns a series with pivot prices and NaN elsewhere.
    """
    close = df["Close"]
    zigzag = [np.nan] * len(df)

    last_pivot = close.iloc[0]
    last_pivot_idx = 0
    trend = None  # "up" or "down"

    for i in range(1, len(df)):
        change_pct = (close.iloc[i] - last_pivot) / last_pivot * 100

        if trend is None:
            if abs(change_pct) >= threshold:
                trend = "up" if change_pct > 0 else "down"
                last_pivot = close.iloc[i]
                last_pivot_idx = i
                zigzag[last_pivot_idx] = last_pivot
        elif trend == "up":
            if close.iloc[i] > last_pivot:
                last_pivot = close.iloc[i]
                last_pivot_idx = i
                zigzag[last_pivot_idx] = last_pivot
            elif (last_pivot - close.iloc[i]) / last_pivot * 100 >= threshold:
                trend = "down"
                last_pivot = close.iloc[i]
                last_pivot_idx = i
                zigzag[last_pivot_idx] = last_pivot
        elif trend == "down":
            if close.iloc[i] < last_pivot:
                last_pivot = close.iloc[i]
                last_pivot_idx = i
                zigzag[last_pivot_idx] = last_pivot
            elif (close.iloc[i] - last_pivot) / last_pivot * 100 >= threshold:
                trend = "up"
                last_pivot = close.iloc[i]
                last_pivot_idx = i
                zigzag[last_pivot_idx] = last_pivot

    return pd.Series(zigzag, index=df.index)

def predict_elliott_wave(df, zigzag_threshold=5):
    """
    Basic Elliott Wave-inspired buy/sell signals using zigzag pivots.
    Buy near pivot lows after corrective waves (wave 2 or 4).
    Sell near pivot highs after impulse wave 5.
    This is a simplified heuristic.
    """

    zz = compute_zigzag(df, threshold=zigzag_threshold)
    pivots = zz.dropna()
    signals_buy = pd.Series(False, index=df.index)
    signals_sell = pd.Series(False, index=df.index)

    pivot_points = pivots.values
    pivot_indices = pivots.index

    lows = []
    highs = []

    for i in range(1, len(pivot_points) - 1):
        prev_p = pivot_points[i-1]
        curr = pivot_points[i]
        next_p = pivot_points[i+1]
        idx = pivot_indices[i]

        if curr < prev_p and curr < next_p:
            lows.append(idx)
        elif curr > prev_p and curr > next_p:
            highs.append(idx)

    for low_idx in lows:
        if low_idx in signals_buy.index:
            signals_buy.loc[low_idx] = True

    for high_idx in highs:
        if high_idx in signals_sell.index:
            signals_sell.loc[high_idx] = True

    return signals_buy, signals_sell

def add_elliott_wave_signals(df, zigzag_threshold=5):
    buy_sig, sell_sig = predict_elliott_wave(df, zigzag_threshold)
    df = df.copy()
    df["Elliott_Buy"] = buy_sig
    df["Elliott_Sell"] = sell_sig
    return df

def get_features_for_all(tickers, sma_windows, support_window, zigzag_threshold=5):
    multi_df = download_data_multi(tickers)
    if multi_df is None or multi_df.empty:
        return pd.DataFrame()

    features_list = []
    elliott_signals_list = []
    if isinstance(multi_df.columns, pd.MultiIndex):
        available = multi_df.columns.get_level_values(0).unique()
        for ticker in tickers:
            if ticker not in available:
                continue
            tdf = multi_df[ticker].dropna()
            if tdf.empty:
                continue
            feats = get_latest_features_for_ticker(tdf, ticker, sma_windows, support_window)
            if feats:
                features_list.append(feats)
            elliott_df = add_elliott_wave_signals(tdf, zigzag_threshold=zigzag_threshold)
            latest_elliott = elliott_df.iloc[-1][["Elliott_Buy", "Elliott_Sell"]].to_dict()
            latest_elliott["Ticker"] = ticker
            elliott_signals_list.append(latest_elliott)
    else:
        feats = get_latest_features_for_ticker(multi_df.dropna(), tickers[0], sma_windows, support_window)
        if feats:
            features_list.append(feats)
        elliott_df = add_elliott_wave_signals(multi_df.dropna(), zigzag_threshold=zigzag_threshold)
        latest_elliott = elliott_df.iloc[-1][["Elliott_Buy", "Elliott_Sell"]].to_dict()
        latest_elliott["Ticker"] = tickers
        elliott_signals_list.append(latest_elliott)

    features_df = pd.DataFrame(features_list)
    elliott_df = pd.DataFrame(elliott_signals_list)

    merged = features_df.merge(elliott_df, on="Ticker", how="left")
    return merged

def combine_signals(rule_df, ml_df, elliott_df):
    df = rule_df.copy()
    df = df.set_index("Ticker")
    ml_df = ml_df.set_index("Ticker")
    elliott_df = elliott_df.set_index("Ticker")

    combined_df = df.copy()
    combined_df["ML_Pred"] = ml_df["ML_Pred"]
    combined_df["Elliott_Buy"] = elliott_df["Elliott_Buy"]
    combined_df["Elliott_Sell"] = elliott_df["Elliott_Sell"]

    combined_df["Combined_Buy"] = (((combined_df["Buy_Point"]) | (combined_df["ML_Pred"] == "BUY")) & combined_df["Elliott_Buy"])
    combined_df["Combined_Sell"] = (((combined_df["Sell_Point"]) | (combined_df["ML_Pred"] == "SELL")) & combined_df["Elliott_Sell"])

    combined_df = combined_df.reset_index()
    return combined_df

# ---------------- RULE-BASED STRATEGY ----------------
def predict_buy_sell_rule(df, rsi_buy=30, rsi_sell=70):
    if df.empty:
        return df
    results = df.copy()

    results["Reversal_Buy"] = (
        (results["RSI"] < rsi_buy) &
        (results["Bullish_Div"]) &
        (np.abs(results["Close"] - results["Support"]) < 0.03 * results["Close"]) &
        (results["Close"] > results["SMA20"])
    )

    results["Trend_Buy"] = (
        (results["Close"] > results["SMA20"]) &
        (results["SMA20"] > results["SMA50"]) &
        (results["SMA50"] > results["SMA200"]) &
        (results["RSI"] > 50)
    )

    # Fix labeling: Reversal_Buy or Trend_Buy is Buy_Point (not Sell_Point)
    results["Sell_Point"] = results["Reversal_Buy"] | results["Trend_Buy"]  # Buy signals

    results["Buy_Point"] = (
        ((results["RSI"] > rsi_sell) & (results["Bearish_Div"])) |
        (results["Close"] < results["Support"]) |
        ((results["SMA20"] < results["SMA50"]) & (results["SMA50"] < results["SMA200"]))
    )
    return results

# ---------------- ML PIPELINE ----------------
@st.cache_data(show_spinner=False)
def load_history_for_ticker(ticker, period="5y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True)
        return df
    except Exception:
        return pd.DataFrame()

def label_from_rule_based(df, rsi_buy=30, rsi_sell=70):
    rules = predict_buy_sell_rule(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    label = pd.Series(0, index=rules.index, dtype=int)  # 0 = Hold
    label[rules["Buy_Point"]] = 1                       # 1 = Buy
    label[rules["Sell_Point"]] = -1                     # -1 = Sell
    return label

def label_from_future_returns(df, horizon=60, buy_thr=0.03, sell_thr=-0.03):
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = pd.Series(0, index=df.index, dtype=int)
    label[fut_ret >= buy_thr] = 1
    label[fut_ret <= sell_thr] = -1
    return label

def build_ml_dataset_for_tickers(
    tickers, sma_windows, support_window,
    label_mode="rule",
    horizon=60, buy_thr=0.03, sell_thr=-0.03,
    rsi_buy=30, rsi_sell=70,
    min_rows=250
):
    X_list, y_list, meta_list = [], [], []
    feature_cols = None

    for t in stqdm(tickers, desc="Preparing ML data"):
        hist = load_history_for_ticker(t, period="5y", interval="1d")
        if hist is None or hist.empty or len(hist) < min_rows:
            continue

        feat = compute_features(hist, sma_windows, support_window)
        if feat.empty:
            continue

        if label_mode == "rule":
            y = label_from_rule_based(feat, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
        else:
            y = label_from_future_returns(feat, horizon=horizon, buy_thr=buy_thr, sell_thr=sell_thr)

        data = feat.join(y.rename("Label")).dropna()
        if data.empty:
            continue

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

def latest_feature_row_for_ticker(ticker, sma_windows, support_window, feature_cols):
    hist = load_history_for_ticker(ticker, period="3y", interval="1d")
    if hist is None or hist.empty:
        return None
    feat = compute_features(hist, sma_windows, support_window).dropna()
    if feat.empty:
        return None
    use = feat.select_dtypes(include=[np.number])
    row = use.iloc[-1:].copy()
    for m in [c for c in feature_cols if c not in row.columns]:
        row[m] = 0.0
    row = row[feature_cols]
    return row

# ---------------- UI ----------------
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

if run_analysis:
    sma_tuple = (sma_w1, sma_w2, sma_w3)

    with st.spinner("Fetching data & computing features including Elliott Wave..."):
        feats_with_elliott = get_features_for_all(selected_tickers, sma_tuple, support_window, zigzag_threshold=5)
        if feats_with_elliott is None or feats_with_elliott.empty:
            st.error("No valid data for selected tickers.")
        else:
            # use columns needed by rules
            feature_cols_for_rules = set(feats_with_elliott.columns) & set(
                ["Close", "RSI", "Support", "SMA20", "SMA50", "SMA200", "Bullish_Div", "Bearish_Div"]
            )
            preds_rule = predict_buy_sell_rule(feats_with_elliott, rsi_buy, rsi_sell)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Rule Buy (current snapshot)",
        "❌ Rule Sell (current snapshot)",
        "📈 Chart",
        "🤖 ML Signals",
        "📊 Elliott & Combined Signals"
    ])

    with tab1:
        if feats_with_elliott.empty:
            st.info("No rule-based buy signals.")
        else:
            df_buy = preds_rule[preds_rule["Buy_Point"]]
            df_buy["TradingView"] = df_buy["Ticker"].apply(lambda x: f'<a href="https://in.tradingview.com/chart/?symbol=NSE%3A{x.replace(".NS","")}" target="_blank">📈 Chart</a>')
            cols = df_buy.columns.tolist()
            if "Ticker" in cols and "TradingView" in cols:
                cols.remove("TradingView")
                ticker_index = cols.index("Ticker")
                cols.insert(ticker_index + 1, "TradingView")
                df_buy = df_buy[cols]
            st.write(df_buy.to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab2:
        if feats_with_elliott.empty:
            st.info("No rule-based sell signals.")
        else:
            df_sell = preds_rule[preds_rule["Sell_Point"]]
            df_sell["TradingView"] = df_sell["Ticker"].apply(lambda x: f'<a href="https://in.tradingview.com/chart/?symbol=NSE%3A{x.replace(".NS","")}" target="_blank">📈 Chart</a>')
            cols = df_sell.columns.tolist()
            if "Ticker" in cols and "TradingView" in cols:
                cols.remove("TradingView")
                ticker_index = cols.index("Ticker")
                cols.insert(ticker_index + 1, "TradingView")
                df_sell = df_sell[cols]
            st.write(df_sell.to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab3:
        ticker_for_chart = st.selectbox("Chart Ticker", selected_tickers)
        chart_df = yf.download(ticker_for_chart, period="6mo", interval="1d", progress=False, threads=True)
        if not chart_df.empty:
            chart_df = compute_features(chart_df, sma_tuple, support_window).dropna()
            if not chart_df.empty:
                st.line_chart(chart_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
                st.line_chart(chart_df[["RSI"]])
        else:
            st.warning("No chart data available.")

    with tab4:
        if not SKLEARN_OK:
            st.error("scikit-learn not available. Install with: pip install scikit-learn")
        else:
            with st.spinner("Building ML dataset & training model..."):
                if label_mode == "Rule-based (teach the rules)":
                    X, y, feature_cols_ml, tickers_series = build_ml_dataset_for_tickers(
                        selected_tickers, sma_tuple, support_window,
                        label_mode="rule", rsi_buy=rsi_buy, rsi_sell=rsi_sell
                    )
                else:
                    X, y, feature_cols_ml, tickers_series = build_ml_dataset_for_tickers(
                        selected_tickers, sma_tuple, support_window,
                        label_mode="future", horizon=ml_horizon, buy_thr=ml_buy_thr, sell_thr=ml_sell_thr
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
                        row = latest_feature_row_for_ticker(t, sma_tuple, support_window, feature_cols_ml)
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
                    ml_df = pd.DataFrame(rows).sort_values(["ML_Pred","Prob_Buy"], ascending=[True, False])
                    def tradingview_link(ticker):
                        return f"https://in.tradingview.com/chart/?symbol=NSE%3A{ticker.replace('.NS','')}"
                    ml_df["TradingView"] = ml_df["Ticker"].apply(tradingview_link)

    with tab5:
        if not SKLEARN_OK:
            st.error("scikit-learn not available. Install with: pip install scikit-learn")
        else:
            if 'ml_df' not in locals() or ml_df.empty:
                st.info("Run ML Signals tab first to generate ML predictions.")
            else:
                elliott_signals_df = feats_with_elliott[["Ticker", "Elliott_Buy", "Elliott_Sell"]]
                combined_df = combine_signals(preds_rule, ml_df, elliott_signals_df)
                st.subheader("Elliott Wave Signals (Latest)")
                elliott_disp = elliott_signals_df[(elliott_signals_df["Elliott_Buy"]) | (elliott_signals_df["Elliott_Sell"])]
                if elliott_disp.empty:
                    st.write("No Elliott Wave buy or sell signals currently.")
                else:
                    st.write(elliott_disp)

                st.subheader("Combined Buy/Sell Signals")
                combined_disp = combined_df[(combined_df["Combined_Buy"]) | (combined_df["Combined_Sell"])][["Ticker", "Combined_Buy", "Combined_Sell"]]
                if combined_disp.empty:
                    st.write("No combined buy/sell signals currently.")
                else:
                    st.write(combined_disp)

                st.download_button(
                    "Download Combined Signals CSV",
                    combined_df.to_csv(index=False).encode(),
                    "combined_signals.csv",
                    "text/csv",
                )

    if 'preds_rule' in locals() and preds_rule is not None and not preds_rule.empty:
        st.download_button(
            "📥 Download Rule-based Results (snapshot)",
            preds_rule.to_csv(index=False).encode(),
            "nifty500_rule_signals.csv",
            "text/csv",
        )


st.markdown("⚠ Educational use only — not financial advice.")


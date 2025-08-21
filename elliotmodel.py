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
    # Add all tickers from your list here...
    "ECLERX.NS",
]

# ---------------- HELPERS ----------------

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
        idx = pd.MultiIndex.from_tuples(list(dict.fromkeys(out.columns.tolist())))
        out = out.loc[:, idx]
    return out

def compute_features(df, sma_windows=(20, 50, 200), support_window=30):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.DataFrame()

    df = df.copy()

    try:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()

    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)

    for w in (1, 3, 5, 10):
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    for win in sma_windows:
        df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]

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

def get_features_for_all(tickers, sma_windows, support_window):
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
            feats = get_latest_features_for_ticker(tdf, ticker, sma_windows, support_window)
            if feats:
                features_list.append(feats)
    else:
        feats = get_latest_features_for_ticker(multi_df.dropna(), tickers[0], sma_windows, support_window)
        if feats:
            features_list.append(feats)
    return pd.DataFrame(features_list)

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

    results["Buy_Point"] = results["Reversal_Buy"] | results["Trend_Buy"]  # Buy signals

    results["Sell_Point"] = (
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

# -------------- Streamlit UI --------------

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

    with st.spinner("Fetching data & computing rule-based features..."):
        feats = get_features_for_all(selected_tickers, sma_tuple, support_window)
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
        if feats.empty:
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
                    X, y, feature_cols, tickers_series = build_ml_dataset_for_tickers(
                        selected_tickers, sma_tuple, support_window,
                        label_mode="rule", rsi_buy=rsi_buy, rsi_sell=rsi_sell
                    )
                else:
                    X, y, feature_cols, tickers_series = build_ml_dataset_for_tickers(
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
                        row = latest_feature_row_for_ticker(t, sma_tuple, support_window, feature_cols)
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

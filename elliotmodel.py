import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Demo Nifty 500 tickers subset for testing limit & example
NIFTY_500_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "KOTAKBANK.NS"
]

def tradingview_link(ticker):
    base = ticker if ".NS" not in ticker else ticker[:-3]
    return f"https://www.tradingview.com/chart/?symbol=NSE:{base}"

def fetch_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    return df

def calculate_features(df):
    df = df.copy()
    # Calculate indicators
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["sma200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["support"] = df["Low"].rolling(20).min()
    df["resistance"] = df["High"].rolling(20).max()
    # Compute Fibonacci levels based on last 60 days
    high_60 = df["High"].rolling(60).max()
    low_60 = df["Low"].rolling(60).min()
    diff = high_60 - low_60
    df["fib23.6"] = high_60 - 0.236 * diff
    df["fib38.2"] = high_60 - 0.382 * diff
    df["fib50.0"] = high_60 - 0.5 * diff
    df["fib61.8"] = high_60 - 0.618 * diff
    df["fib78.6"] = high_60 - 0.786 * diff
    df.dropna(inplace=True)
    return df

# Label generation for training: Buy if next day close > current close else Sell (simple)
def generate_labels(df):
    df = df.copy()
    df['label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# Placeholder small ML training model for demo
def train_model(df):
    features = ['Close', 'rsi', 'sma20', 'sma50', 'sma200', 'support', 'resistance', 'fib23.6', 'fib38.2', 'fib50.0', 'fib61.8', 'fib78.6']
    df = df[features + ['label']].dropna()
    X = df[features]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

# Placeholder Elliott wave state (random for demo)
def elliott_wave_state():
    import random
    waves = ["impulsive", "corrective"]
    wave_nums = ["1", "2", "3", "4", "5", "a", "b", "c"]
    return random.choice(waves), random.choice(wave_nums)

def analyze_stock(ticker, model):
    df = fetch_data(ticker)
    if len(df) < 60:
        return None
    df = calculate_features(df)

    # Predict last available day
    last_row = df.iloc[-1]
    features = ['Close', 'rsi', 'sma20', 'sma50', 'sma200', 'support', 'resistance', 'fib23.6', 'fib38.2', 'fib50.0', 'fib61.8', 'fib78.6']
    X_pred = last_row[features].values.reshape(1, -1)
    pred_prob = model.predict_proba(X_pred)[0][1]
    decision = "Buy" if pred_prob > 0.5 else "Sell"

    wave_state, wave_label = elliott_wave_state()
    link = tradingview_link(ticker)

    return {
        "Stock": ticker,
        "Price": round(last_row['Close'], 2),
        "TradingView": link,
        "Signal": decision,
        "Probability": round(pred_prob, 2),
        "WaveState": wave_state,
        "WaveLabel": wave_label,
        "RSI": round(last_row['rsi'], 2),
        "SMA20": round(last_row['sma20'], 2),
        "SMA50": round(last_row['sma50'], 2),
        "SMA200": round(last_row['sma200'], 2),
        "Support": round(last_row['support'], 2),
        "Resistance": round(last_row['resistance'], 2),
        "Fib23.6": round(last_row['fib23.6'], 2),
        "Fib38.2": round(last_row['fib38.2'], 2),
        "Fib50.0": round(last_row['fib50.0'], 2),
        "Fib61.8": round(last_row['fib61.8'], 2),
        "Fib78.6": round(last_row['fib78.6'], 2),
    }

# Main Streamlit app
st.title("Nifty 500 ML Wave & Fibonacci Screener")

selected = st.multiselect("Select Stocks to Analyze", NIFTY_500_TICKERS, default=NIFTY_500_TICKERS[:5])

if st.button("Run Analysis"):
    # Train aggregated model on first stock for demo
    df_train = fetch_data(selected[0])
    df_train = calculate_features(df_train)
    df_train = generate_labels(df_train)
    model = train_model(df_train)

    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(selected):
        res = analyze_stock(ticker, model)
        if res is not None:
            results.append(res)
        progress.progress((i+1)/len(selected))
    if results:
        df_res = pd.DataFrame(results)
        # Make TradingView link clickable in markdown
        df_res['TradingView'] = df_res['TradingView'].apply(lambda x: f"[Chart]({x})")
        st.markdown(df_res.to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.write("No data found or error fetching stock data.")

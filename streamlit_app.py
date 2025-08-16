import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📈 Momentum Chaser - ATR/RRR/Trailing Stop Calculator")

ticker = st.text_input("銘柄コード (例: 1911.T)", "1911.T")
entry_prices = st.text_input("エントリー価格 (カンマ区切り)", "1000,1060")

if st.button("計算する"):
    entries = [float(x.strip()) for x in entry_prices.split(",")]

    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if data.empty:
        st.error("データが取得できませんでした")
    else:
        data["H-L"] = data["High"] - data["Low"]
        data["H-C"] = (data["High"] - data["Close"].shift()).abs()
        data["L-C"] = (data["Low"] - data["Close"].shift()).abs()
        tr = data[["H-L", "H-C", "L-C"]].max(axis=1)
        data["ATR"] = tr.rolling(14).mean()

        latest_close = data["Close"].iloc[-1]
        atr = data["ATR"].iloc[-1]

        st.write(f"📌 終値: {latest_close:.2f} 円")
        st.write(f"ATR(14): {atr:.2f} 円")

        init_stop = entries[0] - 2 * atr
        st.write(f"初期ストップ (1st Entry基準): {init_stop:.2f} 円")

        avg_entry = sum(entries) / len(entries)
        trail_stop = latest_close - 2 * atr
        st.write(f"平均取得価格: {avg_entry:.2f} 円")
        st.write(f"ATRトレーリングストップ: {trail_stop:.2f} 円")

        if trail_stop > avg_entry:
            st.success("✅ 利益確保モードに入っています！")

# streamlit_app.py — Momentum Chaser Coach (robust + yfinance列フラット化)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ===== ページ設定 =====
st.set_page_config(page_title="Momentum Chaser Coach", page_icon="🚀", layout="centered")
st.title("🚀 Momentum Chaser - ATR / RRR / Trailing Stop (Robust)")

# ===== パラメータ =====
ATR_N = 14
ATR_MULT_STOP  = 2.0
ATR_MULT_TRAIL = 2.0
ADD_STEP_ATR   = 1.0
TARGET_ATR     = 3.0
MIN_RRR_ADD    = 1.5
CHECK_HI20     = True
AUTO_ADJ       = False  # yfinanceのauto_adjust（明示）

# ===== 入力UI =====
with st.form(key="mc_form"):
    col1, col2 = st.columns([1,1])
    with col1:
        symbol = st.text_input("銘柄コード（例: 1911.T）", "1911.T").strip()
    with col2:
        entries_text = st.text_input("エントリー価格（カンマ区切り）", "1000,1060").strip()
    run = st.form_submit_button("計算する")

# ===== ユーティリティ =====
def parse_entries(s: str):
    try:
        vals = [float(x) for x in s.split(",") if x.strip() != ""]
        return sorted(vals)
    except Exception:
        return []

def calc_atr_ewm(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceが返す MultiIndex 列
      例: ('Close','1911.T') / ('1911.T','Close')
    を単層列 'Close','High','Low','Open','Volume' に正規化する。
    """
    if isinstance(df.columns, pd.MultiIndex):
        keys = {"Open","High","Low","Close","Adj Close","Volume"}
        new_cols = []
        for col in df.columns.to_list():
            if isinstance(col, tuple):
                chosen = None
                for tok in col:
                    t = str(tok)
                    if t in keys:
                        chosen = t
                        break
                if chosen is None:
                    chosen = str(col[0])
                new_cols.append(chosen)
            else:
                new_cols.append(str(col))
        df = df.copy()
        df.columns = new_cols
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

    # 'Close' が無く 'Adj Close' しかない場合は 'Close' に寄せる
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # 余分な列は落とす（存在するものだけ）
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep]

def fetch_history(sym: str, back_days: int = 420) -> pd.DataFrame:
    df = yf.download(
        sym, period=f"{back_days}d", interval="1d",
        auto_adjust=AUTO_ADJ, group_by="column", progress=False, threads=True
    )
    # 失敗したら .T 付きで軽くリトライ（数字のみ等の保険）
    if (df is None or df.empty) and not sym.endswith(".T"):
        df = yf.download(
            sym + ".T", period=f"{back_days}d", interval="1d",
            auto_adjust=AUTO_ADJ, group_by="column", progress=False, threads=True
        )
    if df is None:
        return pd.DataFrame()
    return _flatten_yf_columns(df)

def last_valid_row(df: pd.DataFrame, cols=("Close","High","Low")):
    valid = df.dropna(subset=list(cols))
    if valid.empty:
        return None
    return valid.iloc[-1]

# ===== 本体 =====
if run:
    entries = parse_entries(entries_text)
    if not entries:
        st.error("エントリー価格の形式が不正です。例: 1000,1060")
        st.stop()

    df = fetch_history(symbol, back_days=420)
    if df.empty:
        st.error("価格データを取得できませんでした。銘柄コードや市場サフィックス（.T）をご確認ください。")
        st.stop()

    required = {"Open","High","Low","Close"}
    if not required.issubset(df.columns):
        st.error(f"取得データに必要列が不足しています: {list(df.columns)}")
        with st.expander("取得データ（末尾）"):
            st.dataframe(df.tail(10))
        st.stop()

    # 指標
    df["ATR"]  = calc_atr_ewm(df, ATR_N)
    df["HI20"] = df["High"].rolling(20).max()

    row = last_valid_row(df, cols=("Close","High","Low","ATR"))
    if row is None or pd.isna(row["ATR"]):
        st.error("直近バーに有効な価格/ATRがありません（データ欠損/本数不足の可能性）。")
        st.dataframe(df.tail(10))
        st.stop()

    price = float(row["Close"])
    atr   = float(row["ATR"])
    hi20  = (float(row["HI20"]) if pd.notna(row["HI20"]) else None)

    st.subheader("現在値・指標")
    st.write(
        (f"**終値**: {price:.2f} / **ATR({ATR_N})**: {atr:.2f}"
         + (f" / **20日高値**: {hi20:.2f}" if hi20 is not None else " / **20日高値**: NA"))
    )

    # ストップ（均等ロット想定）
    entry0      = entries[0]
    base_stop   = entry0 - ATR_MULT_STOP * atr
    ladder_stop = base_stop
    if len(entries) >= 2:
        ladder_stop = max(ladder_stop, entries[-2])  # 直前エントリー以上に引き上げ

    trail_stop  = (hi20 - ATR_MULT_TRAIL * atr) if hi20 is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        comp     = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        comp     = "はしご"

    st.subheader("ストップ")
    st.write(f"- 初期ストップ: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d高値- {ATR_MULT_TRAIL:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{comp}）**: {stop_use:.2f}")

    # 追加 & RRR
    next_add_trigger = entries[-1] + ADD_STEP_ATR * atr
    risk_now   = max(1e-9, price - stop_use)
    reward_now = TARGET_ATR * atr
    rrr_now    = reward_now / risk_now
    add_ok     = (price >= next_add_trigger) and (rrr_now >= MIN_RRR_ADD)
    if CHECK_HI20 and hi20 is not None:
        add_ok = add_ok and (price >= hi20)

    st.subheader("追加条件 & RRR")
    cond_line = f"次の追加: 価格≧ {next_add_trigger:.2f}"
    if CHECK_HI20 and hi20 is not None:
        cond_line += f" かつ 価格≧ 20日高値({hi20:.2f})"
    st.write(cond_line)
    st.write(f"RRR(今追加想定): **{rrr_now:.2f}**  (risk={risk_now:.2f}, reward≈{reward_now:.2f}, 目安≥{MIN_RRR_ADD})")
    st.info("🟢 追加OK") if add_ok else st.warning("🔸 見送り（条件未達 or RRR不足）")

    st.subheader("モード")
    st.write("✅ 利益確保" if stop_use >= entry0 else "—")

    with st.expander("データ末尾（デバッグ用）"):
        st.dataframe(df.tail(5))

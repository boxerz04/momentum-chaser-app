# streamlit_app.py — Momentum Chaser Coach (UI調整版・堅牢)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ========== ページ設定 ==========
st.set_page_config(page_title="Momentum Chaser Coach", page_icon="🚀", layout="centered")
st.title("🚀 Momentum Chaser - ATR / RRR / Trailing Stop (Robust & Tunable)")

# ========== ユーティリティ ==========
def parse_entries(s: str):
    """カンマ区切りの価格文字列 -> 昇順のfloat配列"""
    try:
        vals = [float(x) for x in s.split(",") if x.strip() != ""]
        return sorted(vals)
    except Exception:
        return []

def calc_atr_ewm(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder系の平滑に近いEMA版ATR"""
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceのMultiIndex列を単層に正規化し、必要列だけ残す。
    例: ('Close','1911.T') / ('1911.T','Close') -> 'Close'
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
                new_cols.append(chosen if chosen else str(col[0]))
            else:
                new_cols.append(str(col))
        df = df.copy()
        df.columns = new_cols
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep]

def fetch_history(sym: str, back_days: int = 420, auto_adjust: bool = False) -> pd.DataFrame:
    """価格取得（失敗時は.Tを付けて再試行）"""
    df = yf.download(
        sym, period=f"{back_days}d", interval="1d",
        auto_adjust=auto_adjust, group_by="column",
        progress=False, threads=True
    )
    if (df is None or df.empty) and not sym.endswith(".T"):
        df = yf.download(
            sym + ".T", period=f"{back_days}d", interval="1d",
            auto_adjust=auto_adjust, group_by="column",
            progress=False, threads=True
        )
    if df is None or df.empty:
        return pd.DataFrame()
    return _flatten_yf_columns(df)

def last_valid_row(df: pd.DataFrame, cols=("Close","High","Low","ATR")):
    valid = df.dropna(subset=list(cols))
    return None if valid.empty else valid.iloc[-1]

# ========== 入力UI ==========
with st.form(key="mc_form"):
    c1, c2 = st.columns([1,1])
    with c1:
        symbol = st.text_input("銘柄コード（例: 1911.T）", "1911.T").strip()
    with c2:
        entries_text = st.text_input("エントリー価格（カンマ区切り）", "1000,1060").strip()

    st.markdown("**パラメータ**")
    p1, p2, p3 = st.columns([1,1,1])
    with p1:
        atr_n = st.number_input("ATR期間", min_value=5, max_value=50, value=14, step=1)
        atr_mult_stop = st.number_input("初期ストップ倍率（ATR×）", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    with p2:
        atr_mult_trail = st.number_input("トレーリング倍率（ATR×）", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        add_step_atr = st.number_input("追加間隔（ATR×）", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    with p3:
        target_atr = st.number_input("目標利幅（ATR×）", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        rrr_min = st.number_input("追加判定の最小RRR", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

    q1, q2 = st.columns([1,1])
    with q1:
        check_hi20 = st.checkbox("20日高値ブレイクも要求", value=True)
    with q2:
        auto_adj = st.checkbox("終値を調整済みで取得（auto_adjust）", value=False)

    run = st.form_submit_button("計算する")

# ========== 本体 ==========
if run:
    # 入力チェック
    entries = parse_entries(entries_text)
    if not entries:
        st.error("エントリー価格の形式が不正です。例: 1000,1060")
        st.stop()

    # データ取得
    df = fetch_history(symbol, back_days=420, auto_adjust=auto_adj)
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
    df["ATR"]  = calc_atr_ewm(df, n=int(atr_n))
    df["HI20"] = df["High"].rolling(20, min_periods=1).max()

    row = last_valid_row(df, cols=("Close","High","Low","ATR"))
    if row is None or pd.isna(row["ATR"]):
        st.error("直近バーに有効な価格/ATRがありません（データ欠損/本数不足の可能性）。")
        st.dataframe(df.tail(10))
        st.stop()

    price = float(row["Close"])
    atr   = float(row["ATR"])
    hi20  = (float(row["HI20"]) if pd.notna(row["HI20"]) else None)

    # 表示：現在値・指標
    st.subheader("現在値・指標")
    st.write(
        f"**終値**: {price:.2f} / **ATR({int(atr_n)})**: {atr:.2f}"
        + (f" / **20日高値**: {hi20:.2f}" if hi20 is not None else " / **20日高値**: NA")
    )

    # ストップ（均等ロット想定・はしご式 + ATRトレイルの高い方）
    entry0      = entries[0]
    base_stop   = entry0 - atr_mult_stop * atr
    ladder_stop = base_stop
    if len(entries) >= 2:
        ladder_stop = max(ladder_stop, entries[-2])  # 直前エントリー以上に引き上げ

    trail_stop = (hi20 - atr_mult_trail * atr) if hi20 is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        comp     = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        comp     = "はしご"

    st.subheader("ストップ")
    st.write(f"- 初期ストップ: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d高値 - {atr_mult_trail:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{comp}）**: {stop_use:.2f}")

    # 追加条件 & RRR（“次の一段”の目安を提示）
    next_add_trigger = entries[-1] + add_step_atr * atr
    risk_now   = max(1e-9, price - stop_use)   # 現在の推奨ストップを前提にした下側リスク
    reward_now = target_atr * atr              # 上側目安（ATR×）
    rrr_now    = reward_now / risk_now

    add_ok = (price >= next_add_trigger) and (rrr_now >= rrr_min)
    if check_hi20 and hi20 is not None:
        add_ok = add_ok and (price >= hi20)

    st.subheader("追加条件 & RRR")
    cond_line = f"次の追加: 価格≧ {next_add_trigger:.2f}"
    if check_hi20 and hi20 is not None:
        cond_line += f" かつ 価格≧ 20日高値({hi20:.2f})"
    st.write(cond_line)
    st.write(f"RRR(今追加想定): **{rrr_now:.2f}**  (risk={risk_now:.2f}, reward≈{reward_now:.2f}, 目安≥{rrr_min})")

    if add_ok:
        st.info("🟢 追加OK（条件達成）")
    else:
        st.warning("🔸 見送り（条件未達 or RRR不足）")

    # 利益確保モード表示
    st.subheader("モード")
    st.write("✅ 利益確保モード" if stop_use >= entry0 else "—")

    with st.expander("データ末尾（デバッグ用）"):
        st.dataframe(df.tail(8))

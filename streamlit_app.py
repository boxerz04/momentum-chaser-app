# streamlit_app.py — Momentum Chaser Coach
# (as-of / Shares / P&L, no verbose logs, hide ".T" in display)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

# ========== ページ設定 ==========
st.set_page_config(page_title="Momentum Chaser Coach", page_icon="🚀", layout="centered")
st.title("🚀 Momentum Chaser - ATR / RRR / Trailing Stop")

# ========== ユーティリティ ==========
def parse_entries(s: str):
    try:
        vals = [float(x) for x in s.split(",") if x.strip() != ""]
        return sorted(vals)
    except Exception:
        return []

def calc_atr_ewm(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        df = df.copy(); df.columns = new_cols
    else:
        df = df.copy(); df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep]

def fetch_history(sym: str, back_days: int = 900, auto_adjust: bool = False) -> pd.DataFrame:
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
    df = _flatten_yf_columns(df)
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df

def last_valid_row_on_or_before(df: pd.DataFrame, asof: pd.Timestamp, cols=("Close","High","Low","ATR")):
    m = (df.index <= asof)
    if not m.any():
        return None
    view = df.loc[m].dropna(subset=list(cols))
    if view.empty:
        return None
    return view.iloc[-1]

def display_symbol(sym: str) -> str:
    return sym[:-2] if sym.endswith(".T") else sym

# ========== 入力UI ==========
with st.form(key="mc_form"):
    c1, c2 = st.columns([1,1])
    with c1:
        symbol = st.text_input("銘柄コード（例: 9513.T）", "9513").strip()
    with c2:
        entries_text = st.text_input("エントリー価格（カンマ区切り: 例 2763.5,2814,2865）", "1000,1060").strip()

    s1, s2 = st.columns([1,1])
    with s1:
        shares_per_entry = st.number_input("単位株数（1回のエントリーあたり）", min_value=1, value=100, step=1)
    with s2:
        auto_adj = st.checkbox("終値を調整済みで取得（auto_adjust）", value=False)

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

    st.markdown("**基準日（as-of）**")
    use_asof = st.checkbox("過去日で計算する（as-of固定）", value=False)
    asof_date = st.date_input("基準日（市場営業日でなくてもOK）", value=date.today())

    run = st.form_submit_button("計算する")

# ========== 本体 ==========
if run:
    entries = parse_entries(entries_text)
    if not entries:
        st.error("エントリー価格の形式が不正です。例: 2763.5,2814,2865")
        st.stop()

    df = fetch_history(symbol, back_days=900, auto_adjust=auto_adj)
    if df.empty:
        st.error("価格データを取得できませんでした。銘柄コードや市場サフィックス（.T）をご確認ください。")
        st.stop()

    required = {"Open","High","Low","Close"}
    if not required.issubset(df.columns):
        st.error(f"取得データに必要列が不足しています: {list(df.columns)}")
        st.stop()

    # 指標
    df["ATR"]  = calc_atr_ewm(df, n=int(atr_n))
    df["HI20"] = df["High"].rolling(20, min_periods=1).max()

    # as-of
    if use_asof:
        asof_ts = pd.to_datetime(asof_date)
        row = last_valid_row_on_or_before(df, asof_ts, cols=("Close","High","Low","ATR"))
        if row is None:
            st.error("指定の基準日以前に有効なバーが見つかりませんでした（データ不足／上場前の可能性）。")
            st.stop()
        effective_date = pd.to_datetime(row.name).date()
    else:
        row = df.dropna(subset=["Close","High","Low","ATR"]).iloc[-1]
        effective_date = pd.to_datetime(row.name).date()

    price = float(row["Close"]); atr = float(row["ATR"])
    hi20 = float(row["HI20"]) if pd.notna(row["HI20"]) else None

    # ポジション集計（均等ロット）
    n_entries = len(entries)
    qty_total = shares_per_entry * n_entries
    avg_entry = sum(entries) / n_entries
    notional  = avg_entry * qty_total

    # ストップ（はしご式 + ATRトレイルの高い方）
    entry0      = entries[0]
    base_stop   = entry0 - atr_mult_stop * atr
    ladder_stop = base_stop
    if n_entries >= 2:
        ladder_stop = max(ladder_stop, entries[-2])  # 直前エントリー以上
    trail_stop = (hi20 - atr_mult_trail * atr) if hi20 is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        comp     = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        comp     = "はしご"

    # RRR・次の一手
    next_add_trigger = entries[-1] + add_step_atr * atr
    risk_per_share   = max(0.0, price - stop_use)
    reward_per_share = target_atr * atr
    rrr_now          = (reward_per_share / max(1e-9, risk_per_share)) if risk_per_share > 0 else float("inf")

    # ========== 表示 ==========
    st.subheader(f"現在値・指標（{display_symbol(symbol)} / 評価バー）")
    top_caption = f"**評価日**: {effective_date} / **終値**: {price:.2f} / **ATR({int(atr_n)})**: {atr:.2f}"
    if hi20 is not None:
        top_caption += f" / **20日高値**: {hi20:.2f}"
    st.write(top_caption)
    if use_asof and effective_date != asof_date:
        st.caption(f"※基準日 {asof_date} は休場または欠損のため、直近営業日 {effective_date} で評価。")

    st.subheader("ポジション（均等ロット）")
    st.write(f"- エントリー: {', '.join(f'{e:.2f}' for e in entries)}")
    st.write(f"- 1回あたり株数: {shares_per_entry:,} 株 / 回")
    st.write(f"- 建玉本数: {n_entries} 回  → **総株数: {qty_total:,} 株**")
    st.write(f"- 平均取得: **{avg_entry:.2f} 円**  / 総建玉額: 約 **{notional:,.0f} 円**")

    st.subheader("ストップ")
    st.write(f"- 初期ストップ: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d高値 - {atr_mult_trail:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{comp}）**: {stop_use:.2f}")

    # P/L（現在・ストップ・目標）
    st.subheader("含み損益 / リスク&リワード（合計）")
    pl_now_total    = (price - avg_entry) * qty_total
    risk_total      = (price - stop_use) * qty_total if stop_use < price else 0.0
    target_price    = price + reward_per_share
    reward_total    = (target_price - price) * qty_total

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("含み損益（いま）", f"{pl_now_total:,.0f} 円", help=f"(現在 {price:.2f} − 平均 {avg_entry:.2f}) × {qty_total:,}")
    with colB:
        st.metric("想定損失（ストップ）", f"{-risk_total:,.0f} 円", help=f"(現在 {price:.2f} − 推奨 {stop_use:.2f}) × {qty_total:,}")
    with colC:
        st.metric("想定利益（目標）", f"{reward_total:,.0f} 円", help=f"(目標 {target_price:.2f} − 現在 {price:.2f}) × {qty_total:,}")

    st.caption(f"※ 目標価格 = 現在値 + {target_atr}×ATR = {target_price:.2f} 円 / RRR ≈ {rrr_now:.2f}")

    st.subheader("追加条件 & RRR（この位置で追加した場合の目安）")
    cond_line = f"- 追加指値候補: **{next_add_trigger:.2f} 円**"
    if hi20 is not None:
        cond_line += f"（20日高値 {hi20:.2f} 円ブレイクも要求がデフォルト想定）"
    st.write(cond_line)
    st.write(f"- 想定RRR: **{rrr_now:.2f}**  (risk/株={risk_per_share:.2f}, reward/株≈{reward_per_share:.2f}, 最低目安≥{rrr_min})")

    # 明示 if/else（副作用の戻り値をUIに出さない）
    add_ok = (price >= next_add_trigger) and (rrr_now >= rrr_min)
    if hi20 is not None:
        add_ok = add_ok and (price >= hi20)

    if add_ok:
        st.info("🟢 追加OK（条件達成）")
    else:
        st.warning("🔸 見送り（条件未達 or RRR不足）")


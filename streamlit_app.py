# streamlit_app.py — Momentum Chaser Coach (Compact / Mobile-first)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="Momentum Chaser", page_icon="🚀", layout="centered")
st.markdown("## 🚀 Momentum Chaser — ATR / RRR / Trailing Stop")

# ========= Utils =========
def disp_symbol(sym: str) -> str:
    return sym[:-2] if sym.endswith(".T") else sym

def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        keys = {"Open","High","Low","Close","Adj Close","Volume"}
        cols = []
        for col in df.columns.to_list():
            if isinstance(col, tuple):
                chosen = None
                for tok in col:
                    t = str(tok)
                    if t in keys:
                        chosen = t; break
                cols.append(chosen if chosen else str(col[0]))
            else:
                cols.append(str(col))
        df = df.copy(); df.columns = cols
    else:
        df = df.copy(); df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df

def fetch_history(sym: str, back_days: int = 900, auto_adjust: bool = False) -> pd.DataFrame:
    df = yf.download(sym, period=f"{back_days}d", interval="1d",
                     auto_adjust=auto_adjust, group_by="column",
                     progress=False, threads=True)
    if (df is None or df.empty) and not sym.endswith(".T"):
        df = yf.download(sym + ".T", period=f"{back_days}d", interval="1d",
                         auto_adjust=auto_adjust, group_by="column",
                         progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    return flatten_yf(df)

def calc_atr_ewm(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def last_row_on_or_before(df: pd.DataFrame, asof: pd.Timestamp, cols=("Close","High","Low","ATR")):
    m = (df.index <= asof)
    if not m.any(): return None
    view = df.loc[m].dropna(subset=list(cols))
    if view.empty: return None
    return view.iloc[-1]

def parse_positions(text: str):
    """
    入力例:
      2763.5 100
      2814 50
    空行は無視。スペース/カンマ区切り OK。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        ln = ln.replace(",", " ")
        toks = [t for t in ln.split() if t]
        if len(toks) < 2:  # price shares 必須
            continue
        try:
            price = float(toks[0]); qty = int(float(toks[1]))
            if price > 0 and qty > 0:
                out.append((price, qty))
        except Exception:
            pass
    return out

# ========= Top controls (compact) =========
with st.form("mc_compact"):
    c0, c1 = st.columns([1.2, 1])
    with c0:
        symbol = st.text_input("銘柄コード（例: 9513）", "9513").strip()
    with c1:
        run = st.form_submit_button("計算する", use_container_width=True)

    st.caption("**エントリー（最大5行）** 例: `2763.5 100`（価格 株数）")
    positions_text = st.text_area(
        "", value="1000 100\n1060 100", height=110, placeholder="価格 株数 を1行ずつ"
    )

    with st.expander("詳細設定（必要なときだけ開く）", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            atr_n = st.number_input("ATR期間", 5, 50, 14, 1)
            auto_adj = st.checkbox("調整終値で取得（auto_adjust）", value=False)
        with a2:
            atr_mult_stop  = st.number_input("初期ストップ(ATR×)", 0.5, 5.0, 2.0, 0.5)
            atr_mult_trail = st.number_input("トレイル倍率(ATR×)", 1.0, 5.0, 2.0, 0.5)
            require_hi20   = st.checkbox("20日高値ブレイクも要求", value=True)
        with a3:
            add_step_atr = st.number_input("追加間隔(ATR×)", 0.5, 5.0, 1.0, 0.1)
            target_atr   = st.number_input("目標利幅(ATR×)", 1.0, 10.0, 3.0, 0.5)
            rrr_min      = st.number_input("追加の最小RRR", 1.0, 5.0, 1.5, 0.1)

        asof_switch = st.checkbox("過去日で評価（as-of）", value=False)
        asof_date   = st.date_input("基準日", value=date.today())

# ========= Main =========
if run:
    entries = parse_positions(positions_text)[:5]
    if not entries:
        st.error("少なくとも1行は『価格 株数』で入力してください。例: 2763.5 100")
        st.stop()

    df = fetch_history(symbol, back_days=900, auto_adjust=auto_adj)
    if df.empty:
        st.error("価格データを取得できませんでした。銘柄コードや市場サフィックス（.T）をご確認ください。")
        st.stop()

    need = {"Open","High","Low","Close"}
    if not need.issubset(df.columns):
        st.error(f"取得データに必要列が不足しています: {list(df.columns)}")
        st.stop()

    # 指標
    df["ATR"]  = calc_atr_ewm(df, n=int(atr_n))
    df["HI20"] = df["High"].rolling(20, min_periods=1).max()

    # as-of
    if asof_switch:
        row = last_row_on_or_before(df, pd.to_datetime(asof_date), cols=("Close","High","Low","ATR"))
        if row is None:
            st.error("指定基準日以前に有効データが見つかりませんでした（データ不足／上場前の可能性）。")
            st.stop()
        eff_date = pd.to_datetime(row.name).date()
    else:
        row = df.dropna(subset=["Close","High","Low","ATR"]).iloc[-1]
        eff_date = pd.to_datetime(row.name).date()

    price = float(row["Close"])
    atr   = float(row["ATR"])
    hi20  = float(row["HI20"]) if pd.notna(row["HI20"]) else None

    # 集計（可変ロット）
    qty_total = sum(q for _, q in entries)
    notional  = sum(p*q for p, q in entries)
    avg_entry = notional / qty_total

    # ストップ（はしご式 + ATRトレイル の高い方）
    first_price = entries[0][0]
    base_stop   = first_price - atr_mult_stop * atr
    ladder_stop = base_stop
    if len(entries) >= 2:
        prev_price = entries[-2][0]         # 直前(最後から1つ前)
        ladder_stop = max(ladder_stop, prev_price)
    trail_stop  = (hi20 - atr_mult_trail * atr) if hi20 is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        stop_basis = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        stop_basis = "はしご"

    # 次の一手 & RRR
    last_entry_price = entries[-1][0]
    next_add_trigger = last_entry_price + add_step_atr * atr
    risk_per_share   = max(0.0, price - stop_use)
    reward_per_share = target_atr * atr
    rrr_now = (reward_per_share / max(1e-9, risk_per_share)) if risk_per_share > 0 else float("inf")

    # ===== Display (lean) =====
    st.markdown(f"### {disp_symbol(symbol)} — 現在値・指標")
    top = f"**評価日**: {eff_date} / **終値**: {price:.2f} / **ATR({int(atr_n)})**: {atr:.2f}"
    if hi20 is not None: top += f" / **20日高値**: {hi20:.2f}"
    st.write(top)
    if asof_switch and eff_date != asof_date:
        st.caption(f"※基準日 {asof_date} は休場・欠損のため、直近営業日の {eff_date} で評価。")

    st.markdown("### ポジション（可変ロット）")
    df_entries = pd.DataFrame([{"回": i+1, "価格": p, "株数": q, "金額(約)": int(round(p*q))} for i,(p,q) in enumerate(entries)])
    st.table(df_entries.style.format({"価格":"{:.2f}"}))
    st.write(f"- **総株数**: {qty_total:,} 株 / **平均取得**: {avg_entry:.2f} 円 / **総建玉額(約)**: {notional:,.0f} 円")

    st.markdown("### ストップ")
    st.write(f"- 初期ストップ（1st基準）: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d高値 − {atr_mult_trail:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{stop_basis}）: {stop_use:.2f}**")
    st.caption("利益確保モード" if stop_use >= first_price else "—")

    st.markdown("### 含み損益 / リスク&リワード（合計）")
    pl_now_total = (price - avg_entry) * qty_total
    risk_total   = (price - stop_use) * qty_total if stop_use < price else 0.0
    target_price = price + reward_per_share
    reward_total = (target_price - price) * qty_total

    a,b,c = st.columns(3)
    a.metric("含み損益（いま）", f"{pl_now_total:,.0f} 円")
    b.metric("想定損失（ストップ）", f"{-risk_total:,.0f} 円")
    c.metric("想定利益（目標）", f"{reward_total:,.0f} 円")
    st.caption(f"※ 目標価格 = 現在値 + {target_atr}×ATR = {target_price:.2f} 円 / RRR ≈ {rrr_now:.2f}")

    st.markdown("### 次の追加 & RRR（目安）")
    line = f"- **追加指値候補**: {next_add_trigger:.2f} 円"
    if require_hi20 and hi20 is not None:
        line += f"（20日高値 {hi20:.2f} 円もブレイク要求）"
    st.write(line)
    st.write(f"- **想定RRR**: {rrr_now:.2f}  (risk/株={risk_per_share:.2f}, reward/株≈{reward_per_share:.2f})")

    add_ok = (price >= next_add_trigger) and (rrr_now >= rrr_min)
    if require_hi20 and hi20 is not None:
        add_ok = add_ok and (price >= hi20)
    st.info("🟢 追加OK（条件達成）") if add_ok else st.warning("🔸 見送り（条件未達 or RRR不足）")

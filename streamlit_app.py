# streamlit_app.py — Momentum Chaser Coach
# (Per-entry price & shares, as-of / ATR stops / P&L / RRR)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

# ===== Page =====
st.set_page_config(page_title="Momentum Chaser Coach", page_icon="🚀", layout="centered")
st.title("🚀 Momentum Chaser - ATR / RRR / Trailing Stop")

# ===== Utils =====
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
    df = flatten_yf(df)
    df.index = pd.to_datetime(df.index).normalize()
    return df

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
    if not m.any():
        return None
    view = df.loc[m].dropna(subset=list(cols))
    if view.empty:
        return None
    return view.iloc[-1]

def disp_symbol(sym: str) -> str:
    return sym[:-2] if sym.endswith(".T") else sym

# ===== Form =====
with st.form("mc_form"):
    c1, c2 = st.columns([1,1])
    with c1:
        symbol = st.text_input("銘柄コード（例: 9513）", "9513").strip()
        auto_adj = st.checkbox("終値を調整済みで取得（auto_adjust）", value=False)
    with c2:
        asof_switch = st.checkbox("過去日で計算（as-of）", value=False)
        asof_date = st.date_input("基準日（市場休場でも可）", value=date.today())

    st.markdown("### エントリー（最大5回）")
    st.caption("空欄は無視。価格と株数は正の数で入力。")
    entry_rows = []
    cols = st.columns(5)
    # 5枠分まとめて横並びにするより、縦の方がわかりやすければここを調整
    for i in range(5):
        with st.container():
            e1, e2 = st.columns([1,1])
            price = e1.number_input(f"{i+1}回目 価格", min_value=0.0, value=0.0, step=0.1, key=f"p_{i}")
            shares = e2.number_input(f"{i+1}回目 株数", min_value=0, value=0, step=1, key=f"s_{i}")
            entry_rows.append((price, shares))

    st.markdown("### パラメータ")
    p1, p2, p3 = st.columns([1,1,1])
    with p1:
        atr_n = st.number_input("ATR期間", min_value=5, max_value=50, value=14, step=1)
        atr_mult_stop = st.number_input("初期ストップ（ATR×）", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    with p2:
        atr_mult_trail = st.number_input("トレーリング倍率（ATR×）", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        add_step_atr = st.number_input("追加間隔（ATR×）", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    with p3:
        target_atr = st.number_input("目標利幅（ATR×）", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        rrr_min = st.number_input("追加判定の最小RRR", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

    require_hi20 = st.checkbox("20日高値ブレイクを追加条件に含める", value=True)

    run = st.form_submit_button("計算する")

# ===== Main =====
if run:
    # 有効エントリー抽出（入力順保持）
    entries = [(float(p), int(q)) for (p, q) in entry_rows if (p > 0 and q > 0)]
    if not entries:
        st.error("少なくとも1つは『価格＞0 と 株数＞0』で入力してください。")
        st.stop()

    # データ取得
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
            st.error("指定基準日以前に有効データが見つかりません（データ不足／上場前の可能性）。")
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

    # ストップ：はしご式 + ATRトレイル
    entry0_price = entries[0][0]
    base_stop    = entry0_price - atr_mult_stop * atr
    ladder_stop  = base_stop
    if len(entries) >= 2:
        # 直前（最後から1つ前）のエントリー価格以上へ
        prev_price = entries[-2][0]
        ladder_stop = max(ladder_stop, prev_price)
    trail_stop   = (hi20 - atr_mult_trail * atr) if hi20 is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        comp_str = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        comp_str = "はしご"

    # RRR・追加トリガー（次の追加は「最後のエントリー価格 + ATR×間隔」）
    last_entry_price = entries[-1][0]
    next_add_trigger = last_entry_price + add_step_atr * atr
    risk_per_share   = max(0.0, price - stop_use)
    reward_per_share = target_atr * atr
    rrr_now          = (reward_per_share / max(1e-9, risk_per_share)) if risk_per_share > 0 else float("inf")

    # ===== 出力 =====
    st.subheader(f"現在値・指標（{disp_symbol(symbol)} / 評価バー）")
    top = f"**評価日**: {eff_date} / **終値**: {price:.2f} / **ATR({int(atr_n)})**: {atr:.2f}"
    if hi20 is not None:
        top += f" / **20日高値**: {hi20:.2f}"
    st.write(top)
    if asof_switch and eff_date != asof_date:
        st.caption(f"※基準日 {asof_date} は休場または欠損のため、直近営業日 {eff_date} で評価。")

    st.subheader("ポジション（可変ロット）")
    df_entries = pd.DataFrame([{"回": i+1, "価格": p, "株数": q, "金額(約)": p*q} for i,(p,q) in enumerate(entries)])
    st.table(df_entries)
    st.write(f"- **総株数**: {qty_total:,} 株 / **平均取得**: {avg_entry:.2f} 円 / **総建玉額(約)**: {notional:,.0f} 円")

    st.subheader("ストップ")
    st.write(f"- 初期ストップ（1st基準）: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d高値 − {atr_mult_trail:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{comp_str}）**: {stop_use:.2f}")
    st.write("— 利益確保モード" if stop_use >= entry0_price else "—")

    st.subheader("含み損益 / リスク&リワード（合計）")
    pl_now_total = (price - avg_entry) * qty_total
    risk_total   = (price - stop_use) * qty_total if stop_use < price else 0.0
    target_price = price + reward_per_share
    reward_total = (target_price - price) * qty_total

    cA,cB,cC = st.columns(3)
    with cA:
        st.metric("含み損益（いま）", f"{pl_now_total:,.0f} 円", help=f"(現在 {price:.2f} − 平均 {avg_entry:.2f}) × {qty_total:,}")
    with cB:
        st.metric("想定損失（ストップ）", f"{-risk_total:,.0f} 円", help=f"(現在 {price:.2f} − 推奨 {stop_use:.2f}) × {qty_total:,}")
    with cC:
        st.metric("想定利益（目標）", f"{reward_total:,.0f} 円", help=f"(目標 {target_price:.2f} − 現在 {price:.2f}) × {qty_total:,}")
    st.caption(f"※ 目標価格 = 現在値 + {target_atr}×ATR = {target_price:.2f} 円 / RRR ≈ {rrr_now:.2f}")

    st.subheader("追加条件 & RRR（この位置で追加した場合の目安）")
    line = f"- 追加指値候補: **{next_add_trigger:.2f} 円**"
    if require_hi20 and hi20 is not None:
        line += f"（20日高値 {hi20:.2f} 円のブレイクも要求）"
    st.write(line)
    st.write(f"- 想定RRR: **{rrr_now:.2f}**  (risk/株={risk_per_share:.2f}, reward/株≈{reward_per_share:.2f}, 最低目安≥{rrr_min})")

    add_ok = (price >= next_add_trigger) and (rrr_now >= rrr_min)
    if require_hi20 and hi20 is not None:
        add_ok = add_ok and (price >= hi20)
    if add_ok:
        st.info("🟢 追加OK（条件達成）")
    else:
        st.warning("🔸 見送り（条件未達 or RRR不足）")

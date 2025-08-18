# streamlit_app.py — Momentum Chaser Coach (Compact / Big CTA / price→qty, previous-day HI20)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="Momentum Chaser", page_icon="🚀", layout="centered")
st.markdown("## 🚀 Momentum Chaser — ATR / RRR / Trailing Stop")

# ---- CSS: 強調ボタン（スマホ前提） ----
st.markdown("""
<style>
div.stButton > button:first-child {
  height: 3rem;
  font-size: 1.1rem;
  font-weight: 700;
  border-radius: 9999px;
}
</style>
""", unsafe_allow_html=True)

# ========= Utils =========
def disp_symbol(sym: str) -> str:
    return sym[:-2] if sym.endswith(".T") else sym

def normalize_symbol(sym: str) -> str:
    """JP株前提: 入力は '9513' などを想定。内部リクエスト時のみ '.T' を補完。"""
    s = sym.strip().upper()
    return s if "." in s else (s + ".T")

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

def last_row_on_or_before(df: pd.DataFrame, asof: pd.Timestamp, cols=("Close","High","Low","ATR","HI20_PREV")):
    m = (df.index <= asof)
    if not m.any(): return None
    view = df.loc[m].dropna(subset=list(cols))
    if view.empty: return None
    return view.iloc[-1]

# ========= Inputs =========
# 1) 銘柄コード
symbol_in = st.text_input("銘柄コード（例: 9513）", "9513")

# 2) 基準日（過去日で評価）
c_asof, c_date, c_adj = st.columns([1,1,1])
with c_asof:
    asof_switch = st.checkbox("過去日で評価", value=False)
with c_date:
    asof_date = st.date_input("基準日", value=date.today())
with c_adj:
    auto_adj = st.checkbox("調整終値（auto_adjust）", value=False)

# 3) エントリー（価格→株数）— 空の表から入力
st.markdown("#### エントリー（最大5行）")
st.caption("価格 と 株数 を入力。未使用行は空のままでOK。")
seed = pd.DataFrame({"価格": [None, None, None, None, None],
                     "株数": [None, None, None, None, None]})
edited = st.data_editor(
    seed, num_rows="fixed", use_container_width=True, hide_index=True,
    column_config={
        "価格": st.column_config.NumberColumn("価格", format="%.2f", step=0.1, help="約定価格"),
        "株数": st.column_config.NumberColumn("株数", min_value=0, step=1, help="株数（整数）"),
    }
)

# 4) 詳細設定（折りたたみ）
with st.expander("詳細設定", expanded=False):
    p1, p2, p3 = st.columns(3)
    with p1:
        atr_n = st.number_input("ATR期間", 5, 50, 14, 1)
        atr_mult_stop  = st.number_input("初期ストップ(ATR×)", 0.5, 5.0, 2.0, 0.5)
    with p2:
        atr_mult_trail = st.number_input("トレイル倍率(ATR×)", 1.0, 5.0, 2.0, 0.5)
        require_hi20   = st.checkbox("20日高値ブレイクを目安に含める", value=True)
    with p3:
        add_step_atr = st.number_input("次の追加間隔(ATR×)", 0.5, 5.0, 1.0, 0.1)
        target_atr   = st.number_input("目標利幅(ATR×)", 1.0, 10.0, 3.0, 0.5)

# === Big CTA at bottom ===
go = st.button("計算する", use_container_width=True)

# ========= Main =========
if go:
    symbol = normalize_symbol(symbol_in)

    # エントリー抽出（空・NaN行を除外）
    df_in = edited.copy().dropna(how="all").dropna(subset=["価格","株数"])
    try:
        df_in["価格"] = df_in["価格"].astype(float)
        df_in["株数"] = df_in["株数"].astype(int)
    except Exception:
        st.error("エントリー表の値が数値に変換できません。『価格=数値』『株数=整数』で入力してください。")
        st.stop()
    entries = [(float(r["価格"]), int(r["株数"])) for _, r in df_in.iterrows()]
    if not entries:
        st.error("少なくとも1行は『価格・株数』を入力してください。")
        st.stop()

    # データ取得
    df = fetch_history(symbol, back_days=900, auto_adjust=auto_adj)
    if df.empty:
        st.error("価格データを取得できませんでした。銘柄コードをご確認ください。")
        st.stop()
    need = {"Open","High","Low","Close"}
    if not need.issubset(df.columns):
        st.error(f"取得データに必要列が不足しています: {list(df.columns)}")
        st.stop()

    # 指標計算
    df["ATR"] = calc_atr_ewm(df, n=int(atr_n))

    # 20日高値を“前日まで”に統一（Momentum Chaser と同じ基準）
    # 1) 20日ローリング高値（20本そろったときのみ）
    df["HI20_ROLL"] = df["High"].rolling(20, min_periods=20).max()
    # 2) それを1日シフトして「前日まで」の値に
    df["HI20_PREV"] = df["HI20_ROLL"].shift(1)

    # as-of評価
    if asof_switch:
        row = last_row_on_or_before(df, pd.to_datetime(asof_date),
                                    cols=("Close","High","Low","ATR","HI20_PREV"))
        if row is None:
            st.error("指定基準日以前に有効データが見つかりません（データ不足／上場前の可能性）。")
            st.stop()
        eff_date = pd.to_datetime(row.name).date()
    else:
        row = df.dropna(subset=["Close","High","Low","ATR"]).iloc[-1]
        eff_date = pd.to_datetime(row.name).date()

    price = float(row["Close"])
    atr   = float(row["ATR"])
    hi20_prev = float(row["HI20_PREV"]) if pd.notna(row["HI20_PREV"]) else None

    # 集計（可変ロット）
    qty_total = int(sum(q for _, q in entries))
    notional  = float(sum(p*q for p, q in entries))
    avg_entry = notional / max(1, qty_total)

    # ストップ（はしご式 + ATRトレイル の高い方）
    first_price = entries[0][0]
    base_stop   = first_price - atr_mult_stop * atr
    ladder_stop = base_stop
    if len(entries) >= 2:
        prev_price = entries[-2][0]
        ladder_stop = max(ladder_stop, prev_price)
    trail_stop  = (hi20_prev - atr_mult_trail * atr) if hi20_prev is not None else None
    if trail_stop is not None:
        stop_use = max(ladder_stop, trail_stop)
        stop_basis = "max(はしご, ATRトレイル)"
    else:
        stop_use = ladder_stop
        stop_basis = "はしご"

    # 次の追加“目安”（判定は出さない）
    last_entry_price = entries[-1][0]
    next_add_price   = last_entry_price + add_step_atr * atr
    next_add_note = f"（20日高値(前日まで) {hi20_prev:.2f} 円も目安）" if (hi20_prev is not None and require_hi20) else ""

    # RRR（現状ベースの参考値）
    risk_per_share   = max(0.0, price - stop_use)
    reward_per_share = target_atr * atr
    rrr_now = (reward_per_share / risk_per_share) if risk_per_share > 0 else float("inf")
    target_price = price + reward_per_share

    # ===== Display =====
    st.markdown(f"### {disp_symbol(symbol)} — 現在値・指標")
    top = f"**評価日**: {eff_date} / **終値**: {price:.2f} / **ATR({int(atr_n)})**: {atr:.2f}"
    if hi20_prev is not None: top += f" / **20日高値(前日まで)**: {hi20_prev:.2f}"
    st.write(top)
    if asof_switch and eff_date != asof_date:
        st.caption(f"※基準日 {asof_date} は休場・欠損のため、直近営業日の {eff_date} で評価。")

    st.markdown("### ポジション（可変ロット）")
    show_df = pd.DataFrame(
        [{"回": i+1, "価格": p, "株数": q, "金額(約)": int(round(p*q))} for i,(p,q) in enumerate(entries)]
    )
    st.table(show_df.style.format({"価格":"{:.2f}"}))
    st.write(f"- **総株数**: {qty_total:,} 株 / **平均取得**: {avg_entry:.2f} 円 / **総建玉額(約)**: {notional:,.0f} 円")

    st.markdown("### ストップ")
    st.write(f"- 初期ストップ（1st基準）: {base_stop:.2f}")
    st.write(f"- はしご式: {ladder_stop:.2f}")
    st.write(f"- ATRトレイル(20d前日高値 − {atr_mult_trail:.1f}×ATR): {trail_stop:.2f}" if trail_stop is not None else "- ATRトレイル: NA")
    st.success(f"**推奨ストップ（{stop_basis}）: {stop_use:.2f}**")
    st.caption("利益確保モード" if stop_use >= first_price else "—")

    st.markdown("### 含み損益 / リスク&リワード（合計）")
    pl_now_total = (price - avg_entry) * qty_total
    risk_total   = (price - stop_use) * qty_total if stop_use < price else 0.0
    reward_total = (target_price - price) * qty_total
    c1, c2, c3 = st.columns(3)
    c1.metric("含み損益（いま）", f"{pl_now_total:,.0f} 円")
    c2.metric("想定損失（ストップ）", f"{-risk_total:,.0f} 円")
    c3.metric("想定利益（目標）", f"{reward_total:,.0f} 円")
    st.caption(f"※ 目標価格 = 現在値 + {target_atr}×ATR = {target_price:.2f} 円 / RRR ≈ {rrr_now:.2f}")

    st.markdown("### 次の追加（目安）")
    st.write(f"- **候補価格**: {next_add_price:.2f} 円 {next_add_note}")

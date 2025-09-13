# ================== BingX Signal Bot (Render/GitHub one-file) ==================
# Indicators: EMA20/50/200, CCI(20), MACD(12,26,9), KDJ(9,3,3), BB(20,2), Volume
# Entry: Pullback tới EMA20/BB basis (one-sided range). SL: ATR + structure, kẹp theo leverage.
# TP: RR ladder. Score filter >= 70. Chống trùng. Ghim ✅ bằng edit tin gốc + reply TP/SL (ước tính PnL).
# ------------------------------------------------------------------------------

import os, time, traceback, warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ============== DEPS ==============
try:
    import ccxt, pandas as pd, numpy as np, requests
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ccxt", "pandas", "numpy", "requests"])
    import ccxt, pandas as pd, numpy as np, requests

# ================== CONFIG ==================
EXCHANGE_ID        = "bingx"
MARKET_TYPE        = "swap"              # "spot" hoặc "swap"
QUOTE              = "USDT"
ENTRY_TF           = "15m"
LOOKBACK_BARS      = 360
PAIRS_LIMIT        = 80                  # 0 = tất cả
SLEEP_SECONDS      = 20

# Indicators
MA_SHORT, MA_MID, MA_LONG = 20, 50, 200
CCI_LEN = 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_LEN, KDJ_K, KDJ_D = 9, 3, 3
BB_LEN, BB_STD = 20, 2.0
VOL_MA_LEN, VOL_FACTOR = 20, 1.2
USE_MACD_DIVERGENCE = True

# Scoring filter
MIN_SCORE_TO_ALERT  = 80

# Risk & Leverage aware
LEVERAGE            = 20                 # chỉ để tính PnL ước tính (bot KHÔNG đặt lệnh)
MIN_SL_PCT          = max(0.002, 0.25 / LEVERAGE)  # ~1.25% cho 20x
MAX_SL_PCT          = 0.60 / LEVERAGE               # ~3.0% cho 20x
ATR_LEN             = 14
ATR_MULT_SL         = 1.8

# Entry range (one-sided)
ENTRY_RANGE_DOWN_PCT = 0.003             # BUY: entry_low = entry*(1-0.3%), entry_high = entry
ENTRY_RANGE_UP_PCT   = 0.003             # SELL: entry_low = entry, entry_high = entry*(1+0.3%)
MIN_TP_GAP_PCT       = 0.001             # 0.1% để TP1 nằm ngoài vùng entry

# RR ladder
RR_LEVELS           = [1.0, 1.5, 2.0, 2.5, 3.0]

# Telegram (Render ENV: TELEGRAM_TOKEN, CHAT_ID)
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID", "").strip()

# ================== UTILITIES ==================
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def fmt6(x):
    try:
        v = float(x)
        if v == 0: return "0"
        # Giữ độ động cho coin giá rất nhỏ
        if abs(v) < 1e-3: return f"{v:.8f}"
        return f"{v:.6f}"
    except:
        return str(x)

def _tg(url, data):
    try:
        r = requests.post(url, data=data, timeout=15)
        return r.json()
    except Exception as e:
        print("[Telegram] error:", e)
        return {}

def send_message(text, reply_to=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM OFF]\n" + text)
        return None
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if reply_to is not None:
        data["reply_to_message_id"] = reply_to
        data["allow_sending_without_reply"] = True
    r = _tg(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data)
    return (r.get("result") or {}).get("message_id")

def edit_message(msg_id, text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or not msg_id:
        return
    data = {"chat_id": TELEGRAM_CHAT_ID, "message_id": msg_id, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True}
    _tg(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText", data)

def est_profit_pct(entry, exit_px, side, leverage=None):
    if side == "BUY":
        pct = (exit_px - entry) / entry * 100.0
    else:
        pct = (entry - exit_px) / entry * 100.0
    return (pct, pct * leverage) if leverage else (pct, None)

# ================== INDICATORS ==================
def ema(s, l): return s.ewm(span=l, adjust=False).mean()
def sma(s, l): return s.rolling(l, min_periods=max(2, l//2)).mean()

def cci(df, l=20):
    tp = (df["high"]+df["low"]+df["close"])/3.0
    sma_tp = sma(tp, l)
    md = (tp - sma_tp).abs().rolling(l, min_periods=max(2, l//2)).mean()
    return (tp - sma_tp) / (0.015 * (md.replace(0, np.nan) + 1e-12))

def macd(series, f=12,s=26,sg=9):
    ef, es = ema(series,f), ema(series,s)
    line = ef - es
    sig  = ema(line, sg)
    hist = line - sig
    return line, sig, hist

def kdj(h,l,c,n=9,m1=3,m2=3):
    low_min = l.rolling(n, min_periods=max(2,n//2)).min()
    high_max= h.rolling(n, min_periods=max(2,n//2)).max()
    rsv = (c - low_min) / (high_max - low_min + 1e-12) * 100
    k = rsv.rolling(m1, min_periods=max(2,m1//2)).mean()
    d = k.rolling(m2, min_periods=max(2,m2//2)).mean()
    j = 3*k - 2*d
    return k,d,j

def bollinger(c, l=20, std=2.0):
    basis = c.rolling(l, min_periods=max(2,l//2)).mean()
    dev   = c.rolling(l, min_periods=max(2,l//2)).std(ddof=0)
    up    = basis + std * dev
    dn    = basis - std * dev
    pctb  = (c - dn) / (up - dn + 1e-12)
    bw    = (up - dn) / (basis.abs() + 1e-12)
    return basis, up, dn, pctb, bw

def atr(df,l=14):
    h,lw,c = df["high"], df["low"], df["close"]
    hl = h - lw
    hc = (h - c.shift()).abs()
    lc = (lw - c.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(l, min_periods=max(2,l//2)).mean()

def crossed_up(a, b):
    return len(a)>2 and len(b)>2 and a.iloc[-3] <= b.iloc[-3] and a.iloc[-2] > b.iloc[-2]

def crossed_down(a, b):
    return len(a)>2 and len(b)>2 and a.iloc[-3] >= b.iloc[-3] and a.iloc[-2] < b.iloc[-2]

def find_pivots(s, win=5, lb=80, mode="low"):
    start=max(0,len(s)-lb); sub=s.iloc[start:]
    idx=[]
    for i in range(win, len(sub)-win):
        w=sub.iloc[i-win:i+win+1]
        if mode=="low" and sub.iloc[i]==w.min(): idx.append(sub.index[i])
        if mode=="high"and sub.iloc[i]==w.max(): idx.append(sub.index[i])
    return (idx[-2], idx[-1]) if len(idx)>=2 else (None,None)

def macd_divergence(df):
    p1,p2=find_pivots(df["close"],mode="low")
    bull = bool(p1 and p2 and df.loc[p2,"close"]<df.loc[p1,"close"] and df.loc[p2,"macd"]>df.loc[p1,"macd"])
    q1,q2=find_pivots(df["close"],mode="high")
    bear = bool(q1 and q2 and df.loc[q2,"close"]>df.loc[q1,"close"] and df.loc[q2,"macd"]<df.loc[q1,"macd"])
    return bull,bear

# ================== STRATEGY ==================
def compute_indicators(df):
    df=df.copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["ma_s"]=ema(df["close"],MA_SHORT)
    df["ma_m"]=ema(df["close"],MA_MID)
    df["ma_l"]=ema(df["close"],MA_LONG)
    df["cci"]=cci(df,CCI_LEN)
    df["macd"],df["macd_sig"],df["macd_hist"]=macd(df["close"],MACD_FAST,MACD_SLOW,MACD_SIGNAL)
    df["vol_ma"]=sma(df["volume"],VOL_MA_LEN)
    df["kdj_k"],df["kdj_d"],df["kdj_j"]=kdj(df["high"],df["low"],df["close"],KDJ_LEN,KDJ_K,KDJ_D)
    df["bb_basis"],df["bb_up"],df["bb_dn"],df["bb_pctb"],df["bb_bw"]=bollinger(df["close"],BB_LEN,BB_STD)
    return df.dropna().reset_index(drop=True)

def ma_aligned_up(r):   return r["ma_s"] > r["ma_m"] > r["ma_l"]
def ma_aligned_down(r): return r["ma_s"] < r["ma_m"] < r["ma_l"]

def score_signal(df, side):
    r=df.iloc[-2]
    pts,reasons=0,[]
    if side=="BUY":
        if ma_aligned_up(r): pts+=35; reasons.append("MA↑ (20>50>200)")
        if r["macd"]>0: pts+=20; reasons.append("MACD>0")
        if r["cci"]>0:  pts+=15; reasons.append("CCI>0")
        if r["macd_hist"]>df["macd_hist"].iloc[-3]: pts+=6; reasons.append("Hist↑")
        if r["kdj_k"]>r["kdj_d"] and r["kdj_j"]>df["kdj_j"].iloc[-3]: pts+=10; reasons.append("KDJ↑")
        if r["close"]>r["bb_basis"]: pts+=6; reasons.append("BB trên basis")
        if crossed_up(df["close"], df["bb_up"]): pts+=6; reasons.append("Break ↑ BB")
        if r["volume"]>=r["vol_ma"]*VOL_FACTOR: pts+=6; reasons.append("Vol↑")
        if USE_MACD_DIVERGENCE and macd_divergence(df)[0]: pts+=6; reasons.append("Bull div")
    else:
        if ma_aligned_down(r): pts+=35; reasons.append("MA↓ (20<50<200)")
        if r["macd"]<0: pts+=20; reasons.append("MACD<0")
        if r["cci"]<0:  pts+=15; reasons.append("CCI<0")
        if r["macd_hist"]<df["macd_hist"].iloc[-3]: pts+=6; reasons.append("Hist↓")
        if r["kdj_k"]<r["kdj_d"] and r["kdj_j"]<df["kdj_j"].iloc[-3]: pts+=10; reasons.append("KDJ↓")
        if r["close"]<r["bb_basis"]: pts+=6; reasons.append("BB dưới basis")
        if crossed_down(df["close"], df["bb_dn"]): pts+=6; reasons.append("Break ↓ BB")
        if r["volume"]>=r["vol_ma"]*VOL_FACTOR: pts+=6; reasons.append("Vol↑")
        if USE_MACD_DIVERGENCE and macd_divergence(df)[1]: pts+=6; reasons.append("Bear div")
    return min(100,int(pts)), reasons

def last_swing_low(df, win=5):
    s=df["low"]
    for i in range(len(s)-win-2, win, -1):
        if s.iloc[i] == s.iloc[i-win:i+win+1].min():
            return float(s.iloc[i])
    return float(df["low"].iloc[-5])

def last_swing_high(df, win=5):
    s=df["high"]
    for i in range(len(s)-win-2, win, -1):
        if s.iloc[i] == s.iloc[i-win:i+win+1].max():
            return float(s.iloc[i])
    return float(df["high"].iloc[-5])

def build_trade_plan(df, side, entry_close):
    r = df.iloc[-2]
    a = float(atr(df, ATR_LEN).iloc[-2])

    if side=="BUY":
        entry_anchor = max(float(r["ma_s"]), float(r["bb_basis"]))
        entry = min(float(entry_close), entry_anchor)
        entry_low  = entry * (1 - ENTRY_RANGE_DOWN_PCT)
        entry_high = entry
        swing = last_swing_low(df)
        sl_struct = min(swing, float(r["bb_dn"]))
        sl_atr = entry - max(a*ATR_MULT_SL, entry*MIN_SL_PCT)
        sl = min(sl_struct - a*0.2, sl_atr)
        sl = max(sl, entry - entry*MAX_SL_PCT)
    else:
        entry_anchor = min(float(r["ma_s"]), float(r["bb_basis"]))
        entry = max(float(entry_close), entry_anchor)
        entry_low  = entry
        entry_high = entry * (1 + ENTRY_RANGE_UP_PCT)
        swing = last_swing_high(df)
        sl_struct = max(swing, float(r["bb_up"]))
        sl_atr = entry + max(a*ATR_MULT_SL, entry*MIN_SL_PCT)
        sl = max(sl_struct + a*0.2, sl_atr)
        sl = min(sl, entry + entry*MAX_SL_PCT)

    risk = abs(entry - sl)
    raw_tps = [entry + risk*rr if side=="BUY" else entry - risk*rr for rr in RR_LEVELS]

    if side=="BUY":
        tps = [max(tp, entry_high*(1+MIN_TP_GAP_PCT)) for tp in raw_tps]
    else:
        tps = [min(tp, entry_low*(1-MIN_TP_GAP_PCT)) for tp in raw_tps]

    return {
        "entry": entry,
        "entry_low": min(entry_low, entry_high),
        "entry_high": max(entry_low, entry_high),
        "sl": sl,
        "tps": tps,
        "hit": [False]*len(tps),
        "risk_dist": risk
    }

def plan_is_valid(side, plan):
    if side=="BUY":
        return plan["sl"] < plan["entry_low"] < plan["entry_high"] < min(plan["tps"])
    else:
        return plan["sl"] > plan["entry_high"] > plan["entry_low"] > max(plan["tps"])

# ================== TELEGRAM TEMPLATES ==================
LEVERAGE_TEXT = f"LEV/{LEVERAGE}X"

def format_signal(sym, side, plan, score, reasons):
    emoji = "🟢" if side=="BUY" else "🔴"
    lines = [
        f"{emoji} <b>{side}</b> #{sym}  {LEVERAGE_TEXT}",
        f"ENTRY {fmt6(plan['entry_low'])}–{fmt6(plan['entry_high'])}",
        f"STOPLOSS {fmt6(plan['sl'])}",
        "TARGET"
    ]
    for i,(tp,hit) in enumerate(zip(plan["tps"], plan["hit"]), 1):
        lines.append(f"{i}  {fmt6(tp)} {'✅' if hit else ''}")
    if score is not None:
        lines += ["", f"Score {score}/100"]
    if reasons:
        lines += [f"Lý do: {', '.join(reasons)}"]
    lines += [f"🕒 {now_utc_iso()}"]
    return "\n".join(lines)

def format_tp_reply(sym, side, plan, i_hit, px):
    spot, lev = est_profit_pct(plan["entry"], plan["tps"][i_hit], side, leverage=LEVERAGE)
    return (f"✅ <b>TP{i_hit+1}</b> hit #{sym} @ {fmt6(px)}\n"
            f"Entry {fmt6(plan['entry'])} | SL {fmt6(plan['sl'])}\n"
            f"PnL ước tính: {spot:.2f}%"
            + (f"  (~{lev:.2f}% x{LEVERAGE})" if lev is not None else ""))

def format_sl_reply(sym, side, plan, px):
    spot, lev = est_profit_pct(plan["entry"], plan["sl"], side, leverage=LEVERAGE)
    return (f"❌ <b>SL</b> hit #{sym} @ {fmt6(px)}\n"
            f"Entry {fmt6(plan['entry'])}\n"
            f"Lỗ ước tính: {spot:.2f}%"
            + (f"  (~{lev:.2f}% x{LEVERAGE})" if lev is not None else ""))

def format_all_done(sym, side, plan):
    spot, lev = est_profit_pct(plan["entry"], plan["tps"][-1], side, leverage=LEVERAGE)
    return (f"🏁 <b>ALL TARGET DONE</b> #{sym}\n"
            f"PnL ước tính tới TP{len(plan['tps'])}: {spot:.2f}%"
            + (f"  (~{lev:.2f}% x{LEVERAGE})" if lev is not None else ""))

# ================== EXCHANGE ==================
def make_ex():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "options": {"defaultType": MARKET_TYPE}})
    ex.load_markets()
    return ex

def list_syms(ex):
    syms=[]
    for s,m in ex.markets.items():
        if m.get("quote")==QUOTE and m.get("active",True):
            if MARKET_TYPE=="spot" and m.get("spot"): syms.append(s)
            elif MARKET_TYPE=="swap" and (m.get("swap") or m.get("future")): syms.append(s)
    syms = sorted(set(syms))
    if PAIRS_LIMIT>0: syms = syms[:PAIRS_LIMIT]
    return syms

def fetch_df(ex,sym,tf=ENTRY_TF,limit=LOOKBACK_BARS):
    try:
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=limit)
        if not ohlcv or len(ohlcv)<220: return None
        df=pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["ts"]=pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception:
        return None

# ================== MAIN LOOP ==================
last_signal = {}   # {"SYMBOL": "BUY"/"SELL"}
open_trades = {}   # {"SYMBOL": {"side":..., "plan":..., "msg_id":..., "score":..., "reasons":[...]}}
last_ts = {}

def run():
    ex = make_ex()
    syms = list_syms(ex)
    print(f"[init] symbols={len(syms)} | TF={ENTRY_TF} | lev={LEVERAGE}x")

    while True:
        try:
            for sym in syms:
                df_raw = fetch_df(ex, sym)
                if df_raw is None: continue

                # chỉ xử lý nến đóng
                ts = df_raw["ts"].iloc[-2]
                if last_ts.get(sym)==ts: continue
                last_ts[sym]=ts

                df = compute_indicators(df_raw)
                if len(df)<220: continue
                row = df.iloc[-2]

                # Lọc trend/volume để giảm nhiễu
                sep = abs(row["ma_s"] - row["ma_m"]) / max(1e-12, abs(row["ma_m"]))
                strong_trend = sep >= 0.003          # >= 0.3%
                vol_ok = row["volume"] >= row["vol_ma"]

                buy_base  = ma_aligned_up(row)   and row["cci"]>0 and row["macd"]>0 and strong_trend and vol_ok
                sell_base = ma_aligned_down(row) and row["cci"]<0 and row["macd"]<0 and strong_trend and vol_ok
                side = "BUY" if buy_base else ("SELL" if sell_base else None)

                # ========= NEW SIGNAL =========
                if side:
                    score, reasons = score_signal(df, side)
                    if score >= MIN_SCORE_TO_ALERT and last_signal.get(sym) != side:
                        last_signal[sym] = side
                        entry_close = float(row["close"])
                        plan = build_trade_plan(df, side, entry_close)
                        if plan_is_valid(side, plan):
                            msg_text = format_signal(sym, side, plan, score, reasons)
                            msg_id = send_message(msg_text)
                            open_trades[sym] = {"side": side, "plan": plan, "msg_id": msg_id,
                                                "score": score, "reasons": reasons}

                # ========= TRACK OPEN TRADES =========
                if sym in open_trades:
                    trade = open_trades[sym]
                    plan  = trade["plan"]
                    px    = float(row["close"])
                    changed = False

                    # SL trước
                    if trade["side"]=="BUY" and px <= plan["sl"]:
                        send_message(format_sl_reply(sym, trade["side"], plan, px), reply_to=trade["msg_id"])
                        del open_trades[sym]
                        continue
                    if trade["side"]=="SELL" and px >= plan["sl"]:
                        send_message(format_sl_reply(sym, trade["side"], plan, px), reply_to=trade["msg_id"])
                        del open_trades[sym]
                        continue

                    # TP ladder + trailing SL
                    for i,tp in enumerate(plan["tps"]):
                        if not plan["hit"][i]:
                            if (trade["side"]=="BUY" and px>=tp) or (trade["side"]=="SELL" and px<=tp):
                                plan["hit"][i]=True; changed=True
                                send_message(format_tp_reply(sym, trade["side"], plan, i, px),
                                             reply_to=trade["msg_id"])
                                # Move SL: TP1 -> entry, TP2 -> TP1
                                if i>=0:
                                    if trade["side"]=="BUY":
                                        plan["sl"] = max(plan["sl"], plan["entry"])
                                    else:
                                        plan["sl"] = min(plan["sl"], plan["entry"])
                                if i>=1:
                                    if trade["side"]=="BUY":
                                        plan["sl"] = max(plan["sl"], plan["tps"][0])
                                    else:
                                        plan["sl"] = min(plan["sl"], plan["tps"][0])

                    if changed:
                        edit_message(trade["msg_id"],
                                     format_signal(sym, trade["side"], plan, trade["score"], trade["reasons"]))
                        if all(plan["hit"]):
                            send_message(format_all_done(sym, trade["side"], plan), reply_to=trade["msg_id"])
                            del open_trades[sym]

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("⛔ Stopped by user"); break
        except Exception as e:
            print("Loop error:", e); traceback.print_exc(); time.sleep(5)

# ================== START ==================
if __name__ == "__main__":
    run()

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ── Fetch BTC/USDT 1H data from Binance ──────────────────────────────────────
def fetch_binance_ohlcv(symbol, interval, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current = start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current, "endTime": end_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=20)
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_candles.extend(data)
        current = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(all_candles, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.set_index("open_time")

def ts(dt_str):
    return int(datetime.fromisoformat(dt_str).replace(
        tzinfo=timezone.utc).timestamp() * 1000)

print("Descargando datos BTC/USDT 1H (2019-01 → 2024-12)…")
df = fetch_binance_ohlcv("BTCUSDT", "1h", ts("2019-01-01"), ts("2024-12-31 23:59:59"))
print(f"  {len(df):,} velas descargadas")

# ── Fetch SPY 1H from Binance (no disponible → usar proxy BTC/ETH ratio) ─────
# SPY no está en Binance; usamos un proxy: solo filtramos con la tendencia
# macro de BTC de marco mensual (close > SMA200 del propio BTC en 1H)
# Esto aproxima la lógica del filtro macro con datos disponibles.

# ── Indicadores ───────────────────────────────────────────────────────────────
df["sma20"]  = df["close"].rolling(20).mean()
df["sma50"]  = df["close"].rolling(50).mean()
df["sma100"] = df["close"].rolling(100).mean()
df["sma200"] = df["close"].rolling(200).mean()

# RSI
delta  = df["close"].diff()
gain   = delta.clip(lower=0).rolling(14).mean()
loss   = (-delta.clip(upper=0)).rolling(14).mean()
rs     = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - 100 / (1 + rs)

# MACD
ema12  = df["close"].ewm(span=12, adjust=False).mean()
ema26  = df["close"].ewm(span=26, adjust=False).mean()
df["macd"]   = ema12 - ema26
df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["hist"]   = df["macd"] - df["signal"]

# ATR
hl  = df["high"] - df["low"]
hpc = (df["high"] - df["close"].shift()).abs()
lpc = (df["low"]  - df["close"].shift()).abs()
df["atr"]     = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
df["atr_mean"]= df["atr"].rolling(50).mean()

# ── Condiciones ───────────────────────────────────────────────────────────────
trend   = ((df["sma20"] > df["sma50"]) & (df["sma50"] > df["sma100"]) &
           (df["sma100"] > df["sma200"]) & (df["close"] > df["sma50"]))

momentum = ((df["rsi"] > 55) & (df["macd"] > df["signal"]) &
            (df["hist"] > df["hist"].shift(1)))

volatility = df["atr"] > df["atr_mean"]

# Macro filter proxy: BTC close > su propia SMA200 (equivale a bull estructural)
macro = df["close"] > df["sma200"]

# Pullback: low actual o anterior tocó SMA20
pullback = (df["low"] <= df["sma20"]) | (df["low"].shift(1) <= df["sma20"].shift(1))

# Filtro mecha
candle_range = df["high"] - df["low"]
upper_wick   = df["high"] - df[["open","close"]].max(axis=1)
lower_wick   = df[["open","close"]].min(axis=1) - df["low"]
wick_ok      = (candle_range > 0) & (upper_wick/candle_range < 0.6) & (lower_wick/candle_range < 0.6)

# Ruptura de vela anterior
breakout = df["close"] > df["high"].shift(1)

df["long_signal"] = trend & momentum & volatility & macro & pullback & wick_ok & breakout
df.dropna(inplace=True)

# ── Backtest ──────────────────────────────────────────────────────────────────
capital    = 10_000.0
position   = 0.0
entry_px   = 0.0
stop_px    = 0.0
tp_px      = 0.0
trades     = []
in_trade   = False
equity_curve = []

for idx, row in df.iterrows():
    equity = capital + position * row["close"] if in_trade else capital
    equity_curve.append(equity)

    if in_trade:
        # Check SL / TP on this bar (using high/low)
        hit_sl = row["low"] <= stop_px
        hit_tp = row["high"] >= tp_px

        if hit_sl and hit_tp:
            # Both hit: conservatively assign SL (worse case)
            exit_px = stop_px
            hit = "SL"
        elif hit_sl:
            exit_px = stop_px
            hit = "SL"
        elif hit_tp:
            exit_px = tp_px
            hit = "TP"
        else:
            exit_px = None

        if exit_px:
            pnl_pct = (exit_px - entry_px) / entry_px
            pnl_usd = position * (exit_px - entry_px)
            capital += pnl_usd
            trades.append({
                "entry_time": entry_time,
                "exit_time":  idx,
                "entry_px":   entry_px,
                "exit_px":    exit_px,
                "exit_type":  hit,
                "pnl_pct":    pnl_pct * 100,
                "pnl_usd":    pnl_usd,
                "capital":    capital,
            })
            position = 0.0
            in_trade = False

    if not in_trade and row["long_signal"]:
        risk_pct  = 0.01  # 1% of equity per trade
        stop_dist = row["atr"] * 1.2
        position  = (capital * risk_pct) / stop_dist * row["close"] / row["close"]
        # invest 1% of equity as notional
        position  = (capital * 0.01) / row["close"]
        entry_px  = row["close"]
        stop_px   = row["close"] - row["atr"] * 1.2
        tp_px     = row["close"] + row["atr"] * 1.8
        entry_time= idx
        in_trade  = True

# ── Métricas ──────────────────────────────────────────────────────────────────
trades_df = pd.DataFrame(trades)

if trades_df.empty:
    print("Sin trades en el período.")
else:
    n          = len(trades_df)
    winners    = trades_df[trades_df["pnl_usd"] > 0]
    losers     = trades_df[trades_df["pnl_usd"] <= 0]
    win_rate   = len(winners) / n * 100
    avg_win    = winners["pnl_usd"].mean() if len(winners) else 0
    avg_loss   = losers["pnl_usd"].mean()  if len(losers)  else 0
    rr         = abs(avg_win / avg_loss)   if avg_loss != 0 else np.nan
    total_pnl  = trades_df["pnl_usd"].sum()
    final_cap  = 10_000 + total_pnl
    pnl_pct    = total_pnl / 10_000 * 100

    # Max drawdown on equity curve
    eq  = pd.Series(equity_curve)
    roll_max = eq.cummax()
    dd  = (eq - roll_max) / roll_max * 100
    max_dd = dd.min()

    # By year
    trades_df["year"] = trades_df["exit_time"].dt.year
    by_year = trades_df.groupby("year").agg(
        trades=("pnl_usd","count"),
        pnl_usd=("pnl_usd","sum"),
        win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100)
    ).round(2)

    print("\n" + "="*55)
    print("  BTC TENDENCIAL 1H PRO — BACKTEST 2019-2024")
    print("="*55)
    print(f"  Capital inicial :  $10,000.00")
    print(f"  Capital final   :  ${final_cap:>10,.2f}")
    print(f"  PnL total       :  ${total_pnl:>10,.2f}  ({pnl_pct:+.2f}%)")
    print(f"  Max Drawdown    :  {max_dd:.2f}%")
    print("-"*55)
    print(f"  Total trades    :  {n}")
    print(f"  Ganadores       :  {len(winners)}  ({win_rate:.1f}%)")
    print(f"  Perdedores      :  {len(losers)}")
    print(f"  Avg win         :  ${avg_win:,.2f}")
    print(f"  Avg loss        :  ${avg_loss:,.2f}")
    print(f"  Ratio R:R       :  {rr:.2f}")
    print(f"  TP hits         :  {(trades_df['exit_type']=='TP').sum()}")
    print(f"  SL hits         :  {(trades_df['exit_type']=='SL').sum()}")
    print("="*55)
    print("\n  Resultados por año:")
    print(by_year.to_string())
    print()

    # Top 5 mejores y peores trades
    print("\n  Top 5 mejores trades:")
    print(trades_df.nlargest(5,"pnl_usd")[["entry_time","exit_time","entry_px","exit_px","pnl_usd","pnl_pct"]].to_string(index=False))
    print("\n  Top 5 peores trades:")
    print(trades_df.nsmallest(5,"pnl_usd")[["entry_time","exit_time","entry_px","exit_px","pnl_usd","pnl_pct"]].to_string(index=False))

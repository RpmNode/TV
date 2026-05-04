"""
PO3 Strategy Backtest — BTC/USDT 1H | Jan 2017 – Dec 2021
Replica exacta de la lógica del PO3_Strategy.pine + optimizaciones v2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS BASE
# ─────────────────────────────────────────────────────────────────────────────
BASE_PARAMS = dict(
    use_pdhl     = True,
    use_pwhl     = True,
    use_eqhl     = True,
    eq_look      = 10,
    eq_tol       = 0.15,
    vol_mult     = 1.5,
    vol_len      = 20,
    wick_min_pct = 0.20,
    use_ob       = True,
    use_fvg      = True,
    need_conf    = False,
    ob_look      = 3,
    fvg_min_pct  = 0.05,
    rr           = 2.5,
    sl_buf_pct   = 0.10,
    max_day_tr   = 2,
    trend_filter = False,     # NUEVO: solo longs en uptrend / shorts en downtrend
    trend_period = 200,       # NUEVO: SMA período (en barras 1H)
    commission   = 0.05,
    initial_cap  = 10_000,
    qty_pct      = 100,
)

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol="BTC/USDT", tf="1h",
                start="2017-01-01", end="2022-01-01"):
    """
    Genera datos sintéticos calibrados con los waypoints históricos reales de BTC.
    Usa GBM (Geometric Brownian Motion) con cambios de régimen para reproducir:
      - Bull run 2017 (1k → 20k)
      - Bear market 2018 (20k → 3.2k)
      - Recovery/crash 2019-2020 (3.2k → 5k via 13k + COVID crash)
      - Bull run 2020-2021 (5k → 69k)
    """
    print(f"Generando datos sintéticos calibrados BTC/USDT 1H (2017-2021)...")
    np.random.seed(42)

    # Waypoints (fecha, precio, vol_diaria, vol_diaria_base)
    waypoints = [
        ("2017-01-01",  1_000,  0.025),
        ("2017-06-01",  2_800,  0.035),
        ("2017-12-17", 20_000,  0.060),
        ("2018-02-01",  8_500,  0.055),
        ("2018-12-15",  3_200,  0.045),
        ("2019-06-26", 13_800,  0.040),
        ("2019-12-01",  7_200,  0.030),
        ("2020-03-13",  4_900,  0.080),   # COVID crash
        ("2020-08-01", 11_800,  0.030),
        ("2020-12-31", 29_000,  0.040),
        ("2021-04-14", 64_000,  0.055),
        ("2021-07-20", 29_800,  0.045),
        ("2021-11-10", 69_000,  0.050),
        ("2022-01-01", 46_000,  0.040),
    ]

    dates  = pd.date_range(start, end, freq="h", tz="UTC")[:-1]
    prices = np.zeros(len(dates))
    vols   = np.zeros(len(dates))

    wdts = [pd.Timestamp(w[0], tz="UTC") for w in waypoints]
    wpx  = [w[1] for w in waypoints]
    wvl  = [w[2] for w in waypoints]

    for seg in range(len(wdts) - 1):
        t0, t1 = wdts[seg], wdts[seg + 1]
        p0, p1 = wpx[seg], wpx[seg + 1]
        v0, v1 = wvl[seg], wvl[seg + 1]
        mask = (dates >= t0) & (dates < t1)
        n    = mask.sum()
        if n == 0:
            continue
        h_vol = v0 / np.sqrt(24)   # volatilidad horaria
        drift = (np.log(p1 / p0) - 0.5 * h_vol**2 * n) / n
        shocks = np.random.normal(drift, h_vol, n)
        # fat tails ocasionales (2% de bares con shock x2.5)
        jumps = np.random.rand(n) < 0.02
        shocks[jumps] *= 2.5
        path = p0 * np.exp(np.cumsum(shocks))
        prices[mask] = path
        vols[mask]   = np.linspace(v0, v1, n)

    prices = np.where(prices == 0, np.nan, prices)
    prices = pd.Series(prices).ffill().bfill().values

    # Construir OHLCV realista a partir del close sintético
    rows = []
    base_vol_btc = 15_000   # volumen base en BTC/hora
    for i, (dt, c) in enumerate(zip(dates, prices)):
        if i == 0:
            o = c
        else:
            o = prices[i - 1]
        spread = c * vols[i] / np.sqrt(24) * abs(np.random.normal(1, 0.5))
        h = max(o, c) + spread * abs(np.random.normal(0.6, 0.3))
        l = min(o, c) - spread * abs(np.random.normal(0.6, 0.3))
        h = max(h, o, c)
        l = min(l, o, c)
        # Sesiones de volumen (Asia menor, Londres/NY mayor)
        hour = dt.hour
        vol_mult_h = 0.6 if 0 <= hour < 7 else 1.4 if 8 <= hour < 16 else 1.0
        vol = base_vol_btc * vol_mult_h * abs(np.random.lognormal(0, 0.5))
        rows.append({"dt": dt, "open": o, "high": h, "low": l,
                     "close": c, "volume": vol})

    df = pd.DataFrame(rows)
    print(f"Total velas: {len(df):,}  ({df['dt'].iloc[0]} → {df['dt'].iloc[-1]})")
    print(f"Precio inicio: ${df['close'].iloc[0]:,.0f}  |  "
          f"Precio fin: ${df['close'].iloc[-1]:,.0f}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────────────────────────────────────

def pct(p, x):
    return p * x / 100

def add_indicators(df, p):
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume

    # Previous Day High/Low
    df["date"] = df["dt"].dt.floor("D")
    day_hl = df.groupby("date")[["high","low"]].agg({"high":"max","low":"min"})
    day_hl.index = pd.to_datetime(day_hl.index, utc=True)
    df["pdh"] = df["date"].map(day_hl["high"].shift(1, freq="D"))
    df["pdl"] = df["date"].map(day_hl["low"].shift(1,  freq="D"))

    # Previous Week High/Low
    df["week"] = df["dt"].dt.to_period("W").apply(lambda x: x.start_time).dt.tz_localize("UTC")
    wk_hl = df.groupby("week")[["high","low"]].agg({"high":"max","low":"min"})
    df["pwh"] = df["week"].map(wk_hl["high"].shift(1, freq="W"))
    df["pwl"] = df["week"].map(wk_hl["low"].shift(1,  freq="W"))

    # Volume SMA
    df["vol_avg"] = v.rolling(p["vol_len"]).mean()
    df["inst_vol"] = v >= df["vol_avg"] * p["vol_mult"]

    # Wicks
    df["lower_wick"] = np.minimum(o, c) - l
    df["upper_wick"] = h - np.maximum(o, c)
    df["bull_wick_ok"] = df["lower_wick"] >= pct(p["wick_min_pct"], l)
    df["bear_wick_ok"] = df["upper_wick"] >= pct(p["wick_min_pct"], h)

    # Trend filter: SMA sobre cierre
    df["sma_trend"] = c.rolling(p["trend_period"]).mean()
    df["uptrend"]   = c > df["sma_trend"]
    df["downtrend"] = c < df["sma_trend"]

    return df

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST BAR-BY-BAR
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df, p):
    equity      = p["initial_cap"]
    trades      = []
    position    = None   # dict con entry, sl, tp, side
    day_count   = 0
    prev_day    = None

    # EQH/EQL state
    eqh_level = np.nan
    eql_level = np.nan

    # FVG state
    bfvg_lo = bfvg_hi = np.nan
    sfvg_lo = sfvg_hi = np.nan
    bfvg_ok = sfvg_ok = False

    # OB state
    bob_lo = bob_hi = np.nan
    soo_lo = soo_hi = np.nan
    bob_ok = soo_ok = False

    N = len(df)

    for i in range(max(p["vol_len"], p["eq_look"], p["ob_look"]) + 2, N):
        row = df.iloc[i]
        o, h, l, c, v = row.open, row.high, row.low, row.close, row.volume

        # Reset diario
        cur_day = row["dt"].date()
        if cur_day != prev_day:
            day_count = 0
            prev_day  = cur_day

        # ── Cerrar posición si SL o TP tocados ──────────────────────────────
        if position is not None:
            hit_sl = hit_tp = False
            if position["side"] == "long":
                if l <= position["sl"]:
                    hit_sl = True
                    exit_px = position["sl"]
                elif h >= position["tp"]:
                    hit_tp = True
                    exit_px = position["tp"]
            else:  # short
                if h >= position["sl"]:
                    hit_sl = True
                    exit_px = position["sl"]
                elif l <= position["tp"]:
                    hit_tp = True
                    exit_px = position["tp"]

            if hit_sl or hit_tp:
                entry_px = position["entry"]
                size     = position["size"]          # BTC
                if position["side"] == "long":
                    pnl_pct = (exit_px - entry_px) / entry_px
                else:
                    pnl_pct = (entry_px - exit_px) / entry_px
                gross   = size * exit_px
                comm    = gross * p["commission"] / 100
                net_pnl = size * entry_px * pnl_pct - comm
                equity += net_pnl
                trades.append({
                    "entry_dt":  position["entry_dt"],
                    "exit_dt":   row["dt"],
                    "side":      position["side"],
                    "entry_px":  entry_px,
                    "exit_px":   exit_px,
                    "pnl":       net_pnl,
                    "pnl_pct":   pnl_pct * 100,
                    "result":    "TP" if hit_tp else "SL",
                    "equity":    equity,
                })
                position = None

        # ── EQH / EQL ────────────────────────────────────────────────────────
        eqh_hit = eql_hit = False
        if p["use_eqhl"]:
            for k in range(1, p["eq_look"] + 1):
                ph = df.iloc[i - k].high
                pl = df.iloc[i - k].low
                if abs(h - ph) <= pct(p["eq_tol"], h):
                    eqh_hit   = True
                    eqh_level = max(h, ph)
                if abs(l - pl) <= pct(p["eq_tol"], l):
                    eql_hit   = True
                    eql_level = min(l, pl)

        # ── Sweep detection ──────────────────────────────────────────────────
        pdh = row["pdh"] if p["use_pdhl"] else np.nan
        pdl = row["pdl"] if p["use_pdhl"] else np.nan
        pwh = row["pwh"] if p["use_pwhl"] else np.nan
        pwl = row["pwl"] if p["use_pwhl"] else np.nan

        bull_wick_ok = row["bull_wick_ok"]
        bear_wick_ok = row["bear_wick_ok"]
        inst_vol     = row["inst_vol"]

        sweep_pdl = p["use_pdhl"] and not np.isnan(pdl) and l < pdl and c > pdl and bull_wick_ok
        sweep_pwl = p["use_pwhl"] and not np.isnan(pwl) and l < pwl and c > pwl and bull_wick_ok
        sweep_eql = p["use_eqhl"] and eql_hit and l < eql_level and c > eql_level and bull_wick_ok

        sweep_pdh = p["use_pdhl"] and not np.isnan(pdh) and h > pdh and c < pdh and bear_wick_ok
        sweep_pwh = p["use_pwhl"] and not np.isnan(pwh) and h > pwh and c < pwh and bear_wick_ok
        sweep_eqh = p["use_eqhl"] and eqh_hit and h > eqh_level and c < eqh_level and bear_wick_ok

        bull_sweep = (sweep_pdl or sweep_pwl or sweep_eql) and inst_vol
        bear_sweep = (sweep_pdh or sweep_pwh or sweep_eqh) and inst_vol

        # ── FVG update ───────────────────────────────────────────────────────
        if p["use_fvg"] and i >= 2:
            prev1 = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]
            fvg_min_sz = pct(p["fvg_min_pct"], c)
            bull_fvg_new = (prev1.close > prev1.open and
                            l > prev2.high and
                            (l - prev2.high) >= fvg_min_sz)
            bear_fvg_new = (prev1.close < prev1.open and
                            h < prev2.low and
                            (prev2.low - h) >= fvg_min_sz)
            if bull_fvg_new:
                bfvg_lo = prev2.high; bfvg_hi = l; bfvg_ok = True
            if bear_fvg_new:
                sfvg_lo = h; sfvg_hi = prev2.low; sfvg_ok = True
            if bfvg_ok and l < bfvg_lo:
                bfvg_ok = False
            if sfvg_ok and h > sfvg_hi:
                sfvg_ok = False

        # ── OB update ────────────────────────────────────────────────────────
        if p["use_ob"] and i >= p["ob_look"]:
            bull_impulse = c > o and c > df.iloc[i-1].close * 1.002
            bear_impulse = c < o and c < df.iloc[i-1].close * 0.998
            ob_row = df.iloc[i - p["ob_look"]]
            if bull_impulse and ob_row.close < ob_row.open:
                bob_lo = ob_row.close; bob_hi = ob_row.open; bob_ok = True
            if bear_impulse and ob_row.close > ob_row.open:
                soo_lo = ob_row.open; soo_hi = ob_row.close; soo_ok = True
            if bob_ok and l < bob_lo:
                bob_ok = False
            if soo_ok and h > soo_hi:
                soo_ok = False

        # ── Confluencia ──────────────────────────────────────────────────────
        bull_ob_in  = p["use_ob"]  and bob_ok and bob_lo <= c <= bob_hi
        bull_fvg_in = p["use_fvg"] and bfvg_ok and bfvg_lo <= c <= bfvg_hi
        bear_ob_in  = p["use_ob"]  and soo_ok and soo_lo <= c <= soo_hi
        bear_fvg_in = p["use_fvg"] and sfvg_ok and sfvg_lo <= c <= sfvg_hi

        bull_conf = bull_ob_in or bull_fvg_in
        bear_conf = bear_ob_in or bear_fvg_in

        # ── Filtro de tendencia ──────────────────────────────────────────────
        uptrend   = row["uptrend"]
        downtrend = row["downtrend"]
        trend_ok_long  = (not p["trend_filter"]) or uptrend
        trend_ok_short = (not p["trend_filter"]) or downtrend

        # ── Señales ──────────────────────────────────────────────────────────
        can_enter    = position is None and day_count < p["max_day_tr"]
        long_signal  = can_enter and bull_sweep and trend_ok_long  and (not p["need_conf"] or bull_conf)
        short_signal = can_enter and bear_sweep and trend_ok_short and (not p["need_conf"] or bear_conf)

        if long_signal:
            sl = l  - pct(p["sl_buf_pct"], l)
            tp = c  + (c - sl) * p["rr"]
            size = (equity * p["qty_pct"] / 100) / c
            comm_entry = size * c * p["commission"] / 100
            equity    -= comm_entry
            position   = {"side":"long","entry":c,"sl":sl,"tp":tp,
                          "size":size,"entry_dt":row["dt"]}
            day_count += 1

        elif short_signal:
            sl = h  + pct(p["sl_buf_pct"], h)
            tp = c  - (sl - c) * p["rr"]
            size = (equity * p["qty_pct"] / 100) / c
            comm_entry = size * c * p["commission"] / 100
            equity    -= comm_entry
            position   = {"side":"short","entry":c,"sl":sl,"tp":tp,
                          "size":size,"entry_dt":row["dt"]}
            day_count += 1

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades, initial_cap):
    if trades.empty:
        print("Sin operaciones."); return

    eq   = trades["equity"]
    pnl  = trades["pnl"]
    wins = trades[trades["pnl"] > 0]
    loss = trades[trades["pnl"] <= 0]

    total_ret   = (eq.iloc[-1] - initial_cap) / initial_cap * 100
    win_rate    = len(wins) / len(trades) * 100
    avg_win     = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss    = loss["pnl_pct"].mean() if len(loss) else 0
    profit_fac  = wins["pnl"].sum() / abs(loss["pnl"].sum()) if len(loss) else np.inf
    expectancy  = pnl.mean()

    # Max Drawdown
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe (anual, asumiendo ~8760 horas/año)
    hourly_ret = trades["pnl"] / initial_cap
    sharpe = (hourly_ret.mean() / hourly_ret.std() * np.sqrt(len(trades))) if hourly_ret.std() > 0 else 0

    # Racha
    results   = (trades["pnl"] > 0).astype(int).tolist()
    max_win_streak = max_loss_streak = cur = 0
    for r in results:
        if r == 1:
            cur = max(0, cur) + 1
            max_win_streak = max(max_win_streak, cur)
        else:
            cur = min(0, cur) - 1
            max_loss_streak = max(max_loss_streak, abs(cur))

    by_year = trades.groupby(trades["entry_dt"].dt.year).agg(
        trades_n=("pnl","count"),
        net_pnl=("pnl","sum"),
        wr=("pnl", lambda x: (x > 0).mean() * 100)
    )

    print("\n" + "═"*58)
    print("  PO3 BACKTEST — BTC/USDT 1H  |  Ene 2017 – Dic 2021")
    print("═"*58)
    print(f"  Capital inicial      : ${initial_cap:>12,.2f}")
    print(f"  Capital final        : ${eq.iloc[-1]:>12,.2f}")
    print(f"  Retorno total        : {total_ret:>+11.2f}%")
    print(f"  Total operaciones    : {len(trades):>12,}")
    print(f"  Win Rate             : {win_rate:>11.1f}%")
    print(f"  Avg ganadora         : {avg_win:>+11.2f}%")
    print(f"  Avg perdedora        : {avg_loss:>+11.2f}%")
    print(f"  Profit Factor        : {profit_fac:>12.2f}")
    print(f"  Expectancy / trade   : ${expectancy:>11.2f}")
    print(f"  Max Drawdown         : {max_dd:>+11.2f}%")
    print(f"  Sharpe (aprox)       : {sharpe:>12.2f}")
    print(f"  Racha ganadora max   : {max_win_streak:>12}")
    print(f"  Racha perdedora max  : {max_loss_streak:>12}")
    print(f"  TP hits              : {len(trades[trades['result']=='TP']):>12,}")
    print(f"  SL hits              : {len(trades[trades['result']=='SL']):>12,}")
    print("─"*58)
    print("  RESULTADOS POR AÑO")
    print("─"*58)
    for yr, row in by_year.iterrows():
        print(f"  {yr}  |  {int(row.trades_n):>4} trades  |  "
              f"PnL ${row.net_pnl:>+10,.2f}  |  WR {row.wr:>5.1f}%")
    print("═"*58)

    return {
        "final_equity": eq.iloc[-1],
        "total_return_pct": total_ret,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_fac,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "by_year": by_year,
    }

def print_result(r, label, initial_cap, trades):
    eq   = trades["equity"]
    wins = trades[trades["pnl"] > 0]
    loss = trades[trades["pnl"] <= 0]
    print("\n" + "═"*62)
    print(f"  {label}")
    print("═"*62)
    print(f"  Capital inicial      : ${initial_cap:>12,.2f}")
    print(f"  Capital final        : ${r['final_equity']:>12,.2f}")
    print(f"  Retorno total        : {r['total_return_pct']:>+11.2f}%")
    print(f"  Total operaciones    : {r['n_trades']:>12,}")
    print(f"  Win Rate             : {r['win_rate']:>11.1f}%")
    print(f"  Avg ganadora         : {wins['pnl_pct'].mean() if len(wins) else 0:>+11.2f}%")
    print(f"  Avg perdedora        : {loss['pnl_pct'].mean() if len(loss) else 0:>+11.2f}%")
    print(f"  Profit Factor        : {r['profit_factor']:>12.2f}")
    print(f"  Expectancy / trade   : ${trades['pnl'].mean():>11.2f}")
    print(f"  Max Drawdown         : {r['max_dd']:>+11.2f}%")
    print(f"  Sharpe (aprox)       : {r['sharpe']:>12.2f}")
    print(f"  TP hits              : {len(trades[trades['result']=='TP']):>12,}")
    print(f"  SL hits              : {len(trades[trades['result']=='SL']):>12,}")
    print("─"*62)
    print("  RESULTADOS POR AÑO")
    print("─"*62)
    for yr, row in r["by_year"].iterrows():
        print(f"  {yr}  |  {int(row.trades_n):>4} trades  |  "
              f"PnL ${row.net_pnl:>+10,.2f}  |  WR {row.wr:>5.1f}%")
    print("═"*62)

# ─────────────────────────────────────────────────────────────────────────────
# GRID DE OPTIMIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

GRID = {
    "need_conf":    [False, True],
    "vol_mult":     [1.5, 2.0, 2.5],
    "trend_filter": [False, True],
    "max_day_tr":   [1, 2],
    "wick_min_pct": [0.20, 0.35],
}

def run_grid(df_raw, base):
    keys   = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    results = []
    print(f"\nOptimizando {len(combos)} combinaciones...", flush=True)
    for combo in combos:
        p = {**base}
        for k, v in zip(keys, combo):
            p[k] = v
        # re-calcular inst_vol y trend con nuevos params
        df = df_raw.copy()
        df["vol_avg"]  = df["volume"].rolling(p["vol_len"]).mean()
        df["inst_vol"] = df["volume"] >= df["vol_avg"] * p["vol_mult"]
        df["lower_wick"] = np.minimum(df.open, df.close) - df.low
        df["upper_wick"] = df.high - np.maximum(df.open, df.close)
        df["bull_wick_ok"] = df["lower_wick"] >= pct(p["wick_min_pct"], df.low)
        df["bear_wick_ok"] = df["upper_wick"] >= pct(p["wick_min_pct"], df.high)
        df["sma_trend"] = df["close"].rolling(p["trend_period"]).mean()
        df["uptrend"]   = df["close"] > df["sma_trend"]
        df["downtrend"] = df["close"] < df["sma_trend"]
        t = run_backtest(df, p)
        if t.empty:
            continue
        eq  = t["equity"]
        win = (t["pnl"] > 0).mean() * 100
        pf  = t[t["pnl"]>0]["pnl"].sum() / abs(t[t["pnl"]<=0]["pnl"].sum() + 1e-9)
        ret = (eq.iloc[-1] - p["initial_cap"]) / p["initial_cap"] * 100
        peak = eq.cummax(); dd = ((eq - peak)/peak*100).min()
        results.append({**dict(zip(keys, combo)),
                        "n_trades": len(t), "win_rate": win,
                        "profit_factor": pf, "return_pct": ret,
                        "max_dd": dd, "trades_df": t})
    return pd.DataFrame([{k:v for k,v in r.items() if k!="trades_df"} for r in results]), results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Generar datos base
    df_raw = fetch_ohlcv("BTC/USDT", "1h", "2017-01-01", "2022-01-01")
    df_base = add_indicators(df_raw, BASE_PARAMS)

    # 2. Baseline (parámetros originales)
    print("\n[1/3] Corriendo baseline (parámetros originales)...")
    t_base = run_backtest(df_base, BASE_PARAMS)
    r_base = metrics(t_base, BASE_PARAMS["initial_cap"])
    print_result(r_base, "BASELINE — Parámetros originales", BASE_PARAMS["initial_cap"], t_base)

    # 3. Optimizado v2: need_conf + vol_mult 2.0 + trend_filter
    p_v2 = {**BASE_PARAMS, "need_conf": True, "vol_mult": 2.0,
            "trend_filter": True, "max_day_tr": 1, "wick_min_pct": 0.35}
    df_v2 = df_raw.copy()
    df_v2 = add_indicators(df_v2, p_v2)
    df_v2["vol_avg"]  = df_v2["volume"].rolling(p_v2["vol_len"]).mean()
    df_v2["inst_vol"] = df_v2["volume"] >= df_v2["vol_avg"] * p_v2["vol_mult"]
    df_v2["bull_wick_ok"] = (np.minimum(df_v2.open,df_v2.close)-df_v2.low) >= pct(p_v2["wick_min_pct"],df_v2.low)
    df_v2["bear_wick_ok"] = (df_v2.high-np.maximum(df_v2.open,df_v2.close)) >= pct(p_v2["wick_min_pct"],df_v2.high)
    print("\n[2/3] Corriendo v2 (need_conf + vol_mult 2.0 + trend_filter + wick 0.35%)...")
    t_v2 = run_backtest(df_v2, p_v2)
    r_v2 = metrics(t_v2, p_v2["initial_cap"])
    print_result(r_v2, "V2 — need_conf + vol_mult 2.0 + trend_filter + wick 0.35%", p_v2["initial_cap"], t_v2)

    # 4. Grid search: mejor por profit_factor
    print("\n[3/3] Grid search completo...")
    grid_df, grid_full = run_grid(df_raw, BASE_PARAMS)
    grid_df_pos = grid_df[grid_df["return_pct"] > 0].sort_values("profit_factor", ascending=False)
    print(f"\n  Combinaciones totales    : {len(grid_df)}")
    print(f"  Combinaciones positivas  : {len(grid_df_pos)}")
    if not grid_df_pos.empty:
        best_idx = grid_df_pos.index[0]
        best_row = grid_df_pos.iloc[0]
        best_trades = grid_full[best_idx]["trades_df"]
        best_params = {**BASE_PARAMS}
        for k in GRID:
            best_params[k] = best_row[k]
        # métricas completas del mejor
        r_best = metrics(best_trades, BASE_PARAMS["initial_cap"])
        print_result(r_best, "MEJOR COMBINACIÓN (por Profit Factor)", BASE_PARAMS["initial_cap"], best_trades)
        print("\n  Parámetros ganadores:")
        for k in GRID:
            print(f"    {k:<16} = {best_row[k]}")
        print()
        best_trades.to_csv("/home/user/TV/backtest_trades_best.csv", index=False)

    t_v2.to_csv("/home/user/TV/backtest_trades_v2.csv", index=False)
    grid_df.to_csv("/home/user/TV/backtest_grid.csv", index=False)
    print("\nArchivos guardados: backtest_trades_v2.csv, backtest_trades_best.csv, backtest_grid.csv")

"""
PO3 Strategy Backtest — BTC/USDT 1H | Jan 2017 – Dec 2021
Vectorizado con numpy/pandas — grid de 48 combos en segundos.
"""

import pandas as pd
import numpy as np
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# DATOS SINTÉTICOS calibrados con waypoints históricos reales de BTC
# ─────────────────────────────────────────────────────────────────────────────

def build_data():
    np.random.seed(42)
    waypoints = [
        ("2017-01-01",  1_000, 0.025),
        ("2017-06-01",  2_800, 0.035),
        ("2017-12-17", 20_000, 0.060),
        ("2018-02-01",  8_500, 0.055),
        ("2018-12-15",  3_200, 0.045),
        ("2019-06-26", 13_800, 0.040),
        ("2019-12-01",  7_200, 0.030),
        ("2020-03-13",  4_900, 0.080),
        ("2020-08-01", 11_800, 0.030),
        ("2020-12-31", 29_000, 0.040),
        ("2021-04-14", 64_000, 0.055),
        ("2021-07-20", 29_800, 0.045),
        ("2021-11-10", 69_000, 0.050),
        ("2022-01-01", 46_000, 0.040),
    ]
    dates = pd.date_range("2017-01-01", "2022-01-01", freq="h", tz="UTC")[:-1]
    N = len(dates)
    closes = np.zeros(N)
    wdts = [pd.Timestamp(w[0], tz="UTC") for w in waypoints]
    wpx  = [w[1] for w in waypoints]
    wvl  = [w[2] for w in waypoints]

    for seg in range(len(wdts) - 1):
        t0, t1 = wdts[seg], wdts[seg+1]
        p0, p1 = wpx[seg], wpx[seg+1]
        v0     = wvl[seg]
        mask   = (dates >= t0) & (dates < t1)
        n      = mask.sum()
        if n == 0:
            continue
        h_vol  = v0 / np.sqrt(24)
        drift  = (np.log(p1/p0) - 0.5*h_vol**2*n) / n
        shocks = np.random.normal(drift, h_vol, n)
        jumps  = np.random.rand(n) < 0.02
        shocks[jumps] *= 2.5
        closes[mask] = p0 * np.exp(np.cumsum(shocks))

    closes = pd.Series(closes).replace(0, np.nan).ffill().bfill().values

    opens  = np.roll(closes, 1); opens[0] = closes[0]
    spread = closes * 0.005 * np.abs(np.random.normal(1, 0.5, N))
    highs  = np.maximum(opens, closes) + spread * np.abs(np.random.normal(0.6, 0.3, N))
    lows   = np.minimum(opens, closes) - spread * np.abs(np.random.normal(0.6, 0.3, N))
    highs  = np.maximum(highs, np.maximum(opens, closes))
    lows   = np.minimum(lows,  np.minimum(opens, closes))

    hour_of_day = dates.hour.values
    vol_h = np.where(hour_of_day < 7, 0.6, np.where(hour_of_day < 16, 1.4, 1.0))
    vols  = 15_000 * vol_h * np.abs(np.random.lognormal(0, 0.5, N))

    df = pd.DataFrame({"dt": dates, "open": opens, "high": highs,
                       "low": lows, "close": closes, "volume": vols})
    print(f"Velas: {N:,}  |  ${closes[0]:,.0f} → ${closes[-1]:,.0f}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# INDICADORES (vectorizados)
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df, p):
    o = df.open.values; h = df.high.values
    l = df.low.values;  c = df.close.values; v = df.volume.values
    N = len(df)

    # ── PDH / PDL ────────────────────────────────────────────────────────────
    date_arr = df["dt"].dt.floor("D")
    df2 = df.copy(); df2["date"] = date_arr
    day_max = df2.groupby("date")["high"].transform("max").values
    day_min = df2.groupby("date")["low"].transform("min").values
    # shift by 1 day: compare date to previous date's max/min
    pdh = np.full(N, np.nan); pdl = np.full(N, np.nan)
    dates_u = pd.DatetimeIndex(sorted(date_arr.unique()))
    day_hi_map = df2.groupby("date")["high"].max()
    day_lo_map = df2.groupby("date")["low"].min()
    for i, d in enumerate(dates_u):
        if i == 0: continue
        prev_d = dates_u[i-1]
        mask = (date_arr == d).values
        pdh[mask] = day_hi_map[prev_d]
        pdl[mask] = day_lo_map[prev_d]

    # ── PWH / PWL ────────────────────────────────────────────────────────────
    week_arr = df["dt"].dt.to_period("W").apply(lambda x: x.start_time)
    df2["week"] = week_arr
    weeks_u = pd.DatetimeIndex(sorted(week_arr.unique()))
    wk_hi_map = df2.groupby("week")["high"].max()
    wk_lo_map = df2.groupby("week")["low"].min()
    pwh = np.full(N, np.nan); pwl = np.full(N, np.nan)
    for i, w in enumerate(weeks_u):
        if i == 0: continue
        prev_w = weeks_u[i-1]
        mask = (week_arr == w).values
        pwh[mask] = wk_hi_map[prev_w]
        pwl[mask] = wk_lo_map[prev_w]

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_avg  = pd.Series(v).rolling(p["vol_len"]).mean().values
    inst_vol = v >= vol_avg * p["vol_mult"]

    # ── Wicks ────────────────────────────────────────────────────────────────
    lower_wick   = np.minimum(o, c) - l
    upper_wick   = h - np.maximum(o, c)
    bull_wick_ok = lower_wick >= p["wick_min_pct"] / 100 * l
    bear_wick_ok = upper_wick >= p["wick_min_pct"] / 100 * h

    # ── EQH / EQL (vectorized rolling window) ────────────────────────────────
    eq_look = p["eq_look"]; eq_tol = p["eq_tol"] / 100
    eqh_hit = np.zeros(N, bool); eql_hit = np.zeros(N, bool)
    eqh_lvl = np.full(N, np.nan); eql_lvl = np.full(N, np.nan)
    for k in range(1, eq_look + 1):
        ph = np.roll(h, k); ph[:k] = np.nan
        pl = np.roll(l, k); pl[:k] = np.nan
        m_h = np.abs(h - ph) <= eq_tol * h
        m_l = np.abs(l - pl) <= eq_tol * l
        eqh_hit |= m_h; eql_hit |= m_l
        eqh_lvl = np.where(m_h, np.fmax(h, ph), eqh_lvl)
        eql_lvl = np.where(m_l, np.fmin(l, pl), eql_lvl)

    # ── Sweeps ───────────────────────────────────────────────────────────────
    bull_sweep = (
        (((l < pdh) & (c > pdh)) |
         ((l < pwl) & (c > pwl)) |
         (eql_hit & (l < eql_lvl) & (c > eql_lvl))) &
        bull_wick_ok & inst_vol
    )
    bear_sweep = (
        (((h > pdh) & (c < pdh)) |
         ((h > pwh) & (c < pwh)) |
         (eqh_hit & (h > eqh_lvl) & (c < eqh_lvl))) &
        bear_wick_ok & inst_vol
    )
    # fix NaN PDH/PDL guard
    has_pdl = ~np.isnan(pdl); has_pdh = ~np.isnan(pdh)
    has_pwl = ~np.isnan(pwl); has_pwh = ~np.isnan(pwh)
    bull_sweep_pdl = has_pdl & (l < pdl) & (c > pdl) & bull_wick_ok & inst_vol
    bull_sweep_pwl = has_pwl & (l < pwl) & (c > pwl) & bull_wick_ok & inst_vol
    bull_sweep_eql = eql_hit & (l < eql_lvl) & (c > eql_lvl) & bull_wick_ok & inst_vol
    bear_sweep_pdh = has_pdh & (h > pdh) & (c < pdh) & bear_wick_ok & inst_vol
    bear_sweep_pwh = has_pwh & (h > pwh) & (c < pwh) & bear_wick_ok & inst_vol
    bear_sweep_eqh = eqh_hit & (h > eqh_lvl) & (c < eqh_lvl) & bear_wick_ok & inst_vol
    bull_sweep = bull_sweep_pdl | bull_sweep_pwl | bull_sweep_eql
    bear_sweep = bear_sweep_pdh | bear_sweep_pwh | bear_sweep_eqh

    # ── FVG (vectorized) ─────────────────────────────────────────────────────
    c1 = np.roll(c, 1); o1 = np.roll(o, 1); h2 = np.roll(h, 2); l2 = np.roll(l, 2)
    fvg_min = p["fvg_min_pct"] / 100 * c
    bull_fvg = (c1 > o1) & (l > h2) & ((l - h2) >= fvg_min)
    bear_fvg = (c1 < o1) & (h < l2) & ((l2 - h) >= fvg_min)
    # active FVG: propagate forward until invalidated
    bfvg_lo = np.where(bull_fvg, h2, np.nan)
    bfvg_hi = np.where(bull_fvg, l,  np.nan)
    sfvg_lo = np.where(bear_fvg, h,  np.nan)
    sfvg_hi = np.where(bear_fvg, l2, np.nan)
    # forward-fill then invalidate
    bfvg_lo_f = pd.Series(bfvg_lo).ffill().values
    bfvg_hi_f = pd.Series(bfvg_hi).ffill().values
    sfvg_lo_f = pd.Series(sfvg_lo).ffill().values
    sfvg_hi_f = pd.Series(sfvg_hi).ffill().values
    bull_fvg_in = (c >= bfvg_lo_f) & (c <= bfvg_hi_f) & ~np.isnan(bfvg_lo_f)
    bear_fvg_in = (c >= sfvg_lo_f) & (c <= sfvg_hi_f) & ~np.isnan(sfvg_lo_f)

    # ── OB (vectorized) ──────────────────────────────────────────────────────
    ob_look = p["ob_look"]
    c_prev  = np.roll(c, 1); c_prev[0] = c[0]
    bull_imp = (c > o) & (c > c_prev * 1.002)
    bear_imp = (c < o) & (c < c_prev * 0.998)
    ob_c = np.roll(c, ob_look); ob_o = np.roll(o, ob_look)
    bob_lo = np.where(bull_imp & (ob_c < ob_o), ob_c, np.nan)
    bob_hi = np.where(bull_imp & (ob_c < ob_o), ob_o, np.nan)
    soo_lo = np.where(bear_imp & (ob_c > ob_o), ob_o, np.nan)
    soo_hi = np.where(bear_imp & (ob_c > ob_o), ob_c, np.nan)
    bob_lo_f = pd.Series(bob_lo).ffill().values
    bob_hi_f = pd.Series(bob_hi).ffill().values
    soo_lo_f = pd.Series(soo_lo).ffill().values
    soo_hi_f = pd.Series(soo_hi).ffill().values
    bull_ob_in = (c >= bob_lo_f) & (c <= bob_hi_f) & ~np.isnan(bob_lo_f)
    bear_ob_in = (c >= soo_lo_f) & (c <= soo_hi_f) & ~np.isnan(soo_lo_f)

    bull_conf = bull_ob_in | bull_fvg_in
    bear_conf = bear_ob_in | bear_fvg_in

    # ── Trend filter ─────────────────────────────────────────────────────────
    sma = pd.Series(c).rolling(p["trend_period"]).mean().values
    uptrend   = c > sma
    downtrend = c < sma

    # ── Final signals ─────────────────────────────────────────────────────────
    conf_ok_long  = bull_conf if p["need_conf"] else np.ones(N, bool)
    conf_ok_short = bear_conf if p["need_conf"] else np.ones(N, bool)
    tr_ok_long    = uptrend   if p["trend_filter"] else np.ones(N, bool)
    tr_ok_short   = downtrend if p["trend_filter"] else np.ones(N, bool)

    long_sig  = bull_sweep & conf_ok_long  & tr_ok_long
    short_sig = bear_sweep & conf_ok_short & tr_ok_short

    return dict(long_sig=long_sig, short_sig=short_sig,
                l=l, h=h, c=c, o=o,
                pdl=pdl, pdh=pdh)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN DE TRADES (loop mínimo sobre señales solamente)
# ─────────────────────────────────────────────────────────────────────────────

def simulate(df, sig, p):
    long_sig  = sig["long_sig"]
    short_sig = sig["short_sig"]
    h_arr = sig["h"]; l_arr = sig["l"]; c_arr = sig["c"]
    dt_arr = df["dt"].values
    dates  = pd.to_datetime(dt_arr).date

    equity    = p["initial_cap"]
    trades    = []
    pos       = None
    day_cnt   = 0
    prev_date = None
    rr = p["rr"]; sl_buf = p["sl_buf_pct"] / 100
    comm = p["commission"] / 100
    max_dt = p["max_day_tr"]
    N = len(df)

    # Only iterate over bars with a signal or active position
    signal_bars = set(np.where(long_sig | short_sig)[0].tolist())

    for i in range(1, N):
        d = dates[i]
        if d != prev_date:
            day_cnt  = 0
            prev_date = d

        h = h_arr[i]; l = l_arr[i]; c = c_arr[i]

        # Close position?
        if pos is not None:
            if pos["side"] == "long":
                hit_sl = l <= pos["sl"]
                hit_tp = h >= pos["tp"]
                exit_px = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else None)
            else:
                hit_sl = h >= pos["sl"]
                hit_tp = l <= pos["tp"]
                exit_px = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else None)

            if exit_px is not None:
                entry_px = pos["entry"]
                size     = pos["size"]
                pnl_pct  = ((exit_px - entry_px) / entry_px
                            if pos["side"] == "long"
                            else (entry_px - exit_px) / entry_px)
                net_pnl  = size * entry_px * pnl_pct - size * exit_px * comm
                equity  += net_pnl
                trades.append({
                    "entry_dt": pos["entry_dt"], "exit_dt": dt_arr[i],
                    "side": pos["side"], "entry_px": entry_px, "exit_px": exit_px,
                    "pnl": net_pnl, "pnl_pct": pnl_pct * 100,
                    "result": "TP" if hit_tp else "SL", "equity": equity,
                })
                pos = None

        # Open position?
        if pos is None and day_cnt < max_dt and i in signal_bars:
            if long_sig[i]:
                sl = l - sl_buf * l
                tp = c + (c - sl) * rr
                size = equity / c
                equity -= size * c * comm
                pos = {"side":"long","entry":c,"sl":sl,"tp":tp,
                       "size":size,"entry_dt":dt_arr[i]}
                day_cnt += 1
            elif short_sig[i]:
                sl = h + sl_buf * h
                tp = c - (sl - c) * rr
                size = equity / c
                equity -= size * c * comm
                pos = {"side":"short","entry":c,"sl":sl,"tp":tp,
                       "size":size,"entry_dt":dt_arr[i]}
                day_cnt += 1

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(trades, cap):
    if trades.empty or len(trades) < 3:
        return None
    eq   = trades["equity"]
    pnl  = trades["pnl"]
    wins = trades[trades["pnl"] > 0]
    loss = trades[trades["pnl"] <= 0]
    wr   = len(wins) / len(trades) * 100
    pf   = wins["pnl"].sum() / abs(loss["pnl"].sum()) if len(loss) and loss["pnl"].sum() != 0 else 0
    ret  = (eq.iloc[-1] - cap) / cap * 100
    peak = eq.cummax(); dd = ((eq - peak) / peak * 100).min()
    by_year = trades.groupby(pd.to_datetime(trades["entry_dt"]).dt.year).agg(
        n=("pnl","count"), pnl=("pnl","sum"),
        wr=("pnl", lambda x: (x>0).mean()*100))
    return {"ret": ret, "n": len(trades), "wr": wr, "pf": pf,
            "dd": dd, "final": eq.iloc[-1], "by_year": by_year,
            "avg_win": wins["pnl_pct"].mean() if len(wins) else 0,
            "avg_loss": loss["pnl_pct"].mean() if len(loss) else 0,
            "exp": pnl.mean(),
            "tp": (trades["result"]=="TP").sum(),
            "sl": (trades["result"]=="SL").sum()}

def print_full(m, label, cap):
    if m is None:
        print(f"\n{label}: sin suficientes trades"); return
    print("\n" + "═"*62)
    print(f"  {label}")
    print("═"*62)
    print(f"  Capital inicial      : ${cap:>12,.2f}")
    print(f"  Capital final        : ${m['final']:>12,.2f}")
    print(f"  Retorno total        : {m['ret']:>+11.2f}%")
    print(f"  Total operaciones    : {m['n']:>12,}")
    print(f"  Win Rate             : {m['wr']:>11.1f}%")
    print(f"  Avg ganadora         : {m['avg_win']:>+11.2f}%")
    print(f"  Avg perdedora        : {m['avg_loss']:>+11.2f}%")
    print(f"  Profit Factor        : {m['pf']:>12.2f}")
    print(f"  Expectancy / trade   : ${m['exp']:>11.2f}")
    print(f"  Max Drawdown         : {m['dd']:>+11.2f}%")
    print(f"  TP hits              : {m['tp']:>12,}")
    print(f"  SL hits              : {m['sl']:>12,}")
    print("─"*62)
    print("  POR AÑO")
    print("─"*62)
    for yr, row in m["by_year"].iterrows():
        print(f"  {yr}  | {int(row['n']):>4} trades | "
              f"PnL ${row['pnl']:>+10,.2f} | WR {row['wr']:>5.1f}%")
    print("═"*62)

# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

GRID = {
    "need_conf":    [False, True],
    "vol_mult":     [1.5, 2.0, 2.5],
    "trend_filter": [False, True],
    "max_day_tr":   [1, 2],
    "wick_min_pct": [0.20, 0.35],
}

BASE = dict(
    use_pdhl=True, use_pwhl=True, use_eqhl=True,
    eq_look=10, eq_tol=0.15, vol_len=20, ob_look=3,
    fvg_min_pct=0.05, rr=2.5, sl_buf_pct=0.10,
    trend_period=200, commission=0.05,
    initial_cap=10_000, qty_pct=100,
    # defaults overridden by grid:
    need_conf=False, vol_mult=1.5, trend_filter=False,
    max_day_tr=2, wick_min_pct=0.20,
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    print("Generando datos sintéticos BTC/USDT 1H (2017-2021)...")
    df = build_data()

    keys   = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    print(f"\nGrid search: {len(combos)} combinaciones\n")

    results = []
    t0 = time.time()
    for idx, combo in enumerate(combos):
        p = {**BASE, **dict(zip(keys, combo))}
        sig = build_signals(df, p)
        trades = simulate(df, sig, p)
        m = calc_metrics(trades, p["initial_cap"])
        if m:
            results.append({**dict(zip(keys, combo)),
                            "ret": m["ret"], "n": m["n"],
                            "wr": m["wr"], "pf": m["pf"],
                            "dd": m["dd"], "_trades": trades, "_m": m})

    elapsed = time.time() - t0
    print(f"Grid completado en {elapsed:.1f}s\n")

    grid_df = pd.DataFrame([{k:v for k,v in r.items()
                              if not k.startswith("_")} for r in results])
    grid_df.to_csv("/home/user/TV/backtest_grid.csv", index=False)

    # ── Tabla resumen ──────────────────────────────────────────────────────
    print("─"*80)
    print(f"  {'need_conf':<10} {'vol_mult':<10} {'trend':<7} {'max_dt':<8} "
          f"{'wick%':<7} {'trades':<8} {'WR%':<7} {'PF':<6} {'Ret%':<9} {'DD%'}")
    print("─"*80)
    for r in sorted(results, key=lambda x: x["pf"], reverse=True):
        mark = " ◄ MEJOR" if r == sorted(results, key=lambda x: x["pf"], reverse=True)[0] else ""
        print(f"  {str(r['need_conf']):<10} {r['vol_mult']:<10} "
              f"{str(r['trend_filter']):<7} {r['max_day_tr']:<8} "
              f"{r['wick_min_pct']:<7} {r['n']:<8} "
              f"{r['wr']:<7.1f} {r['pf']:<6.2f} "
              f"{r['ret']:<9.1f} {r['dd']:.1f}{mark}")
    print("─"*80)

    # ── Detalle del mejor ──────────────────────────────────────────────────
    best = sorted(results, key=lambda x: x["pf"], reverse=True)[0]
    print_full(best["_m"], "MEJOR COMBINACIÓN (mayor Profit Factor)", BASE["initial_cap"])
    print("\n  Parámetros ganadores:")
    for k in keys:
        print(f"    {k:<16} = {best[k]}")

    # ── Baseline para comparación ──────────────────────────────────────────
    base_r = next((r for r in results
                   if not r["need_conf"] and r["vol_mult"]==1.5
                   and not r["trend_filter"] and r["max_day_tr"]==2
                   and r["wick_min_pct"]==0.20), None)
    if base_r:
        print_full(base_r["_m"], "BASELINE (parámetros originales)", BASE["initial_cap"])

    best["_trades"].to_csv("/home/user/TV/backtest_trades_best.csv", index=False)
    print("\nGuardado: backtest_trades_best.csv, backtest_grid.csv")

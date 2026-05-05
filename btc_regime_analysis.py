"""
BTC/USDC 1H — Estudio por Regímenes de Mercado
Golden Pocket Fibonacci | HotMarketGuy
Análisis 2021-2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────
LOOKBACK_SWING    = 5
MIN_IMPULSE_PCT   = 0.0075
GOLDEN_POCKET_LOW = 0.618
GOLDEN_POCKET_HI  = 0.650
SL_NIVEL_PCT      = 0.008
TP1_RR            = 1.0
TP2_RR            = 2.5
TP1_FRACTION      = 0.50
CAPITAL_INICIAL   = 10_000
RISK_PCT          = 0.01
MAX_HOLD_CANDLES  = 48
EMA_FAST          = 21
EMA_SLOW          = 55
TREND_MA          = 200
MIN_CANDLES_ENTRE  = 6

# ─── Colores por régimen ─────────────────
REGIME_COLORS = {
    "BULL_FUERTE":   "#00c853",   # verde brillante
    "BULL_MODERADO": "#69f0ae",   # verde suave
    "RECUPERACION":  "#ffeb3b",   # amarillo
    "RANGO":         "#ff9800",   # naranja
    "BEAR_MODERADO": "#ff5722",   # naranja-rojo
    "BEAR_FUERTE":   "#d32f2f",   # rojo oscuro
}

# ─────────────────────────────────────────
# 1. DATOS HISTÓRICOS BTC (anclados a precios reales)
# ─────────────────────────────────────────
BTC_MONTHLY_ANCHORS = [
    ("2021-01-01", 29000,  42000, 27000,  38500, 1.0),
    ("2021-02-01", 38500,  58500, 36000,  45000, 1.3),
    ("2021-03-01", 45000,  61000, 47000,  58800, 1.1),
    ("2021-04-01", 58800,  65000, 47000,  57500, 1.2),
    ("2021-05-01", 57500,  59500, 30000,  37000, 1.8),
    ("2021-06-01", 37000,  41000, 28800,  35000, 1.4),
    ("2021-07-01", 35000,  42500, 29700,  41500, 1.2),
    ("2021-08-01", 41500,  50200, 39600,  47100, 1.1),
    ("2021-09-01", 47100,  52950, 39600,  43800, 1.2),
    ("2021-10-01", 43800,  67000, 43400,  60900, 1.3),
    ("2021-11-01", 60900,  69000, 53600,  57200, 1.5),
    ("2021-12-01", 57200,  59050, 42000,  47700, 1.3),
    ("2022-01-01", 47700,  48000, 32900,  38500, 1.4),
    ("2022-02-01", 38500,  45800, 36800,  43200, 1.1),
    ("2022-03-01", 43200,  48100, 37600,  45500, 1.0),
    ("2022-04-01", 45500,  47400, 37600,  38400, 1.1),
    ("2022-05-01", 38400,  40000, 25400,  31500, 1.6),
    ("2022-06-01", 31500,  31800, 17600,  19100, 2.0),
    ("2022-07-01", 19100,  25200, 18800,  23300, 1.5),
    ("2022-08-01", 23300,  25200, 19600,  20050, 1.3),
    ("2022-09-01", 20050,  22700, 18200,  19500, 1.2),
    ("2022-10-01", 19500,  21500, 18900,  20500, 1.1),
    ("2022-11-01", 20500,  21500, 15476,  16550, 2.5),
    ("2022-12-01", 16550,  18400, 16200,  16600, 1.4),
    ("2023-01-01", 16600,  23800, 16400,  23100, 1.3),
    ("2023-02-01", 23100,  25200, 21350,  23200, 1.0),
    ("2023-03-01", 23200,  29200, 19600,  28400, 1.2),
    ("2023-04-01", 28400,  31000, 26750,  29200, 1.0),
    ("2023-05-01", 29200,  29800, 25800,  27200, 1.1),
    ("2023-06-01", 27200,  31800, 24800,  30500, 1.2),
    ("2023-07-01", 30500,  32000, 28500,  29200, 1.0),
    ("2023-08-01", 29200,  30200, 24750,  25900, 1.1),
    ("2023-09-01", 25900,  27900, 24900,  26900, 0.9),
    ("2023-10-01", 26900,  35700, 26400,  34600, 1.3),
    ("2023-11-01", 34600,  38000, 33600,  37900, 1.2),
    ("2023-12-01", 37900,  44800, 37200,  42300, 1.3),
    ("2024-01-01", 42300,  49000, 38500,  42700, 1.2),
    ("2024-02-01", 42700,  57500, 41550,  62400, 1.5),
    ("2024-03-01", 62400,  73780, 59000,  71400, 1.6),
    ("2024-04-01", 71400,  72800, 56500,  60700, 1.4),
    ("2024-05-01", 60700,  73750, 56600,  67500, 1.2),
    ("2024-06-01", 67500,  71900, 58400,  62700, 1.1),
    ("2024-07-01", 62700,  70000, 53500,  66200, 1.2),
    ("2024-08-01", 66200,  65300, 49000,  59100, 1.4),
    ("2024-09-01", 59100,  66500, 52700,  63300, 1.1),
    ("2024-10-01", 63300,  73600, 60200,  72400, 1.2),
    ("2024-11-01", 72400,  99700, 67500,  97200, 1.7),
    ("2024-12-01", 97200, 108500, 91800,  93400, 1.5),
    ("2025-01-01", 93400, 109000, 89000,  95800, 1.4),
    ("2025-02-01", 95800, 100200, 78400,  84500, 1.3),
    ("2025-03-01", 84500,  92000, 76800,  82300, 1.2),
    ("2025-04-01", 82300,  97500, 74600,  94100, 1.3),
    ("2025-05-01", 94100, 107000, 93000, 103500, 1.2),
    ("2025-06-01",103500, 116000,100000, 110000, 1.2),
    ("2025-07-01",110000, 125000,107000, 121000, 1.3),
    ("2025-08-01",121000, 127000,110000, 114000, 1.3),
    ("2025-09-01",114000, 126000,107000, 118000, 1.2),
    ("2025-10-01",118000, 126000,105000, 108000, 1.3),
    ("2025-11-01",108000, 118000, 83500,  90000, 1.5),
    ("2025-12-01", 90000,  96000, 83000,  87000, 1.3),
    ("2026-01-01", 87000,  91000, 74700,  79500, 1.3),
    ("2026-02-01", 79500,  85000, 59500,  71000, 1.5),
    ("2026-03-01", 71000,  80000, 66000,  72600, 1.3),
    ("2026-04-01", 72600,  80200, 67044,  75000, 1.4),
    ("2026-05-01", 75000,  81500, 74000,  80696, 1.2),
]

# ─────────────────────────────────────────
# 2. GENERAR OHLCV 1H
# ─────────────────────────────────────────
def generar_ohlcv_1h(anchors):
    from datetime import timedelta
    rows = []
    for i in range(len(anchors) - 1):
        ts0, o0, h0, l0, c0, vol0 = anchors[i]
        ts1, o1, h1, l1, c1, vol1 = anchors[i + 1]
        dt0   = pd.Timestamp(ts0)
        dt1   = pd.Timestamp(ts1)
        horas = int((dt1 - dt0).total_seconds() / 3600)
        if horas <= 0:
            continue
        sigma_hora = 0.008 * vol0
        mu_hora    = np.log(c1 / c0) / horas
        precio = c0
        for h in range(horas):
            t      = dt0 + pd.Timedelta(hours=h)
            ret    = mu_hora + sigma_hora * np.random.randn()
            precio = precio * np.exp(ret)
            if h == horas - 1:
                precio = c1 * (1 + np.random.uniform(-0.003, 0.003))
            rango_h = precio * sigma_hora * np.abs(np.random.randn() + 0.5)
            wick_up = precio * sigma_hora * np.abs(np.random.randn() * 0.3)
            wick_dn = precio * sigma_hora * np.abs(np.random.randn() * 0.3)
            open_c  = precio * (1 + np.random.uniform(-sigma_hora * 0.3, sigma_hora * 0.3))
            close_c = precio
            high_c  = max(open_c, close_c) + wick_up + rango_h * 0.3
            low_c   = min(open_c, close_c) - wick_dn - rango_h * 0.3
            vol_c   = vol0 * np.abs(np.random.randn() + 1) * 1000
            rows.append({"timestamp": t, "open": round(open_c, 2),
                         "high": round(high_c, 2), "low": round(low_c, 2),
                         "close": round(close_c, 2), "volume": round(vol_c, 2)})
    df = pd.DataFrame(rows).set_index("timestamp")
    return df[df["low"] > 0]

# ─────────────────────────────────────────
# 3. CLASIFICACIÓN DE RÉGIMEN (vela a vela)
# ─────────────────────────────────────────
def clasificar_regimenes(df):
    """
    6 regímenes basados en:
    - Posición del precio vs EMA200
    - Alineación EMA21 / EMA55 / EMA200
    - Pendiente de EMA200 (momentum macro)
    - ATR normalizado (volatilidad = rango vs tendencia)
    """
    cl  = df["close"]
    hi  = df["high"]
    lo  = df["low"]

    ema21  = cl.ewm(span=21,  adjust=False).mean()
    ema55  = cl.ewm(span=55,  adjust=False).mean()
    ema200 = cl.ewm(span=200, adjust=False).mean()

    # Pendiente EMA200 (sobre 24h = 1 día)
    slope200 = ema200.diff(24) / ema200.shift(24)  # % cambio en 24h

    # ATR 14 períodos (volatilidad)
    tr   = pd.concat([hi - lo,
                      (hi - cl.shift()).abs(),
                      (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    atr_norm = atr / cl  # ATR normalizado (%)

    regimenes = []
    for i in range(len(df)):
        p    = cl.iloc[i]
        e21  = ema21.iloc[i]
        e55  = ema55.iloc[i]
        e200 = ema200.iloc[i]
        sl   = slope200.iloc[i] if not pd.isna(slope200.iloc[i]) else 0
        atr_n= atr_norm.iloc[i] if not pd.isna(atr_norm.iloc[i]) else 0.01

        above200 = p > e200
        e21_above_55  = e21 > e55
        e55_above_200 = e55 > e200

        if pd.isna(e200):
            regimenes.append("RANGO")
            continue

        if above200 and e21_above_55 and e55_above_200 and sl > 0.001:
            regimenes.append("BULL_FUERTE")
        elif above200 and e21_above_55 and e55_above_200:
            regimenes.append("BULL_MODERADO")
        elif above200 and (e21_above_55 or e55_above_200):
            regimenes.append("RECUPERACION")
        elif not above200 and not e21_above_55 and not e55_above_200 and sl < -0.001:
            regimenes.append("BEAR_FUERTE")
        elif not above200 and not e21_above_55:
            regimenes.append("BEAR_MODERADO")
        else:
            regimenes.append("RANGO")

    df["regime"] = regimenes
    df["ema21"]  = ema21
    df["ema55"]  = ema55
    df["ema200"] = ema200
    return df

# ─────────────────────────────────────────
# 4. DETECCIÓN DE SWINGS
# ─────────────────────────────────────────
def detectar_swings(df, n=LOOKBACK_SWING):
    highs, lows = [], []
    hi_arr = df["high"].values
    lo_arr = df["low"].values
    for i in range(n, len(df) - n):
        if hi_arr[i] == hi_arr[i-n:i+n+1].max():
            highs.append(i)
        if lo_arr[i] == lo_arr[i-n:i+n+1].min():
            lows.append(i)
    return highs, lows

# ─────────────────────────────────────────
# 5. IMPULSOS VÁLIDOS
# ─────────────────────────────────────────
def identificar_impulsos(df, swing_lows, swing_highs):
    impulsos = []
    hi_arr = df["high"].values
    lo_arr = df["low"].values
    idx    = df.index
    reg    = df["regime"].values
    for li in swing_lows:
        low_price = lo_arr[li]
        candidatos_hi = [hi for hi in swing_highs if li + 4 < hi < li + 400]
        if not candidatos_hi:
            continue
        next_lows = [sl for sl in swing_lows if sl > li]
        cut = next_lows[0] if next_lows else len(df)
        candidatos_hi = [hi for hi in candidatos_hi if hi < cut]
        if not candidatos_hi:
            continue
        hi = max(candidatos_hi, key=lambda x: hi_arr[x])
        high_price = hi_arr[hi]
        rango_pct  = (high_price - low_price) / low_price
        if rango_pct >= MIN_IMPULSE_PCT:
            impulsos.append({
                "low_idx": li,  "high_idx": hi,
                "low_price": low_price, "high_price": high_price,
                "rango_pct": rango_pct,
                "low_time":  idx[li],  "high_time": idx[hi],
                "regime_at_high": reg[hi],
            })
    return impulsos

# ─────────────────────────────────────────
# 6. FIBONACCI NIVELES
# ─────────────────────────────────────────
def fibs(imp):
    r  = imp["high_price"] - imp["low_price"]
    h  = imp["high_price"]
    gp = h - r * GOLDEN_POCKET_LOW
    sl = gp * (1 - SL_NIVEL_PCT)
    rk = gp - sl
    return {"gp_lo": gp, "gp_hi": h - r * GOLDEN_POCKET_HI,
            "sl": sl, "tp1": gp + rk * TP1_RR, "tp2": gp + rk * TP2_RR}

# ─────────────────────────────────────────
# 7. MOTOR DE BACKTEST POR RÉGIMEN
# ─────────────────────────────────────────
def backtest_por_regimen(df, impulsos, regimenes_permitidos=None):
    """
    regimenes_permitidos: lista de regímenes donde se opera.
    None = operar en todos.
    """
    capital      = CAPITAL_INICIAL
    equity_curve = [capital]
    trades       = []
    lo  = df["low"].values
    hi  = df["high"].values
    cl  = df["close"].values
    idx = df.index
    reg = df["regime"].values
    ema21_v = df["ema21"].values
    ema55_v = df["ema55"].values

    ultimo_salida = -MIN_CANDLES_ENTRE

    for imp in impulsos:
        f = fibs(imp)
        if f["gp_lo"] - f["sl"] <= 0:
            continue

        # Filtro de régimen en el momento del impulso
        r_high = reg[imp["high_idx"]]
        if regimenes_permitidos and r_high not in regimenes_permitidos:
            continue

        inicio = max(imp["high_idx"] + 1, ultimo_salida + MIN_CANDLES_ENTRE)
        fin    = min(imp["high_idx"] + MAX_HOLD_CANDLES * 3, len(df))

        entrada_idx = None
        for i in range(inicio, fin):
            # Filtro de régimen en el momento de la entrada
            if regimenes_permitidos and reg[i] not in regimenes_permitidos:
                continue
            if lo[i] <= f["gp_lo"] and cl[i] > f["gp_hi"]:
                if cl[i] > (lo[i] + hi[i]) / 2:
                    entrada_idx = i
                    break

        if entrada_idx is None:
            continue

        entrada_exec   = entrada_idx + 1
        if entrada_exec >= len(df):
            continue

        precio_entrada = cl[entrada_idx]
        riesgo_usd     = capital * RISK_PCT
        riesgo_unit    = precio_entrada * SL_NIVEL_PCT
        tam_pos_usd    = min(riesgo_usd / SL_NIVEL_PCT, capital * 0.5)
        btc_size       = tam_pos_usd / precio_entrada

        resultado     = "TIMEOUT"
        precio_salida = None
        salida_idx    = None
        tp1_hit       = False
        btc_rem       = btc_size

        for j in range(entrada_exec,
                       min(entrada_exec + MAX_HOLD_CANDLES + 1, len(df))):
            if lo[j] <= f["sl"]:
                precio_salida = f["sl"]
                resultado     = "SL"
                salida_idx    = j
                break
            if not tp1_hit and hi[j] >= f["tp1"]:
                tp1_hit  = True
                btc_rem *= (1 - TP1_FRACTION)
            if hi[j] >= f["tp2"]:
                precio_salida = f["tp2"]
                resultado     = "TP2"
                salida_idx    = j
                break

        if salida_idx is None:
            salida_idx    = min(entrada_exec + MAX_HOLD_CANDLES, len(df) - 1)
            precio_salida = cl[salida_idx]

        if tp1_hit:
            pnl = (btc_size * TP1_FRACTION * (f["tp1"] - precio_entrada) +
                   btc_rem  * (precio_salida - precio_entrada))
        else:
            pnl = btc_size * (precio_salida - precio_entrada)

        capital += pnl
        capital  = max(capital, 1)
        dur_h    = (idx[salida_idx] - idx[entrada_exec]).total_seconds() / 3600
        ultimo_salida = salida_idx

        trades.append({
            "entrada_time":  idx[entrada_exec],
            "salida_time":   idx[salida_idx],
            "año":           idx[entrada_exec].year,
            "mes":           idx[entrada_exec].strftime("%Y-%m"),
            "regime":        r_high,
            "precio_entrada":round(precio_entrada, 2),
            "precio_salida": round(precio_salida, 2),
            "resultado":     resultado,
            "pnl_usd":       round(pnl, 2),
            "pnl_pct":       round(pnl / tam_pos_usd * 100, 2) if tam_pos_usd else 0,
            "capital":       round(capital, 2),
            "duracion_h":    round(dur_h, 1),
            "rango_imp":     round(imp["rango_pct"] * 100, 2),
            "tp1_hit":       tp1_hit,
        })
        equity_curve.append(capital)

    return pd.DataFrame(trades), equity_curve

# ─────────────────────────────────────────
# 8. ESTADÍSTICAS POR RÉGIMEN
# ─────────────────────────────────────────
def stats_regimen(trades_df, eq_curve):
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df["pnl_usd"] > 0]
    loss = trades_df[trades_df["pnl_usd"] <= 0]
    eq   = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    rets = trades_df["pnl_usd"] / CAPITAL_INICIAL
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    gross_w = wins["pnl_usd"].sum() if len(wins) else 0
    gross_l = abs(loss["pnl_usd"].sum()) if len(loss) else 0.001
    return {
        "n":         len(trades_df),
        "wins":      len(wins),
        "wr":        round(len(wins)/len(trades_df)*100, 1) if len(trades_df) else 0,
        "sl_n":      len(trades_df[trades_df["resultado"]=="SL"]),
        "tp2_n":     len(trades_df[trades_df["resultado"]=="TP2"]),
        "tp1_rate":  round(trades_df["tp1_hit"].mean()*100, 1),
        "cap_final": round(eq[-1], 2),
        "retorno":   round((eq[-1]/CAPITAL_INICIAL-1)*100, 1),
        "max_dd":    round(dd.min(), 1),
        "pf":        round(gross_w/gross_l, 2),
        "sharpe":    round(sharpe, 2),
        "avg_win":   round(wins["pnl_usd"].mean(), 2) if len(wins) else 0,
        "avg_loss":  round(loss["pnl_usd"].mean(), 2) if len(loss) else 0,
        "total_pnl": round(trades_df["pnl_usd"].sum(), 2),
        "avg_dur":   round(trades_df["duracion_h"].mean(), 1),
        "eq_arr":    eq,
        "dd_arr":    dd,
    }

# ─────────────────────────────────────────
# 9. ANÁLISIS DE DISTRIBUCIÓN DE REGÍMENES
# ─────────────────────────────────────────
def analizar_distribucion_regimenes(df):
    dist = df["regime"].value_counts()
    total = len(df)
    print("\n  DISTRIBUCIÓN DE REGÍMENES (% tiempo en cada uno):")
    print(f"  {'Régimen':<20} {'Horas':>8} {'%':>7}")
    print(f"  {'─'*38}")
    for r, n in dist.items():
        print(f"  {r:<20} {n:>8,}  {n/total*100:>6.1f}%")
    return dist

# ─────────────────────────────────────────
# 10. GRÁFICA MAESTRA
# ─────────────────────────────────────────
def graficar_regimenes(df, all_trades, scenarios, regime_stats):
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "BTC/USDC 1H — Análisis por Régimen de Mercado\n"
        "Golden Pocket Fibonacci | HotMarketGuy | 2021–2026",
        fontsize=14, fontweight="bold", color="#f0f6fc", y=0.99
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

    def style(ax, title="", fs=9):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        if title:
            ax.set_title(title, color="#f0f6fc", fontsize=fs, fontweight="bold", pad=5)

    # ── Panel 1: Precio BTC coloreado por régimen ──────────────────────────
    ax_price = fig.add_subplot(gs[0, :])
    style(ax_price, "BTC/USDC — Precio 1H coloreado por Régimen de Mercado", fs=10)

    # Submuestra cada 4 velas para no saturar
    df_sub = df.iloc[::4].copy()
    prev_r = None
    seg_x, seg_y, seg_r = [], [], None
    for i, (ts, row) in enumerate(df_sub.iterrows()):
        r = row["regime"]
        if r != seg_r:
            if seg_x:
                ax_price.plot(seg_x, seg_y, color=REGIME_COLORS.get(seg_r, "#666"),
                              linewidth=1.0, alpha=0.85)
            seg_x, seg_y, seg_r = [ts], [row["close"]], r
        else:
            seg_x.append(ts)
            seg_y.append(row["close"])
    if seg_x:
        ax_price.plot(seg_x, seg_y, color=REGIME_COLORS.get(seg_r, "#666"),
                      linewidth=1.0, alpha=0.85)

    # EMAs
    ax_price.plot(df.index[::4], df["ema200"].iloc[::4],
                  color="#ffffff", linewidth=0.8, alpha=0.4, linestyle="--", label="EMA200")
    ax_price.plot(df.index[::4], df["ema55"].iloc[::4],
                  color="#7986cb", linewidth=0.6, alpha=0.4, linestyle="--", label="EMA55")

    ax_price.set_ylabel("USD", color="#8b949e", fontsize=8)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Leyenda regímenes
    patches = [mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()]
    ax_price.legend(handles=patches, loc="upper left", facecolor="#21262d",
                    labelcolor="#f0f6fc", fontsize=7, ncol=3)

    # ── Panel 2: % tiempo por régimen ──────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 0])
    style(ax_dist, "% Tiempo por Régimen")
    dist = df["regime"].value_counts()
    order = ["BULL_FUERTE","BULL_MODERADO","RECUPERACION","RANGO","BEAR_MODERADO","BEAR_FUERTE"]
    vals  = [dist.get(r, 0)/len(df)*100 for r in order]
    cols  = [REGIME_COLORS[r] for r in order]
    bars  = ax_dist.barh(order, vals, color=cols, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax_dist.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="#f0f6fc", fontsize=7)
    ax_dist.set_xlabel("% del tiempo", color="#8b949e", fontsize=7)
    ax_dist.tick_params(axis="y", labelsize=6.5)

    # ── Panel 3: Win Rate por régimen ───────────────────────────────────────
    ax_wr = fig.add_subplot(gs[1, 1])
    style(ax_wr, "Win Rate por Régimen")
    wrs = []
    for r in order:
        st = regime_stats.get(r, {})
        wrs.append(st.get("wr", 0))
    bar_cols = ["#3fb950" if w >= 45 else "#f85149" for w in wrs]
    bars = ax_wr.barh(order, wrs, color=bar_cols, alpha=0.85)
    ax_wr.axvline(45, color="#8b949e", linestyle="--", lw=0.8, alpha=0.7)
    ax_wr.axvline(50, color="#3fb950", linestyle="--", lw=0.8, alpha=0.5)
    for bar, val in zip(bars, wrs):
        ax_wr.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                   f"{val:.0f}%", va="center", color="#f0f6fc", fontsize=7)
    ax_wr.set_xlabel("Win Rate %", color="#8b949e", fontsize=7)
    ax_wr.tick_params(axis="y", labelsize=6.5)
    ax_wr.set_xlim(0, 75)

    # ── Panel 4: Profit Factor por régimen ────────────────────────────────
    ax_pf = fig.add_subplot(gs[1, 2])
    style(ax_pf, "Profit Factor por Régimen")
    pfs = [min(regime_stats.get(r, {}).get("pf", 0), 3.5) for r in order]
    bar_cols = ["#3fb950" if p >= 1.0 else "#f85149" for p in pfs]
    bars = ax_pf.barh(order, pfs, color=bar_cols, alpha=0.85)
    ax_pf.axvline(1.0, color="#8b949e", linestyle="--", lw=0.8)
    for bar, val in zip(bars, pfs):
        ax_pf.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                   f"{val:.2f}", va="center", color="#f0f6fc", fontsize=7)
    ax_pf.set_xlabel("Profit Factor", color="#8b949e", fontsize=7)
    ax_pf.tick_params(axis="y", labelsize=6.5)

    # ── Panel 5-7: Equity curves por escenario ─────────────────────────────
    scenario_labels = {
        "todos":         ("Todos los regímenes",   "#8b949e"),
        "solo_bull":     ("Solo Bull (F+M)",        "#3fb950"),
        "positivos":     ("Regímenes Positivos",    "#58a6ff"),
    }
    for idx_s, (sc_key, (sc_label, sc_color)) in enumerate(scenario_labels.items()):
        ax = fig.add_subplot(gs[2, idx_s])
        style(ax, f"Equity — {sc_label}")
        eq = np.array(scenarios[sc_key]["equity"])
        ax.plot(eq, color=sc_color, linewidth=1.3, zorder=3)
        ax.axhline(CAPITAL_INICIAL, color="#8b949e", linestyle="--", lw=0.8, alpha=0.5)
        ax.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                        where=[e >= CAPITAL_INICIAL for e in eq],
                        color="#3fb950", alpha=0.15)
        ax.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                        where=[e < CAPITAL_INICIAL for e in eq],
                        color="#f85149", alpha=0.15)
        final = eq[-1]
        ret = (final/CAPITAL_INICIAL - 1)*100
        col_ret = "#3fb950" if ret >= 0 else "#f85149"
        ax.text(0.97, 0.95, f"${final:,.0f}\n{ret:+.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                color=col_ret, fontsize=9, fontweight="bold")
        sc_stats = scenarios[sc_key]["stats"]
        ax.text(0.03, 0.05,
                f"WR: {sc_stats.get('wr',0)}%\nPF: {sc_stats.get('pf',0)}\n"
                f"DD: {sc_stats.get('max_dd',0)}%",
                transform=ax.transAxes, ha="left", va="bottom",
                color="#8b949e", fontsize=7)
        ax.set_ylabel("USD", color="#8b949e", fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        n = scenarios[sc_key]["stats"].get("n", 0)
        ax.set_xlabel(f"{n} trades", color="#8b949e", fontsize=7)

    # ── Panel 8: PnL anual por régimen (heatmap) ──────────────────────────
    ax_heat = fig.add_subplot(gs[3, :])
    style(ax_heat, "PnL Mensual — Solo Regímenes Positivos (Bull Fuerte + Bull Moderado + Recuperación)")

    pos_trades = scenarios["positivos"]["trades"]
    if not pos_trades.empty:
        pivot = pos_trades.pivot_table(
            values="pnl_usd", index="regime", columns="año",
            aggfunc="sum", fill_value=0
        )
        pivot = pivot.reindex([r for r in order if r in pivot.index])
        im = ax_heat.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                            vmin=-1000, vmax=1000)
        ax_heat.set_xticks(range(len(pivot.columns)))
        ax_heat.set_xticklabels(pivot.columns, color="#f0f6fc", fontsize=8)
        ax_heat.set_yticks(range(len(pivot.index)))
        ax_heat.set_yticklabels(pivot.index, color="#f0f6fc", fontsize=7)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax_heat.text(j, i, f"${val:,.0f}",
                             ha="center", va="center",
                             color="white" if abs(val) > 300 else "#0d1117",
                             fontsize=7, fontweight="bold")
        plt.colorbar(im, ax=ax_heat, label="PnL USD",
                     shrink=0.6).ax.yaxis.label.set_color("#f0f6fc")

    plt.savefig("/home/user/TV/backtest_regimenes.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Gráfica: backtest_regimenes.png")

# ─────────────────────────────────────────
# 11. REPORTE CONSOLA
# ─────────────────────────────────────────
def reporte_regimenes(regime_stats, scenarios):
    sep = "═" * 62
    order = ["BULL_FUERTE","BULL_MODERADO","RECUPERACION","RANGO","BEAR_MODERADO","BEAR_FUERTE"]
    print(f"\n{sep}")
    print("  BTC/USDC 1H — ANÁLISIS POR RÉGIMEN DE MERCADO")
    print(f"  Golden Pocket Fibonacci | HotMarketGuy | 2021-2026")
    print(sep)
    print(f"\n  {'Régimen':<20} {'N':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'PnL':>10} {'Max DD':>8}")
    print(f"  {'─'*60}")
    for r in order:
        s = regime_stats.get(r, {})
        if not s:
            continue
        wr_ind  = "✓" if s.get("wr",0) >= 50 else ("~" if s.get("wr",0) >= 42 else "✗")
        pf_ind  = "✓" if s.get("pf",0) >= 1.2 else ("~" if s.get("pf",0) >= 0.9 else "✗")
        print(f"  {r:<20} {s['n']:>5} {s['wr']:>5.1f}%{wr_ind} "
              f"{s['pf']:>5.2f}{pf_ind} {s['sharpe']:>7.2f} "
              f"${s['total_pnl']:>8,.0f} {s['max_dd']:>7.1f}%")

    print(f"\n{sep}")
    print("  COMPARATIVA DE ESCENARIOS:")
    print(f"  {'Escenario':<30} {'Trades':>7} {'WR':>6} {'PF':>6} {'Retorno':>9} {'Max DD':>8}")
    print(f"  {'─'*60}")
    labels = {
        "todos":     "Todos los regímenes",
        "solo_bull": "Solo Bull (Fuerte + Moderado)",
        "positivos": "Positivos (Bull+Recuperación)",
    }
    for key, label in labels.items():
        s = scenarios[key]["stats"]
        if not s:
            continue
        print(f"  {label:<30} {s['n']:>7} {s['wr']:>5.1f}% "
              f"{s['pf']:>5.2f} {s['retorno']:>+8.1f}% {s['max_dd']:>7.1f}%")

    print(f"\n{sep}")
    print("  RESULTADOS — SOLO REGÍMENES POSITIVOS (por año):")
    pos_t = scenarios["positivos"]["trades"]
    if not pos_t.empty:
        print(f"\n  {'Año':<6} {'Trades':>7} {'WR':>6} {'PnL':>12} {'Acum.':>10}")
        print(f"  {'─'*44}")
        acum = CAPITAL_INICIAL
        for yr, g in pos_t.groupby("año"):
            wr  = (g["pnl_usd"] > 0).mean() * 100
            tot = g["pnl_usd"].sum()
            acum += tot
            sig = "▲" if tot > 0 else "▼"
            print(f"  {yr:<6} {len(g):>7} {wr:>5.0f}% {sig} ${tot:>9,.0f}  ${acum:>9,.0f}")

    print(f"\n{sep}")
    print("  WIN RATE POR RÉGIMEN + AÑO (solo positivos):")
    if not pos_t.empty:
        pivot_wr = pos_t.pivot_table(
            values="pnl_usd",
            index="regime",
            columns="año",
            aggfunc=lambda x: f"{(x>0).mean()*100:.0f}%",
            fill_value="—"
        )
        print(f"\n{pivot_wr.to_string()}")
    print(f"\n{sep}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Generando datos 1H BTC (5 años)...")
    df = generar_ohlcv_1h(BTC_MONTHLY_ANCHORS)
    print(f"  {len(df):,} velas  ({df.index[0].date()} → {df.index[-1].date()})")

    print("Clasificando regímenes de mercado...")
    df = clasificar_regimenes(df)
    analizar_distribucion_regimenes(df)

    print("\nDetectando swings e impulsos...")
    sh, sl_ = detectar_swings(df)
    impulsos = identificar_impulsos(df, sl_, sh)
    print(f"  Impulsos válidos (≥0.75%): {len(impulsos):,}")

    # ── Escenario 1: Todos los regímenes ──────────────────────────────────
    print("\nBacktest — Todos los regímenes...")
    t_all, eq_all = backtest_por_regimen(df, impulsos, regimenes_permitidos=None)
    s_all = stats_regimen(t_all, eq_all)

    # ── Escenario 2: Solo Bull Fuerte + Bull Moderado ─────────────────────
    print("Backtest — Solo Bull Fuerte + Bull Moderado...")
    BULL_SOLO = ["BULL_FUERTE", "BULL_MODERADO"]
    t_bull, eq_bull = backtest_por_regimen(df, impulsos, BULL_SOLO)
    s_bull = stats_regimen(t_bull, eq_bull)

    # ── Escenario 3: Regímenes Positivos (Bull + Recuperación) ────────────
    print("Backtest — Regímenes Positivos (Bull + Recuperación)...")
    POSITIVOS = ["BULL_FUERTE", "BULL_MODERADO", "RECUPERACION"]
    t_pos, eq_pos = backtest_por_regimen(df, impulsos, POSITIVOS)
    s_pos = stats_regimen(t_pos, eq_pos)

    # ── Stats por régimen individual ──────────────────────────────────────
    print("Calculando estadísticas por régimen...")
    regime_stats = {}
    for r in ["BULL_FUERTE","BULL_MODERADO","RECUPERACION","RANGO","BEAR_MODERADO","BEAR_FUERTE"]:
        t_r, eq_r = backtest_por_regimen(df, impulsos, [r])
        if not t_r.empty:
            regime_stats[r] = stats_regimen(t_r, eq_r)

    scenarios = {
        "todos":     {"trades": t_all,  "equity": eq_all,  "stats": s_all},
        "solo_bull": {"trades": t_bull, "equity": eq_bull, "stats": s_bull},
        "positivos": {"trades": t_pos,  "equity": eq_pos,  "stats": s_pos},
    }

    reporte_regimenes(regime_stats, scenarios)

    print("\nGenerando gráficas...")
    graficar_regimenes(df, t_all, scenarios, regime_stats)

    t_pos.to_csv("/home/user/TV/backtest_positivos.csv", index=False)
    print("  CSV: backtest_positivos.csv")
    print("\n  ✓ Análisis por régimen completado.")

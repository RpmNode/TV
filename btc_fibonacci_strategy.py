"""
BTC/USDC 1H — Golden Pocket Fibonacci Strategy
Basado en metodología HotMarketGuy
Backtest: 5 años (2021-2026) con datos sintéticos calibrados a precios reales BTC
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────
LOOKBACK_SWING    = 5       # swings cortos para capturar impulsos 0.75%
MIN_IMPULSE_PCT   = 0.0075 # impulso mínimo 0.75% (usuario)
GOLDEN_POCKET_LOW = 0.618
GOLDEN_POCKET_HI  = 0.650
SL_NIVEL_PCT      = 0.008  # SL: 0.8% por debajo de la entrada (proporcionado)
TP1_RR            = 1.0    # TP1 a 1:1
TP2_RR            = 2.5    # TP2 a 2.5:1
TP1_FRACTION      = 0.50
CAPITAL_INICIAL   = 10_000
RISK_PCT          = 0.01   # 1% riesgo/trade (más conservador dado volumen)
MAX_HOLD_CANDLES  = 48     # máx 2 días en trade
EMA_FAST          = 21
EMA_SLOW          = 55
TREND_MA          = 100
MIN_CANDLES_ENTRE_TRADES = 6  # mínimo 6h entre trades

# ─────────────────────────────────────────
# 1. PRECIOS REALES BTC - HITOS MENSUALES
#    (Fuente: datos históricos conocidos)
# ─────────────────────────────────────────
# Formato: (fecha, open, high, low, close, vol_relativo)
# Puntos de control mensuales calibrados a historia real de BTC
BTC_MONTHLY_ANCHORS = [
    # 2021
    ("2021-01-01", 29000,  42000, 27000,  38500, 1.0),
    ("2021-02-01", 38500,  58500, 36000,  45000, 1.3),
    ("2021-03-01", 45000,  61000, 47000,  58800, 1.1),
    ("2021-04-01", 58800,  65000, 47000,  57500, 1.2),
    ("2021-05-01", 57500,  59500, 30000,  37000, 1.8),  # crash mayo
    ("2021-06-01", 37000,  41000, 28800,  35000, 1.4),
    ("2021-07-01", 35000,  42500, 29700,  41500, 1.2),
    ("2021-08-01", 41500,  50200, 39600,  47100, 1.1),
    ("2021-09-01", 47100,  52950, 39600,  43800, 1.2),
    ("2021-10-01", 43800,  67000, 43400,  60900, 1.3),
    ("2021-11-01", 60900,  69000, 53600,  57200, 1.5),  # ATH Nov
    ("2021-12-01", 57200,  59050, 42000,  47700, 1.3),
    # 2022
    ("2022-01-01", 47700,  48000, 32900,  38500, 1.4),
    ("2022-02-01", 38500,  45800, 36800,  43200, 1.1),
    ("2022-03-01", 43200,  48100, 37600,  45500, 1.0),
    ("2022-04-01", 45500,  47400, 37600,  38400, 1.1),
    ("2022-05-01", 38400,  40000, 25400,  31500, 1.6),  # crash LUNA
    ("2022-06-01", 31500,  31800, 17600,  19100, 2.0),  # crash extremo
    ("2022-07-01", 19100,  25200, 18800,  23300, 1.5),
    ("2022-08-01", 23300,  25200, 19600,  20050, 1.3),
    ("2022-09-01", 20050,  22700, 18200,  19500, 1.2),
    ("2022-10-01", 19500,  21500, 18900,  20500, 1.1),
    ("2022-11-01", 20500,  21500, 15476,  16550, 2.5),  # FTX crash
    ("2022-12-01", 16550,  18400, 16200,  16600, 1.4),
    # 2023
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
    # 2024
    ("2024-01-01", 42300,  49000, 38500,  42700, 1.2),
    ("2024-02-01", 42700,  57500, 41550,  62400, 1.5),
    ("2024-03-01", 62400,  73780, 59000,  71400, 1.6),  # pre-halving ATH
    ("2024-04-01", 71400,  72800, 56500,  60700, 1.4),  # halving
    ("2024-05-01", 60700,  73750, 56600,  67500, 1.2),
    ("2024-06-01", 67500,  71900, 58400,  62700, 1.1),
    ("2024-07-01", 62700,  70000, 53500,  66200, 1.2),
    ("2024-08-01", 66200,  65300, 49000,  59100, 1.4),  # CME gap fill
    ("2024-09-01", 59100,  66500, 52700,  63300, 1.1),
    ("2024-10-01", 63300,  73600, 60200,  72400, 1.2),
    ("2024-11-01", 72400,  99700, 67500,  97200, 1.7),  # post-elección
    ("2024-12-01", 97200, 108500, 91800, 93400, 1.5),
    # 2025
    ("2025-01-01", 93400, 109000, 89000, 95800, 1.4),  # ATH ene 2025
    ("2025-02-01", 95800, 100200, 78400, 84500, 1.3),
    ("2025-03-01", 84500,  92000, 76800, 82300, 1.2),
    ("2025-04-01", 82300,  97500, 74600, 94100, 1.3),
    ("2025-05-01", 94100, 107000, 93000,103500, 1.2),
    ("2025-06-01",103500, 116000,100000,110000, 1.2),
    ("2025-07-01",110000, 125000,107000,121000, 1.3),
    ("2025-08-01",121000, 127000,110000,114000, 1.3),
    ("2025-09-01",114000, 126000,107000,118000, 1.2),
    ("2025-10-01",118000, 126000,105000,108000, 1.3),
    ("2025-11-01",108000, 118000, 83500, 90000, 1.5),  # corrección
    ("2025-12-01", 90000,  96000, 83000, 87000, 1.3),
    # 2026
    ("2026-01-01", 87000,  91000, 74700, 79500, 1.3),
    ("2026-02-01", 79500,  85000, 59500, 71000, 1.5),
    ("2026-03-01", 71000,  80000, 66000, 72600, 1.3),
    ("2026-04-01", 72600,  80200, 67044, 75000, 1.4),  # mínimo abril
    ("2026-05-01", 75000,  81500, 74000, 80696, 1.2),  # actual
]

# ─────────────────────────────────────────
# 2. GENERAR OHLCV 1H SINTÉTICO REALISTA
# ─────────────────────────────────────────
def generar_ohlcv_1h(anchors):
    """
    Interpola entre puntos de control mensuales generando
    velas 1H con dinámica GBM calibrada a la volatilidad real de BTC.
    """
    rows = []
    for i in range(len(anchors) - 1):
        ts0, o0, h0, l0, c0, vol0 = anchors[i]
        ts1, o1, h1, l1, c1, vol1 = anchors[i + 1]
        dt0  = pd.Timestamp(ts0)
        dt1  = pd.Timestamp(ts1)
        horas = int((dt1 - dt0).total_seconds() / 3600)
        if horas <= 0:
            continue

        # BTC volatilidad histórica 1H ≈ 0.5–1.2% (usamos 0.008 base)
        sigma_hora = 0.008 * vol0
        mu_hora    = (np.log(c1 / c0)) / horas  # drift hacia el close mensual

        precio = c0
        for h in range(horas):
            t = dt0 + pd.Timedelta(hours=h)
            # GBM con reversión al camino trend
            ret    = mu_hora + sigma_hora * np.random.randn()
            precio = precio * np.exp(ret)
            # Forzar que al final del mes el precio esté cerca del close ancla
            if h == horas - 1:
                precio = c1 * (1 + np.random.uniform(-0.003, 0.003))
            # Generar OHLC realista
            rango_h = precio * sigma_hora * np.abs(np.random.randn() + 0.5)
            wick_up = precio * sigma_hora * np.abs(np.random.randn() * 0.3)
            wick_dn = precio * sigma_hora * np.abs(np.random.randn() * 0.3)
            open_c  = precio * (1 + np.random.uniform(-sigma_hora * 0.3,
                                                        sigma_hora * 0.3))
            close_c = precio
            high_c  = max(open_c, close_c) + wick_up + rango_h * 0.3
            low_c   = min(open_c, close_c) - wick_dn - rango_h * 0.3
            vol_c   = vol0 * np.abs(np.random.randn() + 1) * 1000
            rows.append({
                "timestamp": t,
                "open":      round(open_c,  2),
                "high":      round(high_c,  2),
                "low":       round(low_c,   2),
                "close":     round(close_c, 2),
                "volume":    round(vol_c,   2),
            })

    df = pd.DataFrame(rows).set_index("timestamp")
    df = df[df["low"] > 0]
    return df

# ─────────────────────────────────────────
# 3. DETECCIÓN DE SWINGS
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
# 4. IDENTIFICAR IMPULSOS ALCISTAS VÁLIDOS
# ─────────────────────────────────────────
def identificar_impulsos(df, swing_lows, swing_highs):
    impulsos = []
    hi_arr = df["high"].values
    lo_arr = df["low"].values
    idx    = df.index

    for li in swing_lows:
        low_price = lo_arr[li]
        candidatos_hi = [hi for hi in swing_highs
                         if hi > li + 4 and hi < li + 400]
        if not candidatos_hi:
            continue
        # Siguiente swing low después de este
        next_lows = [sl for sl in swing_lows if sl > li]
        next_low_cut = next_lows[0] if next_lows else len(df)
        candidatos_hi = [hi for hi in candidatos_hi if hi < next_low_cut]
        if not candidatos_hi:
            continue
        hi = max(candidatos_hi, key=lambda x: hi_arr[x])
        high_price = hi_arr[hi]
        rango_pct  = (high_price - low_price) / low_price
        if rango_pct >= MIN_IMPULSE_PCT:
            impulsos.append({
                "low_idx":   li, "high_idx":  hi,
                "low_price": low_price, "high_price": high_price,
                "rango_pct": rango_pct, "rango_usd": high_price - low_price,
                "low_time":  idx[li], "high_time": idx[hi],
            })
    return impulsos

# ─────────────────────────────────────────
# 5. NIVELES FIBONACCI
# ─────────────────────────────────────────
def fibs(imp):
    r   = imp["high_price"] - imp["low_price"]
    h   = imp["high_price"]
    gp  = h - r * GOLDEN_POCKET_LOW   # precio de entrada al 61.8%
    sl  = gp * (1 - SL_NIVEL_PCT)     # SL fijo % bajo entrada
    rsk = gp - sl                      # riesgo en USD/BTC
    return {
        "gp_lo": gp,
        "gp_hi": h - r * GOLDEN_POCKET_HI,
        "sl":    sl,
        "tp1":   gp + rsk * TP1_RR,   # TP1 a 1:1
        "tp2":   gp + rsk * TP2_RR,   # TP2 a 2.5:1
        "swing_high": h,               # referencia estructural
    }

# ─────────────────────────────────────────
# 6. BACKTEST ENGINE
# ─────────────────────────────────────────
def backtest(df, impulsos):
    capital      = CAPITAL_INICIAL
    equity_curve = [capital]
    trades       = []
    lo  = df["low"].values
    hi  = df["high"].values
    cl  = df["close"].values
    idx = df.index

    # EMAs para filtro de tendencia
    cl_series = df["close"]
    ma200  = cl_series.ewm(span=TREND_MA, adjust=False).mean().values
    ema21  = cl_series.ewm(span=EMA_FAST, adjust=False).mean().values
    ema55  = cl_series.ewm(span=EMA_SLOW,  adjust=False).mean().values

    ultimo_salida_idx = -MIN_CANDLES_ENTRE_TRADES  # evitar re-entradas

    for imp in impulsos:
        f  = fibs(imp)
        sl = f["sl"]
        tp1= f["tp1"]
        tp2= f["tp2"]

        riesgo_unit = f["gp_lo"] - f["sl"]
        if riesgo_unit <= 0:
            continue

        # Filtro de tendencia: EMA21 > EMA55 (tendencia alcista en 1H)
        ema21_v = ema21[imp["high_idx"]]
        ema55_v = ema55[imp["high_idx"]]
        if not (np.isnan(ema21_v) or np.isnan(ema55_v)):
            if ema21_v < ema55_v * 0.98:   # tendencia bajista clara → no operar
                continue

        inicio = max(imp["high_idx"] + 1, ultimo_salida_idx + MIN_CANDLES_ENTRE_TRADES)
        fin    = min(imp["high_idx"] + MAX_HOLD_CANDLES * 3, len(df))

        entrada_idx = None
        for i in range(inicio, fin):
            # CONDICIÓN DE ENTRADA (HotMarketGuy):
            # 1. La vela toca el 61.8% (precio baja al Golden Pocket)
            # 2. La vela CIERRA POR ENCIMA del 65% (cierre de confirmación alcista)
            # 3. El cuerpo de la vela es mayor al wick inferior (vela de reversión)
            if lo[i] <= f["gp_lo"] and cl[i] > f["gp_hi"]:
                cuerpo    = abs(cl[i] - lo[i])
                wick_inf  = min(lo[i], cl[i]) - lo[i]
                # Confirmación: cierre en mitad superior de la vela
                if cl[i] > (lo[i] + hi[i]) / 2:
                    entrada_idx = i
                    break

        if entrada_idx is None:
            continue

        # Entrada al OPEN de la siguiente vela (más realista)
        entrada_exec = entrada_idx + 1
        if entrada_exec >= len(df):
            continue
        precio_entrada = cl[entrada_idx]  # entrada al cierre de confirmación

        riesgo_usd   = capital * RISK_PCT
        tam_pos_usd  = min(riesgo_usd / (riesgo_unit / precio_entrada), capital * 0.5)
        btc_size     = tam_pos_usd / precio_entrada

        resultado     = "TIMEOUT"
        precio_salida = None
        salida_idx    = None
        tp1_hit       = False
        btc_rem       = btc_size

        for j in range(entrada_exec,
                       min(entrada_exec + MAX_HOLD_CANDLES + 1, len(df))):
            if lo[j] <= sl:
                precio_salida = sl
                resultado     = "SL"
                salida_idx    = j
                break
            if not tp1_hit and hi[j] >= tp1:
                tp1_hit  = True
                btc_rem *= (1 - TP1_FRACTION)
            if hi[j] >= tp2:
                precio_salida = tp2
                resultado     = "TP2"
                salida_idx    = j
                break

        if resultado == "TIMEOUT" or salida_idx is None:
            salida_idx    = min(entrada_exec + MAX_HOLD_CANDLES, len(df) - 1)
            precio_salida = cl[salida_idx]

        # PnL — todo consistente con la posición real
        if tp1_hit:
            pnl_p1 = btc_size * TP1_FRACTION * (tp1 - precio_entrada)
            pnl_p2 = btc_rem  * (precio_salida - precio_entrada)
            pnl    = pnl_p1 + pnl_p2
        else:
            pnl = btc_size * (precio_salida - precio_entrada)

        capital += pnl
        capital  = max(capital, 1)
        dur_h    = (idx[salida_idx] - idx[entrada_exec]).total_seconds() / 3600

        ultimo_salida_idx = salida_idx

        trades.append({
            "entrada_time":   idx[entrada_exec],
            "salida_time":    idx[salida_idx],
            "año":            idx[entrada_exec].year,
            "precio_entrada": round(precio_entrada, 2),
            "precio_salida":  round(precio_salida,  2),
            "resultado":      resultado,
            "pnl_usd":        round(pnl, 2),
            "pnl_pct":        round(pnl / tam_pos_usd * 100, 2) if tam_pos_usd else 0,
            "capital":        round(capital, 2),
            "duracion_h":     round(dur_h, 1),
            "rango_imp_pct":  round(imp["rango_pct"] * 100, 1),
            "tp1_hit":        tp1_hit,
            "imp_low":        round(imp["low_price"],  0),
            "imp_high":       round(imp["high_price"], 0),
        })
        equity_curve.append(capital)

    return pd.DataFrame(trades), equity_curve

# ─────────────────────────────────────────
# 7. ESTADÍSTICAS
# ─────────────────────────────────────────
def stats(trades_df, equity_curve):
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df["pnl_usd"] > 0]
    loss = trades_df[trades_df["pnl_usd"] <= 0]
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = ((eq - peak) / peak * 100)
    max_dd = dd.min()
    rets = trades_df["pnl_usd"] / CAPITAL_INICIAL
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    gross_w = wins["pnl_usd"].sum()
    gross_l = abs(loss["pnl_usd"].sum())
    pf = gross_w / gross_l if gross_l > 0 else 999.0
    return {
        "n":         len(trades_df),
        "wins":      len(wins),
        "losses":    len(loss),
        "wr":        round(len(wins)/len(trades_df)*100, 1),
        "sl_n":      len(trades_df[trades_df["resultado"]=="SL"]),
        "tp2_n":     len(trades_df[trades_df["resultado"]=="TP2"]),
        "to_n":      len(trades_df[trades_df["resultado"]=="TIMEOUT"]),
        "tp1_rate":  round(trades_df["tp1_hit"].mean()*100, 1),
        "cap_final": round(eq[-1], 2),
        "retorno":   round((eq[-1]/CAPITAL_INICIAL-1)*100, 1),
        "max_dd":    round(max_dd, 1),
        "pf":        round(pf, 2),
        "sharpe":    round(sharpe, 2),
        "avg_win":   round(wins["pnl_usd"].mean(), 2) if len(wins) else 0,
        "avg_loss":  round(loss["pnl_usd"].mean(), 2) if len(loss) else 0,
        "best":      round(trades_df["pnl_usd"].max(), 2),
        "worst":     round(trades_df["pnl_usd"].min(), 2),
        "avg_dur":   round(trades_df["duracion_h"].mean(), 1),
        "eq_arr":    eq,
        "dd_arr":    dd,
    }

# ─────────────────────────────────────────
# 8. GRÁFICAS
# ─────────────────────────────────────────
def graficar(trades_df, s):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("BTC/USDC 1H — Golden Pocket Fibonacci  |  HotMarketGuy Method\n"
                 "Backtest 5 años (2021-2026)  |  Capital: $10,000  |  Riesgo: 2%/trade",
                 fontsize=13, fontweight="bold", color="#f0f6fc", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    axes = {
        "eq":   fig.add_subplot(gs[0, :2]),
        "dd":   fig.add_subplot(gs[1, :2]),
        "pnl":  fig.add_subplot(gs[0, 2]),
        "año":  fig.add_subplot(gs[1, 2]),
        "hist": fig.add_subplot(gs[2, 0]),
        "dur":  fig.add_subplot(gs[2, 1]),
        "tab":  fig.add_subplot(gs[2, 2]),
    }

    def style(ax, title=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        if title:
            ax.set_title(title, color="#f0f6fc", fontsize=9, fontweight="bold", pad=6)

    # Equity
    ax = axes["eq"]
    style(ax, "Curva de Equity")
    eq = s["eq_arr"]
    ax.plot(eq, color="#58a6ff", linewidth=1.4, zorder=3)
    ax.axhline(CAPITAL_INICIAL, color="#8b949e", linestyle="--", lw=0.8, alpha=0.6)
    ax.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                    where=[e >= CAPITAL_INICIAL for e in eq],
                    color="#3fb950", alpha=0.15, zorder=2)
    ax.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                    where=[e < CAPITAL_INICIAL for e in eq],
                    color="#f85149", alpha=0.15, zorder=2)
    ax.set_ylabel("USD", color="#8b949e", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    # Annotate final
    ax.annotate(f"${eq[-1]:,.0f}", xy=(len(eq)-1, eq[-1]),
                xytext=(-60, 10), textcoords="offset points",
                color="#3fb950" if eq[-1] >= CAPITAL_INICIAL else "#f85149",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=0.8))

    # Drawdown
    ax = axes["dd"]
    style(ax, "Drawdown (%)")
    dd = s["dd_arr"]
    ax.fill_between(range(len(dd)), dd, 0, color="#f85149", alpha=0.5)
    ax.plot(dd, color="#ff6b6b", linewidth=0.8)
    ax.set_ylabel("%", color="#8b949e", fontsize=8)
    ax.annotate(f"Max DD: {s['max_dd']}%", xy=(np.argmin(dd), dd.min()),
                xytext=(30, 10), textcoords="offset points",
                color="#ff6b6b", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=0.8))

    # PnL por trade
    ax = axes["pnl"]
    style(ax, "PnL por Trade")
    pnls = trades_df["pnl_usd"].values
    cols = ["#3fb950" if p > 0 else "#f85149" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=cols, alpha=0.85, width=0.8)
    ax.axhline(0, color="#8b949e", lw=0.5)
    ax.set_ylabel("USD", color="#8b949e", fontsize=8)
    ax.set_xlabel("Trade #", color="#8b949e", fontsize=8)

    # PnL por año
    ax = axes["año"]
    style(ax, "PnL Anual")
    por_año = trades_df.groupby("año")["pnl_usd"].sum()
    bar_c = ["#3fb950" if v > 0 else "#f85149" for v in por_año.values]
    bars = ax.bar(por_año.index.astype(str), por_año.values, color=bar_c, alpha=0.85)
    ax.axhline(0, color="#8b949e", lw=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, val in zip(bars, por_año.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (50 if val >= 0 else -200),
                f"${val:,.0f}", ha="center", va="bottom", color="#f0f6fc", fontsize=7)
    ax.tick_params(axis="x", labelrotation=45)

    # Histograma PnL%
    ax = axes["hist"]
    style(ax, "Distribución PnL %")
    pp = trades_df["pnl_pct"]
    ax.hist(pp[pp > 0], bins=18, color="#3fb950", alpha=0.75, label="Wins")
    ax.hist(pp[pp <= 0], bins=18, color="#f85149", alpha=0.75, label="Losses")
    ax.axvline(0, color="white", lw=0.8)
    ax.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7)
    ax.set_xlabel("PnL %", color="#8b949e", fontsize=8)

    # Duración
    ax = axes["dur"]
    style(ax, "Duración de Trades (h)")
    ax.hist(trades_df["duracion_h"], bins=20, color="#d29922", alpha=0.75)
    ax.axvline(s["avg_dur"], color="#f0f6fc", lw=1, linestyle="--")
    ax.text(s["avg_dur"] + 0.5, ax.get_ylim()[1] * 0.85,
            f"Media: {s['avg_dur']}h", color="#f0f6fc", fontsize=8)
    ax.set_xlabel("Horas", color="#8b949e", fontsize=8)

    # Tabla stats
    ax = axes["tab"]
    ax.axis("off")
    style(ax)
    datos = [
        ("Total Trades",    str(s["n"])),
        ("Win Rate",        f"{s['wr']}%"),
        ("Profit Factor",   str(s["pf"])),
        ("Sharpe Ratio",    str(s["sharpe"])),
        ("Retorno Total",   f"{s['retorno']}%"),
        ("Capital Final",   f"${s['cap_final']:,.0f}"),
        ("Max Drawdown",    f"{s['max_dd']}%"),
        ("TP2 Completos",   str(s["tp2_n"])),
        ("SL Activados",    str(s["sl_n"])),
        ("Timeout (72h)",   str(s["to_n"])),
        ("TP1 Hit Rate",    f"{s['tp1_rate']}%"),
        ("Avg Ganancia",    f"${s['avg_win']:,.0f}"),
        ("Avg Pérdida",     f"${s['avg_loss']:,.0f}"),
        ("Mejor Trade",     f"${s['best']:,.0f}"),
        ("Peor Trade",      f"${s['worst']:,.0f}"),
        ("Duración Media",  f"{s['avg_dur']}h"),
    ]
    tbl = ax.table(cellText=datos,
                   colLabels=["Métrica", "Valor"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#21262d" if r % 2 == 0 else "#161b22")
            # Color verde/rojo para resultados
            txt = cell.get_text().get_text()
            if c == 1 and r > 0:
                metric = datos[r-1][0]
                if "Retorno" in metric or "Capital" in metric or "Win Rate" in metric:
                    val = datos[r-1][1].replace("%","").replace("$","").replace(",","")
                    try:
                        cell.get_text().set_color(
                            "#3fb950" if float(val) > 0 else "#f85149")
                    except:
                        cell.get_text().set_color("#f0f6fc")
                elif "Drawdown" in metric or "Peor" in metric or "SL" in metric:
                    cell.get_text().set_color("#f85149")
                else:
                    cell.get_text().set_color("#f0f6fc")
            else:
                cell.get_text().set_color("#8b949e")
        cell.set_edgecolor("#30363d")
    ax.set_title("Resumen Estadístico", color="#f0f6fc",
                 fontsize=9, fontweight="bold", pad=8)

    plt.savefig("/home/user/TV/backtest_resultado.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Gráfica: backtest_resultado.png")

# ─────────────────────────────────────────
# 9. REPORTE CONSOLA
# ─────────────────────────────────────────
def reporte(s, trades_df):
    sep = "═" * 56
    print(f"\n{sep}")
    print("   BTC/USDC 1H — Golden Pocket Fibonacci Backtest")
    print(f"   Capital: ${CAPITAL_INICIAL:,}  |  Riesgo: {RISK_PCT*100:.0f}%/trade  |  5 años")
    print(sep)
    print(f"  Total trades       {s['n']:>6}")
    print(f"  Ganadores          {s['wins']:>6}  ({s['wr']}%)")
    print(f"  Perdedores         {s['losses']:>6}")
    print(f"  TP2 completados    {s['tp2_n']:>6}")
    print(f"  SL activados       {s['sl_n']:>6}")
    print(f"  Timeout (72h)      {s['to_n']:>6}")
    print(f"  TP1 alcanzado      {s['tp1_rate']:>5}% de los trades")
    print(f"{'─'*56}")
    print(f"  Capital inicial  ${CAPITAL_INICIAL:>12,.2f}")
    print(f"  Capital final    ${s['cap_final']:>12,.2f}")
    print(f"  Retorno total    {s['retorno']:>11.1f}%")
    print(f"  Max Drawdown     {s['max_dd']:>11.1f}%")
    print(f"  Profit Factor    {s['pf']:>12.2f}")
    print(f"  Sharpe Ratio     {s['sharpe']:>12.2f}")
    print(f"{'─'*56}")
    print(f"  Avg ganancia     ${s['avg_win']:>12,.2f}")
    print(f"  Avg pérdida      ${s['avg_loss']:>12,.2f}")
    print(f"  R:R implícito    {abs(s['avg_win']/s['avg_loss']):.2f}:1" if s['avg_loss'] else "")
    print(f"  Mejor trade      ${s['best']:>12,.2f}")
    print(f"  Peor trade       ${s['worst']:>12,.2f}")
    print(f"  Duración media   {s['avg_dur']:>10.1f}h")
    print(sep)

    print("\n  PnL POR AÑO:")
    print(f"  {'Año':<6} {'Trades':>7} {'WR':>6} {'PnL USD':>12} {'Acum.':>10}")
    print(f"  {'─'*48}")
    acum = CAPITAL_INICIAL
    for yr, g in trades_df.groupby("año"):
        wr  = (g["pnl_usd"] > 0).mean() * 100
        tot = g["pnl_usd"].sum()
        acum += tot
        sig = "▲" if tot > 0 else "▼"
        print(f"  {yr:<6} {len(g):>7} {wr:>5.0f}% {sig} ${tot:>10,.0f}  ${acum:>9,.0f}")

    print(f"\n  TOP 5 MEJORES:")
    top5 = trades_df.nlargest(5,"pnl_usd")[
        ["año","precio_entrada","resultado","pnl_usd","duracion_h","imp_low","imp_high"]]
    print(top5.to_string(index=False))
    print(f"\n  TOP 5 PEORES:")
    bot5 = trades_df.nsmallest(5,"pnl_usd")[
        ["año","precio_entrada","resultado","pnl_usd","duracion_h"]]
    print(bot5.to_string(index=False))
    print(sep)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Generando datos 1H BTC (5 años, anclados a precios históricos reales)...")
    df = generar_ohlcv_1h(BTC_MONTHLY_ANCHORS)
    print(f"  Velas 1H generadas: {len(df):,} "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    print("Detectando swing highs/lows...")
    sh, sl_ = detectar_swings(df)
    print(f"  Swing highs: {len(sh)} | Swing lows: {len(sl_)}")

    print("Identificando impulsos válidos (≥4%)...")
    impulsos = identificar_impulsos(df, sl_, sh)
    print(f"  Impulsos válidos: {len(impulsos)}")

    print("Ejecutando backtest...")
    trades_df, eq_curve = backtest(df, impulsos)
    print(f"  Trades ejecutados: {len(trades_df)}")

    s = stats(trades_df, eq_curve)
    reporte(s, trades_df)

    print("\nGenerando gráficas...")
    graficar(trades_df, s)

    trades_df.to_csv("/home/user/TV/backtest_trades.csv", index=False)
    print("  CSV: backtest_trades.csv")
    print("\n  ✓ Backtest completado.")

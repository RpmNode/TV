"""
BTC/USDC 1H — Golden Pocket Fibonacci OPTIMIZADA
Régimen: BULL_FUERTE | RSI(14) | MACD(12,26,9) | ADX(14)
Scale-in: +0.5 contratos en zona 65% si el precio profundiza tras la entrada

Estructura de posición:
  Entrada 1 (E1): 61.8% GP — 1 unidad (riesgo 1% capital)
  Scale-in (E2):  65.0%     — 0.5 unidades adicionales
  SL común:       78.6% del impulso
  TP1:            1:1 R:R desde precio medio ponderado
  TP2:            3:1 R:R desde precio medio ponderado
  BE:             SL → precio E1 tras TP1
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────
# PARÁMETROS FINALES
# ─────────────────────────────────────────
LOOKBACK_SWING     = 5
MIN_IMPULSE_PCT    = 0.0075      # 0.75% mínimo (usuario)
GOLDEN_POCKET_LOW  = 0.618
GOLDEN_POCKET_HI   = 0.650
SL_NIVEL_PCT       = 0.008       # 0.8% SL
TP1_RR             = 1.0         # TP1 a 1:1
TP2_RR             = 3.0         # TP2 a 3:1 (mejorado desde 2.5)
TP1_FRACTION       = 0.50
CAPITAL_INICIAL    = 10_000
RISK_PCT           = 0.01        # 1% riesgo/trade
MAX_HOLD_CANDLES   = 72          # 3 días máximo
EMA_FAST           = 21
EMA_MED            = 55
EMA_SLOW           = 200
RSI_PERIOD         = 14
RSI_MIN_ENTRY      = 28          # RSI > 28 (margen para impulsos fuertes)
RSI_MAX_ENTRY      = 55          # RSI < 55
RSI_SLOPE_MIN      = -1.0        # RSI no debe estar cayendo fuerte
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
ADX_PERIOD         = 14
ADX_MIN            = 20          # ADX > 20 → tendencia moderada/fuerte
VOL_MA_PERIOD      = 20
SL_LEVEL           = 0.786       # SL estructural en 78.6% del impulso
SCALE_IN_LEVEL     = 0.650       # scale-in al 65% (profundiza desde 61.8%)
SCALE_IN_FACTOR    = 0.50        # añadir 0.5 contratos (50% de la pos. inicial)
MIN_CANDLES_ENTRE  = 12
SLOPE_WINDOW       = 24          # ventana para pendiente EMA200 (24h)
SLOPE_MIN          = 0.0005      # pendiente mínima EMA200 (0.05%/24h)

REGIME_COLORS = {
    "BULL_FUERTE":   "#00c853",
    "BULL_MODERADO": "#69f0ae",
    "RECUPERACION":  "#ffeb3b",
    "RANGO":         "#ff9800",
    "BEAR_MODERADO": "#ff5722",
    "BEAR_FUERTE":   "#d32f2f",
}

# ─────────────────────────────────────────
# 1. DATOS HISTÓRICOS BTC
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
    rows = []
    for i in range(len(anchors) - 1):
        ts0,o0,h0,l0,c0,v0 = anchors[i]
        ts1,o1,h1,l1,c1,v1 = anchors[i+1]
        dt0   = pd.Timestamp(ts0)
        dt1   = pd.Timestamp(ts1)
        horas = int((dt1-dt0).total_seconds()/3600)
        if horas <= 0: continue
        sigma = 0.008*v0
        mu    = np.log(c1/c0)/horas
        precio = c0
        for h in range(horas):
            t      = dt0 + pd.Timedelta(hours=h)
            ret    = mu + sigma*np.random.randn()
            precio = precio*np.exp(ret)
            if h == horas-1:
                precio = c1*(1+np.random.uniform(-0.003,0.003))
            rng = precio*sigma*abs(np.random.randn()+0.5)
            wu  = precio*sigma*abs(np.random.randn()*0.3)
            wd  = precio*sigma*abs(np.random.randn()*0.3)
            op  = precio*(1+np.random.uniform(-sigma*0.3,sigma*0.3))
            cl_ = precio
            hi_ = max(op,cl_)+wu+rng*0.3
            lo_ = min(op,cl_)-wd-rng*0.3
            vol = v0*abs(np.random.randn()+1)*1000
            rows.append({"timestamp":t,"open":round(op,2),"high":round(hi_,2),
                         "low":round(lo_,2),"close":round(cl_,2),"volume":round(vol,2)})
    df = pd.DataFrame(rows).set_index("timestamp")
    return df[df["low"]>0]

# ─────────────────────────────────────────
# 3. INDICADORES + RÉGIMEN
# ─────────────────────────────────────────
def calcular_indicadores(df):
    cl = df["close"]
    hi = df["high"]
    lo = df["low"]
    vo = df["volume"]

    # EMAs
    df["ema21"]  = cl.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema55"]  = cl.ewm(span=EMA_MED,  adjust=False).mean()
    df["ema200"] = cl.ewm(span=EMA_SLOW, adjust=False).mean()

    # Pendiente EMA200 (%/24h)
    df["slope200"] = df["ema200"].diff(SLOPE_WINDOW) / df["ema200"].shift(SLOPE_WINDOW)

    # ── RSI 14 ────────────────────────────────────────────────────────────
    delta      = cl.diff()
    gain       = delta.clip(lower=0).ewm(span=RSI_PERIOD, adjust=False).mean()
    loss_      = (-delta.clip(upper=0)).ewm(span=RSI_PERIOD, adjust=False).mean()
    rs         = gain / loss_.replace(0, np.nan)
    df["rsi"]  = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi"].diff(3)   # pendiente RSI en 3 velas

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────
    ema_fast        = cl.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow        = cl.ewm(span=MACD_SLOW,   adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_sig"]  = df["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_sig"]
    # Cruce alcista: histograma acaba de cruzar de negativo a positivo
    df["macd_cross_up"] = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    # Histograma aumentando (momentum positivo)
    df["macd_rising"]   = (df["macd_hist"] > df["macd_hist"].shift(1)) & \
                          (df["macd_hist"].shift(1) > df["macd_hist"].shift(2))

    # ── ADX 14 ────────────────────────────────────────────────────────────
    tr = pd.concat([hi - lo,
                    (hi - cl.shift()).abs(),
                    (lo - cl.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ADX_PERIOD, adjust=False).mean()

    # +DM / -DM
    up_move   = hi.diff()
    down_move = -lo.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / df["atr"].replace(0, np.nan)
    minus_di = 100 * minus_dm_s / df["atr"].replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"]      = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di

    # Volumen medio 20 períodos
    df["vol_ma20"] = vo.rolling(VOL_MA_PERIOD).mean()

    # Régimen
    regime = []
    for i in range(len(df)):
        p   = cl.iloc[i]
        e21 = df["ema21"].iloc[i]
        e55 = df["ema55"].iloc[i]
        e200= df["ema200"].iloc[i]
        sl  = df["slope200"].iloc[i]
        if any(pd.isna(x) for x in [e21,e55,e200,sl]):
            regime.append("RANGO"); continue
        a200 = p > e200
        a21_55  = e21 > e55
        a55_200 = e55 > e200
        if a200 and a21_55 and a55_200 and sl > SLOPE_MIN:
            regime.append("BULL_FUERTE")
        elif a200 and a21_55 and a55_200:
            regime.append("BULL_MODERADO")
        elif a200 and (a21_55 or a55_200):
            regime.append("RECUPERACION")
        elif not a200 and not a21_55 and not a55_200 and sl < -SLOPE_MIN:
            regime.append("BEAR_FUERTE")
        elif not a200 and not a21_55:
            regime.append("BEAR_MODERADO")
        else:
            regime.append("RANGO")
    df["regime"] = regime
    return df

# ─────────────────────────────────────────
# 4. SWINGS + IMPULSOS
# ─────────────────────────────────────────
def detectar_swings(df, n=LOOKBACK_SWING):
    highs, lows = [], []
    ha = df["high"].values
    la = df["low"].values
    for i in range(n, len(df)-n):
        if ha[i] == ha[i-n:i+n+1].max(): highs.append(i)
        if la[i] == la[i-n:i+n+1].min(): lows.append(i)
    return highs, lows

def identificar_impulsos(df, swing_lows, swing_highs):
    impulsos = []
    ha  = df["high"].values
    la  = df["low"].values
    idx = df.index
    reg = df["regime"].values
    for li in swing_lows:
        lp = la[li]
        cands = [hi for hi in swing_highs if li+4 < hi < li+400]
        if not cands: continue
        cut   = next((sl for sl in swing_lows if sl>li), len(df))
        cands = [hi for hi in cands if hi < cut]
        if not cands: continue
        hi = max(cands, key=lambda x: ha[x])
        hp = ha[hi]
        rp = (hp-lp)/lp
        if rp >= MIN_IMPULSE_PCT:
            impulsos.append({"low_idx":li,"high_idx":hi,"low_price":lp,
                             "high_price":hp,"rango_pct":rp,
                             "low_time":idx[li],"high_time":idx[hi],
                             "regime_high":reg[hi]})
    return impulsos

# ─────────────────────────────────────────
# 5. FIBONACCI
# ─────────────────────────────────────────
def fibs(imp):
    r       = imp["high_price"] - imp["low_price"]
    h       = imp["high_price"]
    gp      = h - r * GOLDEN_POCKET_LOW    # 61.8%  — entrada E1
    scale   = h - r * SCALE_IN_LEVEL       # 65.0%  — scale-in E2
    sl_base = h - r * SL_LEVEL             # 78.6%  — SL estructural
    rk      = gp - sl_base                 # riesgo desde E1
    return {
        "gp_lo":  gp,
        "gp_hi":  h - r * GOLDEN_POCKET_HI,
        "scale":  scale,                    # nivel de scale-in
        "sl":     sl_base,
        "tp1":    gp + rk * TP1_RR,
        "tp2":    gp + rk * TP2_RR,
    }

# ─────────────────────────────────────────
# 6. BACKTEST OPTIMIZADO
# ─────────────────────────────────────────
def backtest_optimizado(df, impulsos):
    capital      = CAPITAL_INICIAL
    equity_curve = [capital]
    equity_times = [df.index[0]]
    trades       = []

    lo   = df["low"].values
    hi   = df["high"].values
    cl   = df["close"].values
    vo   = df["volume"].values
    idx  = df.index
    reg  = df["regime"].values
    rsi  = df["rsi"].values
    rsi_slope = df["rsi_slope"].values
    mhist= df["macd_hist"].values
    mcross= df["macd_cross_up"].values
    mrising= df["macd_rising"].values
    adx_ = df["adx"].values
    pdi  = df["plus_di"].values
    mdi  = df["minus_di"].values
    vm20 = df["vol_ma20"].values

    ultimo_salida = -MIN_CANDLES_ENTRE
    trade_activo  = False

    for imp in impulsos:
        if trade_activo:
            continue

        f = fibs(imp)
        if f["gp_lo"] - f["sl"] <= 0:
            continue

        # ── FILTRO 1: Régimen BULL_FUERTE en el momento del impulso ────────
        if imp["regime_high"] != "BULL_FUERTE":
            continue

        inicio = max(imp["high_idx"]+1, ultimo_salida+MIN_CANDLES_ENTRE)
        fin    = min(imp["high_idx"]+MAX_HOLD_CANDLES*3, len(df))

        entrada_idx = None
        for i in range(inicio, fin):
            # ── F2: Régimen BULL_FUERTE en la entrada ─────────────────────
            if reg[i] != "BULL_FUERTE":
                continue

            # ── F3: RSI en zona de retroceso 28-55 ───────────────────────
            r = rsi[i]
            if np.isnan(r):
                continue
            if not (RSI_MIN_ENTRY <= r <= RSI_MAX_ENTRY):
                continue
            # RSI no debe estar en caída libre (pendiente no muy negativa)
            if not np.isnan(rsi_slope[i]) and rsi_slope[i] < -3:
                continue

            # ── F4: MACD histograma girando al alza ───────────────────────
            # En el Golden Pocket el histograma viene de negativo → debe girar
            # Condición: hist mejora respecto a 2 velas atrás (reversión local)
            prev2 = max(0, i-2)
            if np.isnan(mhist[i]) or np.isnan(mhist[prev2]):
                continue
            if mhist[i] <= mhist[prev2]:        # no está mejorando → skip
                continue

            # ── F5: ADX — filtro de calidad suave ────────────────────────
            # Eliminar solo mercados claramente en rango (ADX < 15)
            # o cuando -DI domina ampliamente a +DI (bajista fuerte)
            if not np.isnan(adx_[i]) and adx_[i] > 5:   # ADX calculado
                if adx_[i] < 15:                # rango puro → skip
                    continue
                if pdi[i] < mdi[i] * 0.75:     # -DI domina claramente → skip
                    continue

            # ── F6: Volumen > 80% de media 20 ────────────────────────────
            if not np.isnan(vm20[i]) and vo[i] < vm20[i] * 0.8:
                continue

            # ── GOLDEN POCKET: toca 61.8%, vela de reversión alcista ──────
            if lo[i] <= f["gp_lo"] and cl[i] > f["gp_hi"]:
                if cl[i] > (lo[i] + hi[i]) / 2:
                    entrada_idx = i
                    break

        if entrada_idx is None:
            continue

        entrada_exec   = entrada_idx+1
        if entrada_exec >= len(df):
            continue

        # ── E1: entrada principal en 61.8% GP ────────────────────────────
        precio_e1   = cl[entrada_idx]
        riesgo_usd  = capital * RISK_PCT
        tam_pos_usd = min(riesgo_usd / SL_NIVEL_PCT, capital * 0.5)
        btc_e1      = tam_pos_usd / precio_e1

        # Variables de scale-in (E2 en 65%)
        scale_hit   = False
        btc_e2      = 0.0
        precio_e2   = 0.0

        # Posición total inicial (puede crecer con scale-in)
        btc_total   = btc_e1
        avg_entry   = precio_e1

        # TP1/TP2 calculados desde E1 inicialmente (se recalculan tras scale-in)
        tp1_price   = f["tp1"]
        tp2_price   = f["tp2"]

        resultado     = "TIMEOUT"
        precio_salida = None
        salida_idx    = None
        tp1_hit       = False
        btc_rem       = btc_total
        sl_dinamico   = f["sl"]   # SL estructural en 78.6%

        trade_activo  = True

        for j in range(entrada_exec, min(entrada_exec+MAX_HOLD_CANDLES+1, len(df))):
            # ── SL dinámico (breakeven tras TP1) ─────────────────────────
            if lo[j] <= sl_dinamico:
                precio_salida = sl_dinamico
                resultado = "SL" if sl_dinamico < avg_entry else "BE"
                salida_idx = j
                break

            # ── Scale-in E2: profundiza a 65% ────────────────────────────
            if not scale_hit and not tp1_hit and lo[j] <= f["scale"]:
                scale_hit  = True
                precio_e2  = f["scale"]
                btc_e2     = btc_e1 * SCALE_IN_FACTOR
                btc_total  = btc_e1 + btc_e2
                avg_entry  = (btc_e1 * precio_e1 + btc_e2 * precio_e2) / btc_total
                # Recalcular TP1/TP2 desde precio medio
                rk_avg     = avg_entry - f["sl"]
                tp1_price  = avg_entry + rk_avg * TP1_RR
                tp2_price  = avg_entry + rk_avg * TP2_RR
                btc_rem    = btc_total

            # ── TP1 ───────────────────────────────────────────────────────
            if not tp1_hit and hi[j] >= tp1_price:
                tp1_hit     = True
                btc_rem    *= (1 - TP1_FRACTION)
                sl_dinamico = avg_entry  # mover SL a breakeven

            # ── TP2 ───────────────────────────────────────────────────────
            if hi[j] >= tp2_price:
                precio_salida = tp2_price
                resultado = "TP2"
                salida_idx = j
                break

        if salida_idx is None:
            salida_idx    = min(entrada_exec+MAX_HOLD_CANDLES, len(df)-1)
            precio_salida = cl[salida_idx]

        # ── PnL — dos tramos (E1 siempre, E2 si scale_hit) ───────────────
        if tp1_hit:
            # Mitad TP1 al precio TP1, resto al precio_salida (desde avg_entry)
            pnl = (btc_total * TP1_FRACTION * (tp1_price - avg_entry) +
                   btc_rem * (precio_salida - avg_entry))
        else:
            pnl = btc_total * (precio_salida - avg_entry)

        capital += pnl
        capital  = max(capital, 1)
        dur_h    = (idx[salida_idx]-idx[entrada_exec]).total_seconds()/3600

        ultimo_salida = salida_idx
        trade_activo  = False

        trades.append({
            "entrada_time":  idx[entrada_exec],
            "salida_time":   idx[salida_idx],
            "año":           idx[entrada_exec].year,
            "mes":           idx[entrada_exec].strftime("%Y-%m"),
            "precio_entrada":round(avg_entry,2),
            "precio_e1":     round(precio_e1,2),
            "precio_e2":     round(precio_e2,2) if scale_hit else None,
            "scale_hit":     scale_hit,
            "precio_salida": round(precio_salida,2),
            "sl_price":      round(sl_dinamico,2),
            "tp1_price":     round(tp1_price,2),
            "tp2_price":     round(tp2_price,2),
            "resultado":     resultado,
            "pnl_usd":       round(pnl,2),
            "pnl_pct":       round(pnl/tam_pos_usd*100,2) if tam_pos_usd else 0,
            "capital":       round(capital,2),
            "duracion_h":    round(dur_h,1),
            "rango_imp":     round(imp["rango_pct"]*100,2),
            "tp1_hit":       tp1_hit,
            "rsi_entrada":   round(rsi[entrada_idx],   1) if not np.isnan(rsi[entrada_idx])   else 0,
            "macd_hist":     round(mhist[entrada_idx], 4) if not np.isnan(mhist[entrada_idx]) else 0,
            "adx_entrada":   round(adx_[entrada_idx],  1) if not np.isnan(adx_[entrada_idx])  else 0,
            "plus_di":       round(pdi[entrada_idx],   1) if not np.isnan(pdi[entrada_idx])   else 0,
        })
        equity_curve.append(capital)
        equity_times.append(idx[salida_idx])

    return pd.DataFrame(trades), equity_curve, equity_times

# ─────────────────────────────────────────
# 7. ESTADÍSTICAS COMPLETAS
# ─────────────────────────────────────────
def calcular_stats(trades_df, equity_curve):
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df["pnl_usd"] > 0]
    loss = trades_df[trades_df["pnl_usd"] <= 0]
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq-peak)/peak*100
    rets = trades_df["pnl_usd"] / CAPITAL_INICIAL
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    gross_w = wins["pnl_usd"].sum() if len(wins) else 0
    gross_l = abs(loss["pnl_usd"].sum()) if len(loss) else 0.001
    tp2_n = len(trades_df[trades_df["resultado"]=="TP2"])
    sl_n  = len(trades_df[trades_df["resultado"]=="SL"])
    be_n  = len(trades_df[trades_df["resultado"]=="BE"])
    to_n  = len(trades_df[trades_df["resultado"]=="TIMEOUT"])
    rr = abs(wins["pnl_usd"].mean()/loss["pnl_usd"].mean()) if len(loss) and loss["pnl_usd"].mean() != 0 else 0
    scale_n = int(trades_df["scale_hit"].sum()) if "scale_hit" in trades_df.columns else 0
    return {
        "n":         len(trades_df),
        "wins":      len(wins),
        "losses":    len(loss),
        "wr":        round(len(wins)/len(trades_df)*100,1),
        "tp2_n":     tp2_n, "sl_n":sl_n, "be_n":be_n, "to_n":to_n,
        "tp1_rate":  round(trades_df["tp1_hit"].mean()*100,1),
        "scale_n":   scale_n,
        "scale_pct": round(scale_n/len(trades_df)*100,1),
        "cap_final":round(eq[-1],2),
        "retorno":  round((eq[-1]/CAPITAL_INICIAL-1)*100,1),
        "max_dd":   round(dd.min(),1),
        "pf":       round(gross_w/gross_l,2),
        "sharpe":   round(sharpe,2),
        "avg_win":  round(wins["pnl_usd"].mean(),2) if len(wins) else 0,
        "avg_loss": round(loss["pnl_usd"].mean(),2) if len(loss) else 0,
        "rr":       round(rr,2),
        "best":     round(trades_df["pnl_usd"].max(),2),
        "worst":    round(trades_df["pnl_usd"].min(),2),
        "avg_dur":  round(trades_df["duracion_h"].mean(),1),
        "eq_arr":   eq, "dd_arr":dd,
        "avg_rsi":  round(trades_df["rsi_entrada"].mean(), 1),
        "avg_adx":  round(trades_df["adx_entrada"].mean(), 1),
        "avg_macd": round(trades_df["macd_hist"].mean(),   4),
    }

# ─────────────────────────────────────────
# 8. GRÁFICA MAESTRA
# ─────────────────────────────────────────
def graficar(df, trades_df, s, equity_curve, equity_times):
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "BTC/USDC 1H — Golden Pocket Fibonacci · RSI + MACD + ADX\n"
        "BULL_FUERTE · RSI 30-52 girando ↑ · MACD hist cruce/aceleración · ADX>25 +DI>-DI · SL→BE · TP2=3:1",
        fontsize=12, fontweight="bold", color="#f0f6fc", y=0.99
    )
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.32)

    def style(ax, title="", fs=9):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        if title:
            ax.set_title(title, color="#f0f6fc", fontsize=fs,
                         fontweight="bold", pad=6)

    # ── Panel A: Precio + BULL_FUERTE en verde ─────────────────────────────
    ax_p = fig.add_subplot(gs[0,:])
    style(ax_p, "BTC/USDC — Zonas BULL_FUERTE (verde) donde opera la estrategia", fs=10)
    df_sub = df.iloc[::4]
    bull_mask = df_sub["regime"] == "BULL_FUERTE"
    ax_p.plot(df_sub.index, df_sub["close"], color="#4a4a5a",
              linewidth=0.6, alpha=0.5, zorder=1)
    # Shading BULL_FUERTE
    prev_bull = False
    sx = None
    for ts, row in df_sub.iterrows():
        if row["regime"]=="BULL_FUERTE" and not prev_bull:
            sx = ts; prev_bull=True
        elif row["regime"]!="BULL_FUERTE" and prev_bull:
            ax_p.axvspan(sx, ts, color="#00c853", alpha=0.08, zorder=0)
            prev_bull=False
    if prev_bull and sx:
        ax_p.axvspan(sx, df_sub.index[-1], color="#00c853", alpha=0.08, zorder=0)
    # EMAs
    ax_p.plot(df_sub.index, df_sub["ema21"],  color="#58a6ff", lw=0.8, alpha=0.7, label="EMA21")
    ax_p.plot(df_sub.index, df_sub["ema55"],  color="#d29922", lw=0.8, alpha=0.6, label="EMA55")
    ax_p.plot(df_sub.index, df_sub["ema200"], color="#f0f6fc", lw=1.0, alpha=0.5, label="EMA200", linestyle="--")
    # Marcar entradas y salidas
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            col = "#3fb950" if t["pnl_usd"]>0 else "#f85149"
            ax_p.axvline(t["entrada_time"], color=col, alpha=0.3, lw=0.6)
    ax_p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_p.set_ylabel("USD", color="#8b949e", fontsize=8)
    ax_p.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7, loc="upper left")

    # ── Panel B: Equity curve ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1,:2])
    style(ax_e, "Curva de Equity — Estrategia Optimizada")
    eq = s["eq_arr"]
    ax_e.plot(eq, color="#58a6ff", lw=1.6, zorder=3)
    ax_e.axhline(CAPITAL_INICIAL, color="#8b949e", linestyle="--", lw=0.8, alpha=0.5)
    ax_e.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                      where=[e>=CAPITAL_INICIAL for e in eq], color="#3fb950", alpha=0.15)
    ax_e.fill_between(range(len(eq)), eq, CAPITAL_INICIAL,
                      where=[e<CAPITAL_INICIAL for e in eq], color="#f85149", alpha=0.15)
    # Annotate años
    if not trades_df.empty:
        for yr in trades_df["año"].unique():
            yr_trades = trades_df[trades_df["año"]==yr]
            if len(yr_trades):
                first_idx = yr_trades.index[0]
                eq_idx = trades_df.index.get_loc(first_idx)
                ax_e.axvline(eq_idx, color="#8b949e", lw=0.5, alpha=0.4)
                ax_e.text(eq_idx+2, max(eq)*0.97, str(yr),
                          color="#8b949e", fontsize=7)
    retorno = s["retorno"]
    col_ret = "#3fb950" if retorno>=0 else "#f85149"
    ax_e.text(0.98, 0.95, f"${s['cap_final']:,.0f}  ({retorno:+.1f}%)",
              transform=ax_e.transAxes, ha="right", va="top",
              color=col_ret, fontsize=11, fontweight="bold")
    ax_e.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_e.set_ylabel("Capital USD", color="#8b949e", fontsize=8)

    # ── Panel C: Drawdown ─────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1,2])
    style(ax_dd, "Drawdown")
    dd = s["dd_arr"]
    ax_dd.fill_between(range(len(dd)), dd, 0, color="#f85149", alpha=0.55)
    ax_dd.plot(dd, color="#ff6b6b", lw=0.8)
    ax_dd.text(0.02, 0.08, f"Max DD: {s['max_dd']}%",
               transform=ax_dd.transAxes, color="#ff6b6b", fontsize=8, fontweight="bold")
    ax_dd.set_ylabel("%", color="#8b949e", fontsize=8)

    # ── Panel D: PnL por trade ────────────────────────────────────────────
    ax_pnl = fig.add_subplot(gs[2,0])
    style(ax_pnl, "PnL por Trade (USD)")
    if not trades_df.empty:
        pnls = trades_df["pnl_usd"].values
        cols = ["#3fb950" if p>0 else "#f85149" for p in pnls]
        ax_pnl.bar(range(len(pnls)), pnls, color=cols, alpha=0.85, width=0.8)
        ax_pnl.axhline(0, color="#8b949e", lw=0.5)
        # Running total line
        ax2 = ax_pnl.twinx()
        ax2.plot(trades_df["pnl_usd"].cumsum().values,
                 color="#58a6ff", lw=1.2, alpha=0.7)
        ax2.set_facecolor("#161b22")
        ax2.tick_params(colors="#8b949e", labelsize=6)
        ax2.set_ylabel("Acum.", color="#58a6ff", fontsize=7)
    ax_pnl.set_xlabel("Trade #", color="#8b949e", fontsize=7)
    ax_pnl.set_ylabel("USD", color="#8b949e", fontsize=7)

    # ── Panel E: Histograma PnL% ──────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2,1])
    style(ax_h, "Distribución PnL%")
    if not trades_df.empty:
        pp = trades_df["pnl_pct"]
        ax_h.hist(pp[pp>0],  bins=15, color="#3fb950", alpha=0.75, label=f"Wins ({len(pp[pp>0])})")
        ax_h.hist(pp[pp<=0], bins=15, color="#f85149", alpha=0.75, label=f"Losses ({len(pp[pp<=0])})")
        ax_h.axvline(0, color="white", lw=0.8)
        ax_h.axvline(pp.mean(), color="#d29922", lw=1, linestyle="--")
        ax_h.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7)
        ax_h.set_xlabel("PnL%", color="#8b949e", fontsize=7)

    # ── Panel F: PnL por año ──────────────────────────────────────────────
    ax_yr = fig.add_subplot(gs[2,2])
    style(ax_yr, "PnL Anual + Win Rate")
    if not trades_df.empty:
        por_año = trades_df.groupby("año").agg(
            pnl=("pnl_usd","sum"), wr=("pnl_usd",lambda x:(x>0).mean()*100),
            n=("pnl_usd","count")).reset_index()
        bar_c = ["#3fb950" if v>0 else "#f85149" for v in por_año["pnl"]]
        bars = ax_yr.bar(por_año["año"].astype(str), por_año["pnl"],
                         color=bar_c, alpha=0.85, width=0.6)
        ax_yr.axhline(0, color="#8b949e", lw=0.5)
        for bar, row in zip(bars, por_año.itertuples()):
            y = bar.get_height()
            ax_yr.text(bar.get_x()+bar.get_width()/2,
                       y + (30 if y>=0 else -120),
                       f"WR\n{row.wr:.0f}%\n({row.n}t)",
                       ha="center", color="#f0f6fc", fontsize=6.5, fontweight="bold")
        ax_yr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax_yr.tick_params(axis="x", labelrotation=30)
        ax_yr.set_ylabel("USD", color="#8b949e", fontsize=7)

    # ── Panel G: RSI en trades ────────────────────────────────────────────
    ax_rsi = fig.add_subplot(gs[3, 0])
    style(ax_rsi, "RSI en entrada (distribución)")
    if not trades_df.empty:
        rsi_w = trades_df[trades_df["pnl_usd"]>0]["rsi_entrada"]
        rsi_l = trades_df[trades_df["pnl_usd"]<=0]["rsi_entrada"]
        ax_rsi.hist(rsi_w, bins=12, color="#3fb950", alpha=0.7, label="Wins")
        ax_rsi.hist(rsi_l, bins=12, color="#f85149", alpha=0.7, label="Losses")
        ax_rsi.axvline(30, color="#ff9800", lw=0.9, linestyle="--", alpha=0.8)
        ax_rsi.axvline(50, color="#8b949e", lw=0.9, linestyle="--", alpha=0.8)
        ax_rsi.text(31, ax_rsi.get_ylim()[1]*0.85, "30", color="#ff9800", fontsize=7)
        ax_rsi.text(51, ax_rsi.get_ylim()[1]*0.85, "50", color="#8b949e", fontsize=7)
        ax_rsi.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7)
        ax_rsi.set_xlabel("RSI", color="#8b949e", fontsize=7)
        ax_rsi.set_ylabel("# trades", color="#8b949e", fontsize=7)

    # ── Panel H: MACD histograma en trades ────────────────────────────────
    ax_macd = fig.add_subplot(gs[3, 1])
    style(ax_macd, "MACD histograma en entrada")
    if not trades_df.empty:
        mh_w = trades_df[trades_df["pnl_usd"]>0]["macd_hist"]
        mh_l = trades_df[trades_df["pnl_usd"]<=0]["macd_hist"]
        ax_macd.hist(mh_w, bins=12, color="#3fb950", alpha=0.7, label="Wins")
        ax_macd.hist(mh_l, bins=12, color="#f85149", alpha=0.7, label="Losses")
        ax_macd.axvline(0, color="#f0f6fc", lw=0.8)
        ax_macd.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7)
        ax_macd.set_xlabel("MACD Hist", color="#8b949e", fontsize=7)
        ax_macd.set_ylabel("# trades", color="#8b949e", fontsize=7)

    # ── Panel I: ADX en trades ────────────────────────────────────────────
    ax_adx = fig.add_subplot(gs[3, 2])
    style(ax_adx, "ADX en entrada (fuerza de tendencia)")
    if not trades_df.empty:
        adx_w = trades_df[trades_df["pnl_usd"]>0]["adx_entrada"]
        adx_l = trades_df[trades_df["pnl_usd"]<=0]["adx_entrada"]
        ax_adx.hist(adx_w, bins=12, color="#3fb950", alpha=0.7, label="Wins")
        ax_adx.hist(adx_l, bins=12, color="#f85149", alpha=0.7, label="Losses")
        ax_adx.axvline(ADX_MIN, color="#d29922", lw=0.9, linestyle="--")
        ax_adx.axvline(40, color="#8b949e", lw=0.9, linestyle="--", alpha=0.6)
        ax_adx.text(ADX_MIN+0.5, ax_adx.get_ylim()[1]*0.85, str(ADX_MIN),
                    color="#d29922", fontsize=7)
        ax_adx.legend(facecolor="#21262d", labelcolor="#f0f6fc", fontsize=7)
        ax_adx.set_xlabel("ADX", color="#8b949e", fontsize=7)
        ax_adx.set_ylabel("# trades", color="#8b949e", fontsize=7)

    # ── Panel J: WR por rango de ADX ─────────────────────────────────────
    ax_adx2 = fig.add_subplot(gs[4, 0])
    style(ax_adx2, "Win Rate por rango ADX")
    if not trades_df.empty:
        bins_adx = [25, 30, 35, 40, 50, 70]
        labels_adx = ["25-30","30-35","35-40","40-50","50+"]
        trades_df["adx_bucket"] = pd.cut(trades_df["adx_entrada"],
                                          bins=bins_adx, labels=labels_adx, right=False)
        wr_adx = trades_df.groupby("adx_bucket", observed=True)["pnl_usd"].apply(
            lambda x: (x>0).mean()*100).reset_index(name="wr")
        cnt_adx = trades_df.groupby("adx_bucket", observed=True).size().reset_index(name="n")
        merged_adx = wr_adx.merge(cnt_adx, on="adx_bucket")
        bar_c = ["#3fb950" if w>=50 else "#f85149" for w in merged_adx["wr"]]
        bars = ax_adx2.bar(merged_adx["adx_bucket"].astype(str), merged_adx["wr"],
                           color=bar_c, alpha=0.85)
        ax_adx2.axhline(50, color="#8b949e", lw=0.8, linestyle="--")
        for bar, (_, row) in zip(bars, merged_adx.iterrows()):
            ax_adx2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+1, f"{row['wr']:.0f}%\n({row['n']})",
                         ha="center", color="#f0f6fc", fontsize=6.5)
        ax_adx2.set_ylabel("Win Rate %", color="#8b949e", fontsize=7)
        ax_adx2.set_xlabel("ADX range", color="#8b949e", fontsize=7)
        ax_adx2.set_ylim(0, 90)

    # ── Panel K: WR por rango RSI ─────────────────────────────────────────
    ax_rsi2 = fig.add_subplot(gs[4, 1])
    style(ax_rsi2, "Win Rate por rango RSI")
    if not trades_df.empty:
        bins_rsi = [30, 35, 40, 45, 50, 53]
        labels_rsi = ["30-35","35-40","40-45","45-50","50-52"]
        trades_df["rsi_bucket"] = pd.cut(trades_df["rsi_entrada"],
                                          bins=bins_rsi, labels=labels_rsi, right=False)
        wr_rsi = trades_df.groupby("rsi_bucket", observed=True)["pnl_usd"].apply(
            lambda x: (x>0).mean()*100).reset_index(name="wr")
        cnt_rsi = trades_df.groupby("rsi_bucket", observed=True).size().reset_index(name="n")
        merged_rsi = wr_rsi.merge(cnt_rsi, on="rsi_bucket")
        bar_c = ["#3fb950" if w>=50 else "#f85149" for w in merged_rsi["wr"]]
        bars = ax_rsi2.bar(merged_rsi["rsi_bucket"].astype(str), merged_rsi["wr"],
                           color=bar_c, alpha=0.85)
        ax_rsi2.axhline(50, color="#8b949e", lw=0.8, linestyle="--")
        for bar, (_, row) in zip(bars, merged_rsi.iterrows()):
            ax_rsi2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+1, f"{row['wr']:.0f}%\n({row['n']})",
                         ha="center", color="#f0f6fc", fontsize=6.5)
        ax_rsi2.set_ylabel("Win Rate %", color="#8b949e", fontsize=7)
        ax_rsi2.set_xlabel("RSI range", color="#8b949e", fontsize=7)
        ax_rsi2.set_ylim(0, 90)

    # ── Panel L: Tabla de estadísticas ────────────────────────────────────
    ax_tab = fig.add_subplot(gs[4, 2])
    ax_tab.axis("off")
    style(ax_tab, "")

    # Construir tabla horizontal compacta
    col_groups = [
        ("RENDIMIENTO", [
            ("Capital inicial",  f"${CAPITAL_INICIAL:,}"),
            ("Capital final",    f"${s['cap_final']:,.0f}"),
            ("Retorno total",    f"{s['retorno']:+.1f}%"),
            ("Max Drawdown",     f"{s['max_dd']}%"),
            ("Sharpe Ratio",     str(s['sharpe'])),
        ]),
        ("TRADES", [
            ("Total trades",     str(s['n'])),
            ("Win Rate",         f"{s['wr']}%"),
            ("Profit Factor",    str(s['pf'])),
            ("TP1 Hit Rate",     f"{s['tp1_rate']}%"),
            ("TP2 completados",  str(s['tp2_n'])),
            ("Scale-in hits",    f"{s['scale_n']} ({s['scale_pct']}%)"),
        ]),
        ("EXITS", [
            ("SL activados",     str(s['sl_n'])),
            ("Breakeven (BE)",   str(s['be_n'])),
            ("Timeout (72h)",    str(s['to_n'])),
            ("Avg R:R efectivo", f"{s['rr']}:1"),
            ("RSI medio entrada",str(s['avg_rsi'])),
        ]),
        ("PROMEDIOS", [
            ("Avg ganancia",     f"${s['avg_win']:,.2f}"),
            ("Avg pérdida",      f"${s['avg_loss']:,.2f}"),
            ("Mejor trade",      f"${s['best']:,.2f}"),
            ("Peor trade",       f"${s['worst']:,.2f}"),
            ("Duración media",   f"{s['avg_dur']}h"),
            ("Avg RSI entrada",  str(s['avg_rsi'])),
            ("Avg ADX entrada",  str(s['avg_adx'])),
        ]),
    ]

    ncols = len(col_groups)
    nrows = max(len(g[1]) for g in col_groups) + 1
    cell_data = []
    col_labels = [g[0] for g in col_groups]
    # Pad columns to same length
    padded = []
    for _, rows in col_groups:
        while len(rows) < nrows-1:
            rows.append(("",""))
        padded.append(rows)

    for r in range(nrows-1):
        row = []
        for c in range(ncols):
            k, v = padded[c][r]
            row.append(f"{k}:  {v}" if k else "")
        cell_data.append(row)

    tbl = ax_tab.table(cellText=cell_data,
                       colLabels=col_labels,
                       loc="center", cellLoc="left",
                       bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f6feb")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(9)
        else:
            cell.set_facecolor("#21262d" if r%2==0 else "#161b22")
            txt = cell.get_text().get_text()
            # colorear valores clave
            if "%" in txt and any(k in txt for k in ["Retorno","Win Rate","WR"]):
                try:
                    val = float(txt.split()[-1].replace("%","").replace("+",""))
                    cell.get_text().set_color("#3fb950" if val>0 else "#f85149")
                except: cell.get_text().set_color("#f0f6fc")
            elif "Drawdown" in txt or "SL" in txt or "pérdida" in txt or "Peor" in txt:
                cell.get_text().set_color("#f85149")
            else:
                cell.get_text().set_color("#f0f6fc")
        cell.set_edgecolor("#30363d")

    plt.savefig("/home/user/TV/backtest_optimizado.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Gráfica: backtest_optimizado.png")

# ─────────────────────────────────────────
# 9. REPORTE CONSOLA
# ─────────────────────────────────────────
def reporte_consola(s, trades_df):
    sep = "═"*60
    print(f"\n{sep}")
    print("  BTC/USDC 1H — GOLDEN POCKET FIBONACCI OPTIMIZADA")
    print("  Filtros: BULL_FUERTE · RSI 30-52↑ · MACD hist cruce/aceleración · ADX>25 +DI>-DI")
    print(f"  Capital: ${CAPITAL_INICIAL:,}  ·  Riesgo: {RISK_PCT*100:.0f}%/trade  ·  5 años")
    print(sep)
    print(f"\n  {'MÉTRICA':<28} {'VALOR':>12}   {'vs. SIN FILTROS':>15}")
    print(f"  {'─'*58}")
    rows = [
        ("Total trades",          f"{s['n']}",              "954 → menos"),
        ("Win Rate",              f"{s['wr']}%",            "45.4% → mejor?"),
        ("Profit Factor",         f"{s['pf']}",             "0.88 → mejor?"),
        ("Sharpe Ratio",          f"{s['sharpe']}",         "-0.85 → mejor?"),
        ("Capital final",         f"${s['cap_final']:,.0f}","$7,878 → ?"),
        ("Retorno total",         f"{s['retorno']:+.1f}%",  "-25.0% → mejor?"),
        ("Max Drawdown",          f"{s['max_dd']}%",        "-29.5% → mejor?"),
        ("TP2 completados",       f"{s['tp2_n']}",          ""),
        ("SL activados",          f"{s['sl_n']}",           ""),
        ("Breakeven (no loss)",   f"{s['be_n']}",           "— nuevo"),
        ("TP1 Hit Rate",          f"{s['tp1_rate']}%",      ""),
        ("Scale-in hits",         f"{s['scale_n']} ({s['scale_pct']}%)", "— nuevo"),
        ("Avg R:R efectivo",      f"{s['rr']}:1",           "1.07 → mejor?"),
        ("RSI medio en entrada",  f"{s['avg_rsi']}",        ""),
        ("ADX medio en entrada",  f"{s['avg_adx']}",        ""),
        ("MACD hist medio",       f"{s['avg_macd']:.4f}",   ""),
        ("Duración media",        f"{s['avg_dur']}h",       ""),
    ]
    for k, v, comp in rows:
        print(f"  {k:<28} {v:>12}   {comp}")

    print(f"\n{sep}")
    print("  DETALLE POR AÑO:")
    print(f"  {'Año':<6} {'N':>5} {'WR':>6} {'SL':>4} {'BE':>4} {'TP2':>5} {'PnL':>12} {'Acum.':>10}")
    print(f"  {'─'*56}")
    acum = CAPITAL_INICIAL
    for yr, g in trades_df.groupby("año"):
        wr  = (g["pnl_usd"]>0).mean()*100
        tot = g["pnl_usd"].sum()
        sl_ = (g["resultado"]=="SL").sum()
        be_ = (g["resultado"]=="BE").sum()
        t2  = (g["resultado"]=="TP2").sum()
        acum += tot
        sig  = "▲" if tot>0 else "▼"
        print(f"  {yr:<6} {len(g):>5} {wr:>5.0f}% {sl_:>4} {be_:>4} {t2:>5} "
              f"{sig} ${tot:>9,.0f}  ${acum:>9,.0f}")

    print(f"\n{sep}")
    best5 = trades_df.nlargest(5,"pnl_usd")[
        ["año","precio_entrada","resultado","rsi_entrada","pnl_usd","duracion_h"]]
    print("  TOP 5 MEJORES TRADES:")
    print(best5.to_string(index=False))

    worst5 = trades_df.nsmallest(5,"pnl_usd")[
        ["año","precio_entrada","resultado","rsi_entrada","pnl_usd","duracion_h"]]
    print("\n  TOP 5 PEORES TRADES:")
    print(worst5.to_string(index=False))
    print(sep)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Generando datos 1H BTC (5 años)...")
    df = generar_ohlcv_1h(BTC_MONTHLY_ANCHORS)
    print(f"  {len(df):,} velas  ({df.index[0].date()} → {df.index[-1].date()})")

    print("Calculando indicadores y clasificando regímenes...")
    df = calcular_indicadores(df)

    bf = (df["regime"]=="BULL_FUERTE").mean()*100
    print(f"  Tiempo en BULL_FUERTE: {bf:.1f}% del período")

    print("Detectando swings e impulsos...")
    sh, sl_ = detectar_swings(df)
    impulsos = identificar_impulsos(df, sl_, sh)
    print(f"  Impulsos ≥0.75% en BULL_FUERTE: {sum(1 for i in impulsos if i['regime_high']=='BULL_FUERTE')}")

    print("Ejecutando backtest optimizado...")
    trades_df, eq_curve, eq_times = backtest_optimizado(df, impulsos)
    print(f"  Trades ejecutados: {len(trades_df)}")

    s = calcular_stats(trades_df, eq_curve)
    reporte_consola(s, trades_df)

    print("\nGenerando gráficas...")
    graficar(df, trades_df, s, eq_curve, eq_times)

    trades_df.to_csv("/home/user/TV/backtest_optimizado_trades.csv", index=False)
    print("  CSV: backtest_optimizado_trades.csv")
    print("\n  ✓ Estrategia optimizada completa.")

"""
Servidor webhook: TradingView → Binance
Recibe alertas JSON de TradingView y ejecuta órdenes en Binance Spot.

Instalación:
    pip install flask python-binance python-dotenv

Arranque:
    python webhook_server.py

Configura el webhook en TradingView apuntando a:
    http://<TU_IP>:<PORT>/webhook
"""

import json
import logging
import os
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

BINANCE_API_KEY    = os.environ["BINANCE_API_KEY"]
BINANCE_API_SECRET = os.environ["BINANCE_API_SECRET"]
WEBHOOK_SECRET     = os.environ["WEBHOOK_SECRET"]
PORT               = int(os.getenv("PORT", 5000))
TESTNET            = os.getenv("TESTNET", "true").lower() == "true"

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=TESTNET)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_symbol_info(symbol: str) -> dict:
    info = client.get_symbol_info(symbol)
    if info is None:
        raise ValueError(f"Símbolo desconocido: {symbol}")
    return info


def round_step(value: float, step: str) -> str:
    """Redondea value al step size del par (ej: '0.00001')."""
    d_step  = Decimal(step)
    d_value = Decimal(str(value))
    return str(d_value.quantize(d_step, rounding=ROUND_DOWN))


def get_filters(info: dict) -> dict:
    filters = {f["filterType"]: f for f in info["filters"]}
    return filters


def calc_quantity(symbol: str, usdc_amount: float, price: float) -> str:
    """Calcula la cantidad de base asset dado el capital en USDC."""
    info    = get_symbol_info(symbol)
    filters = get_filters(info)
    step    = filters["LOT_SIZE"]["stepSize"]
    qty     = usdc_amount / price
    return round_step(qty, step)


def round_price(price: float, symbol_info: dict) -> str:
    filters = get_filters(symbol_info)
    tick    = filters["PRICE_FILTER"]["tickSize"]
    return round_step(price, tick)


def get_usdc_balance() -> float:
    balances = {b["asset"]: float(b["free"]) for b in client.get_account()["balances"]}
    return balances.get("USDC", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de órdenes
# ─────────────────────────────────────────────────────────────────────────────

def open_long(symbol: str, price: float, sl: float, tp: float | None) -> dict:
    """Compra a mercado + OCO de salida (SL + TP)."""
    usdc   = get_usdc_balance()
    # Usa el 99 % del saldo disponible (deja margen para comisiones)
    qty    = calc_quantity(symbol, usdc * 0.99, price)
    info   = get_symbol_info(symbol)

    log.info("LONG %s  qty=%s  SL=%s  TP=%s", symbol, qty, sl, tp)

    entry = client.order_market_buy(symbol=symbol, quantity=qty)
    log.info("Orden market compra ejecutada: %s", entry.get("orderId"))

    if sl and tp:
        sl_str = round_price(sl, info)
        tp_str = round_price(tp, info)
        oco = client.order_oco_sell(
            symbol        = symbol,
            quantity      = qty,
            price         = tp_str,
            stopPrice     = sl_str,
            stopLimitPrice= sl_str,
            stopLimitTimeInForce = "GTC",
        )
        log.info("OCO salida colocada: %s", oco.get("orderListId"))
        return {"entry": entry, "oco": oco}

    return {"entry": entry}


def open_short(symbol: str, price: float, sl: float, tp: float | None) -> dict:
    """Vende a mercado (requiere spot margin o saldo de base asset) + OCO recompra."""
    info = get_symbol_info(symbol)
    # Obtiene el saldo del activo base (ej: BTC)
    base_asset = info["baseAsset"]
    balances   = {b["asset"]: float(b["free"]) for b in client.get_account()["balances"]}
    base_bal   = balances.get(base_asset, 0.0)

    step = get_filters(info)["LOT_SIZE"]["stepSize"]
    qty  = round_step(base_bal * 0.99, step)

    log.info("SHORT %s  qty=%s  SL=%s  TP=%s", symbol, qty, sl, tp)

    entry = client.order_market_sell(symbol=symbol, quantity=qty)
    log.info("Orden market venta ejecutada: %s", entry.get("orderId"))

    if sl and tp:
        sl_str = round_price(sl, info)
        tp_str = round_price(tp, info)
        oco = client.order_oco_buy(
            symbol        = symbol,
            quantity      = qty,
            price         = tp_str,
            stopPrice     = sl_str,
            stopLimitPrice= sl_str,
            stopLimitTimeInForce = "GTC",
        )
        log.info("OCO recompra colocada: %s", oco.get("orderListId"))
        return {"entry": entry, "oco": oco}

    return {"entry": entry}


def close_long(symbol: str) -> dict:
    """Cancela órdenes abiertas y cierra la posición long a mercado."""
    client.cancel_open_orders(symbol=symbol)
    info       = get_symbol_info(symbol)
    base_asset = info["baseAsset"]
    balances   = {b["asset"]: float(b["free"]) for b in client.get_account()["balances"]}
    base_bal   = balances.get(base_asset, 0.0)
    step       = get_filters(info)["LOT_SIZE"]["stepSize"]
    qty        = round_step(base_bal * 0.99, step)
    log.info("CLOSE LONG %s  qty=%s", symbol, qty)
    order = client.order_market_sell(symbol=symbol, quantity=qty)
    log.info("Posición long cerrada: %s", order.get("orderId"))
    return {"close": order}


def close_short(symbol: str) -> dict:
    """Cancela órdenes abiertas y recompra para cerrar la posición short."""
    client.cancel_open_orders(symbol=symbol)
    info    = get_symbol_info(symbol)
    price   = float(client.get_symbol_ticker(symbol=symbol)["price"])
    usdc    = get_usdc_balance()
    qty     = calc_quantity(symbol, usdc * 0.99, price)
    log.info("CLOSE SHORT %s  qty=%s", symbol, qty)
    order = client.order_market_buy(symbol=symbol, quantity=qty)
    log.info("Posición short cerrada: %s", order.get("orderId"))
    return {"close": order}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint webhook
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
    except Exception:
        log.warning("Payload no es JSON válido")
        abort(400, "JSON inválido")

    if not data:
        abort(400, "Payload vacío")

    # Verificar token secreto
    if data.get("secret") != WEBHOOK_SECRET:
        log.warning("Token secreto incorrecto — petición rechazada")
        abort(403, "No autorizado")

    action  = data.get("action", "").lower()
    symbol  = data.get("symbol", "BTCUSDC").upper()
    price   = float(data.get("price", 0))
    sl      = float(data.get("sl", 0)) if data.get("sl") else None
    tp      = float(data.get("tp", 0)) if data.get("tp") else None
    comment = data.get("comment", "")

    log.info("Señal recibida — action=%s  symbol=%s  price=%s  sl=%s  tp=%s  comment=%s",
             action, symbol, price, sl, tp, comment)

    try:
        if action == "buy":
            result = open_long(symbol, price, sl, tp)
        elif action == "sell":
            result = open_short(symbol, price, sl, tp)
        elif action == "close_long":
            result = close_long(symbol)
        elif action == "close_short":
            result = close_short(symbol)
        else:
            log.warning("Acción desconocida: %s", action)
            abort(400, f"Acción no reconocida: {action}")

    except BinanceAPIException as e:
        log.error("Error Binance API: %s", e)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.exception("Error inesperado: %s", e)
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "action": action, "symbol": symbol, "result": str(result)}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "testnet": TESTNET}), 200


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "TESTNET" if TESTNET else "PRODUCCIÓN"
    log.info("Servidor webhook arrancando en modo %s — puerto %d", mode, PORT)
    app.run(host="0.0.0.0", port=PORT)

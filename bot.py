
import os
import logging
import asyncio

import requests
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================
# ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ
# =========================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GATE_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAIR = "BTC_USDT"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================
# GATE API
# =========================

def get_candles(interval="15m", limit=100):
    response = requests.get(
        GATE_URL,
        params={
            "currency_pair": PAIR,
            "interval": interval,
            "limit": limit,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("Gate-ը տվյալներ չի վերադարձրել")

    rows = []

    for candle in data:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue

        try:
            timestamp, volume, close, high, low, open_price = candle[:6]

            rows.append({
                "time": int(timestamp),
                "volume": float(volume),
                "close": float(close),
                "high": float(high),
                "low": float(low),
                "open": float(open_price),
            })
        except (ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Մոմերի տվյալները սխալ ձևաչափ ունեն")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise ValueError("Բավարար քանակով մոմեր չկան")

    return df


# =========================
# ՑՈՒՑԱՆԻՇՆԵՐ
# =========================

def calculate_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0), "rsi"
    ] = 50

    high_low = df["high"] - df["low"]
    high_close = (
        df["high"] - df["close"].shift(1)
    ).abs()
    low_close = (
        df["low"] - df["close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


# =========================
# ՇՈՒԿԱՅԻ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ
# =========================

def analyze_market():
    df15 = calculate_indicators(get_candles("15m"))
    df1h = calculate_indicators(get_candles("1h"))

    last15 = df15.iloc[-1]
    last1h = df1h.iloc[-1]

    price = float(last15["close"])
    ema9_15 = float(last15["ema9"])
    ema21_15 = float(last15["ema21"])
    ema9_1h = float(last1h["ema9"])
    ema21_1h = float(last1h["ema21"])
    rsi = float(last15["rsi"])
    atr = float(last15["atr"])

    values = [
        price, ema9_15, ema21_15,
        ema9_1h, ema21_1h, rsi, atr,
    ]

    if not all(pd.notna(x) for x in values):
        raise ValueError("Ցուցանիշների հաշվարկը չի ավարտվել")

    if price <= 0 or atr <= 0:
        raise ValueError("Գնի կամ ATR-ի արժեքը սխալ է")

    bull15 = ema9_15 > ema21_15
    bear15 = ema9_15 < ema21_15
    bull1h = ema9_1h > ema21_1h
    bear1h = ema9_1h < ema21_1h

    signal = "NO TRADE"
    reason = "Հստակ ուղղություն չկա"
    waiting_for = "Սպասել նոր ազդանշանի"

    # Հաստատված պայմաններ՝ ըստ այս բոտի կանոնների
    if bull15 and bull1h and 50 < rsi < 68:
        signal = "LONG"
        reason = "15m և 1h ուղղությունները աճող են, RSI-ն հաստատում է"
        waiting_for = "Սցենարը հաստատված է, բայց ստուգիր ռիսկը"

    elif bear15 and bear1h and 32 < rsi < 50:
        signal = "SHORT"
        reason = "15m և 1h ուղղությունները նվազող են, RSI-ն հաստատում է"
        waiting_for = "Սցենարը հաստատված է, բայց ստուգիր ռիսկը"

    # Հնարավոր LONG. 1h աճող է, 15m-ում հետքաշում կա
    elif bull1h and bear15 and 30 < rsi < 45:
        signal = "LONG WATCH"
        reason = "1h աճող է, 15m-ում հնարավոր հետքաշում կա"
        waiting_for = (
            "Սպասիր 15m EMA9-ի բարձրանալուն EMA21-ից վերև "
            "և RSI-ի բարձրանալուն 45-ից վերև"
        )

    # Հնարավոր SHORT. 1h նվազող է, 15m-ում վերականգնում կա
    elif bear1h and bull15 and 55 < rsi < 70:
        signal = "SHORT WATCH"
        reason = "1h նվազող է, 15m-ում հնարավոր վերականգնում կա"
        waiting_for = (
            "Սպասիր 15m EMA9-ի իջնելուն EMA21-ից ներքև "
            "և RSI-ի իջնելուն 55-ից ներքև"
        )

    # Երկու ուղղություններն էլ աճող են, բայց RSI-ն դեռ թույլ է
    elif bull15 and bull1h and rsi <= 50:
        signal = "LONG WATCH"
        reason = "Երկու EMA ուղղություններն աճող են, բայց RSI-ն թույլ է"
        waiting_for = "Սպասիր RSI-ի բարձրանալուն 50-ից վերև"

    # Երկու ուղղություններն էլ նվազող են, բայց RSI-ն դեռ թույլ է
    elif bear15 and bear1h and rsi >= 50:
        signal = "SHORT WATCH"
        reason = "Երկու EMA ուղղություններն նվազող են, բայց RSI-ն բարձր է"
        waiting_for = "Սպասիր RSI-ի իջնելուն 50-ից ներքև"

    # Մոտավոր մակարդակներ ATR-ի հիման վրա
    stop_loss = None
    take_profit = None

    if signal == "LONG":
        stop_loss = price - 1.5 * atr
        take_profit = price + 2 * atr

    elif signal == "SHORT":
        stop_loss = price + 1.5 * atr
        take_profit = price - 2 * atr

    return {
        "signal": signal,
        "price": price,
        "rsi": rsi,
        "ema9_15": ema9_15,
        "ema21_15": ema21_15,
        "ema9_1h": ema9_1h,
        "ema21_1h": ema21_1h,
        "atr": atr,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": reason,
        "waiting_for": waiting_for,
    }


# =========================
# TELEGRAM ՀՐԱՄԱՆՆԵՐ
# =========================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Bot is working!\n\n"
        "BTC/USDT ազդանշանի համար գրիր /signal"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Ստուգում եմ BTC/USDT շուկան..."
    )

    try:
        result = await asyncio.to_thread(analyze_market)

        message = (
            f"BTC/USDT — {result['signal']}\n"
            f"Գին՝ {result['price']:.2f} USDT\n"
            f"RSI (15m)՝ {result['rsi']:.2f}\n"
            f"EMA 9/21 (15m)՝ "
            f"{result['ema9_15']:.2f} / "
            f"{result['ema21_15']:.2f}\n"
            f"EMA 9/21 (1h)՝ "
            f"{result['ema9_1h']:.2f} / "
            f"{result['ema21_1h']:.2f}\n"
            f"ATR (15m)՝ {result['atr']:.2f}\n\n"
            f"Պատճառ՝ {result['reason']}\n"
            f"Հաջորդ պայմանը՝ {result['waiting_for']}\n"
        )

        if result["signal"] in ("LONG", "SHORT"):
            message += (
                f"\nStop Loss՝ {result['stop_loss']:.2f}\n"
                f"Take Profit՝ {result['take_profit']:.2f}\n"
            )
        elif result["signal"] in ("LONG WATCH", "SHORT WATCH"):
            message += (
                "\nՍա դիտարկման ազդանշան է, ոչ թե մուտքի հաստատում։\n"
                "Մակարդակները մի՛ օգտագործիր որպես պատրաստի պատվեր։"
            )
        else:
            message += "\nԱյս պահին մուտքի հստակ պայմաններ չկան։"

        await update.message.reply_text(message)

    except Exception:
        logger.exception("Signal-ի հաշվարկը ձախողվեց")

        await update.message.reply_text(
            "Տվյալների ստացման կամ հաշվարկի սխալ։ "
            "Ստուգիր Fadehost-ի Logs-ը։"
        )


# =========================
# ՄԵԿՆԱՐԿ
# =========================

def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN փոփոխականը կարգավորված չէ"
        )

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("signal", signal_command))

    logger.info("Բոտը գործարկվում է...")
    app.run_polling()


if __name__ == "__main__":
    main()

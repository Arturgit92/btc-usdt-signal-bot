
import os
import asyncio
import logging

import requests
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ==================================
# ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ
# ==================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GATE_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAIR = "BTC_USDT"
CHECK_INTERVAL = 15 * 60  # 15 րոպե

# Բոտին /start ուղարկած օգտատերերը
subscribed_chats = set()

# Վերջին ազդանշանը՝ կրկնվող ծանուցումները կանխելու համար
last_signal_by_chat = {}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ==================================
# GATE API
# ==================================

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
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Մոմերի տվյալները սխալ են")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise ValueError("Բավարար քանակով մոմեր չկան")

    return df


# ==================================
# ՑՈՒՑԱՆԻՇՆԵՐ
# ==================================

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


# ==================================
# ՇՈՒԿԱՅԻ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ
# ==================================

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

    if not all(pd.notna(v) for v in values):
        raise ValueError("Ցուցանիշները պատրաստ չեն")

    if price <= 0 or atr <= 0:
        raise ValueError("Գնի կամ ATR-ի արժեքը սխալ է")

    bull15 = ema9_15 > ema21_15
    bear15 = ema9_15 < ema21_15
    bull1h = ema9_1h > ema21_1h
    bear1h = ema9_1h < ema21_1h

    signal = "NO TRADE"
    reason = "Ուղղությունները չեն հաստատվում"
    waiting_for = "Սպասել նոր պայմանների"

    # Հաստատված LONG՝ երկու ժամանակահատվածում աճ և RSI > 45
    if bull15 and bull1h and 45 < rsi < 68:
        signal = "LONG"
        reason = "15m և 1h EMA-ները աճող են, RSI-ն հաստատում է"
        waiting_for = "Հետևել ռիսկին և շուկայի փոփոխությանը"

    # Հաստատված SHORT՝ երկու ժամանակահատվածում անկում
    elif bear15 and bear1h and 32 < rsi < 55:
        signal = "SHORT"
        reason = "15m և 1h EMA-ները նվազող են, RSI-ն հաստատում է"
        waiting_for = "Հետևել ռիսկին և շուկայի փոփոխությանը"

    # LONG WATCH
    elif bull1h and bear15 and 30 < rsi < 45:
        signal = "LONG WATCH"
        reason = "1h աճող է, 15m-ում հետքաշում կա"
        waiting_for = (
            "15m EMA9-ը բարձրանա EMA21-ից վերև, "
            "իսկ RSI-ն բարձրանա 45-ից վերև"
        )

    # SHORT WATCH
    elif bear1h and bull15 and 55 < rsi < 70:
        signal = "SHORT WATCH"
        reason = "1h նվազող է, 15m-ում վերականգնում կա"
        waiting_for = (
            "15m EMA9-ը իջնի EMA21-ից ներքև, "
            "իսկ RSI-ն իջնի 55-ից ներքև"
        )

    # Աճող ուղղություն, բայց RSI-ն դեռ թույլ է
    elif bull15 and bull1h and rsi <= 45:
        signal = "LONG WATCH"
        reason = "Երկու ուղղություններն աճող են, RSI-ն դեռ թույլ է"
        waiting_for = "RSI-ն բարձրանա 45-ից վերև"

    # Նվազող ուղղություն, բայց RSI-ն դեռ բարձր է
    elif bear15 and bear1h and rsi >= 55:
        signal = "SHORT WATCH"
        reason = "Երկու ուղղություններն նվազող են, RSI-ն դեռ բարձր է"
        waiting_for = "RSI-ն իջնի 55-ից ներքև"

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


# ==================================
# ՀԱՂՈՐԴԱԳՐՈՒԹՅԱՆ ՏԵՔՍՏ
# ==================================

def format_signal(result):
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
            f"\nՆախնական Stop Loss՝ "
            f"{result['stop_loss']:.2f}\n"
            f"Նախնական Take Profit՝ "
            f"{result['take_profit']:.2f}\n"
            "\nՍտուգիր ռիսկը՝ գործարքից առաջ։"
        )

    elif result["signal"] in ("LONG WATCH", "SHORT WATCH"):
        message += (
            "\nՍա դիտարկման ազդանշան է, "
            "ոչ թե գործարքի հաստատում։"
        )

    return message


# ==================================
# TELEGRAM ՀՐԱՄԱՆՆԵՐ
# ==================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id
    subscribed_chats.add(chat_id)

    await update.effective_message.reply_text(
        "Bot is working!\n\n"
        "Ավտոմատ BTC/USDT ստուգումը միացված է։\n"
        "Ստուգում՝ ամեն 15 րոպեն մեկ։\n"
        "LONG կամ SHORT պայմանների դեպքում կստանաս ծանուցում։\n\n"
        "Ձեռքով ստուգելու համար գրիր /signal"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id
    subscribed_chats.add(chat_id)

    await update.effective_message.reply_text(
        "Ստուգում եմ BTC/USDT շուկան..."
    )

    try:
        result = await asyncio.to_thread(analyze_market)

        # Հիշում ենք ձեռքով ստուգված վիճակը,
        # որպեսզի նույն ազդանշանը անմիջապես չկրկնվի։
        last_signal_by_chat[chat_id] = result["signal"]

        await update.effective_message.reply_text(
            format_signal(result)
        )

    except Exception:
        logger.exception("Signal-ի հաշվարկը ձախողվեց")

        await update.effective_message.reply_text(
            "Տվյալների ստացման կամ հաշվարկի սխալ։ "
            "Ստուգիր Fadehost-ի Logs-ը։"
        )


# ==================================
# ԱՎՏՈՄԱՏ ՍՏՈՒԳՈՒՄ
# ==================================

async def periodic_alerts(app):
    # Սպասում ենք, որ բոտը լիովին գործարկվի։
    await asyncio.sleep(10)

    while True:
        try:
            if subscribed_chats:
                result = await asyncio.to_thread(analyze_market)
                signal = result["signal"]

                for chat_id in list(subscribed_chats):
                    previous = last_signal_by_chat.get(chat_id)

                    # Ծանուցում միայն նոր LONG կամ SHORT-ի դեպքում։
                    if (
                        signal in ("LONG", "SHORT")
                        and signal != previous
                    ):
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=(
                                    "🔔 ԱՎՏՈՄԱՏ ԱԶԴԱՆՇԱՆ\n\n"
                                    + format_signal(result)
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "Ծանուցումը չուղարկվեց chat_id=%s",
                                chat_id,
                            )

                    # Հիշում ենք վերջին վիճակը։
                    last_signal_by_chat[chat_id] = signal

                logger.info(
                    "Ավտոմատ ստուգումն ավարտվեց. signal=%s",
                    signal,
                )

        except Exception:
            logger.exception("Ավտոմատ ստուգման սխալ")

        await asyncio.sleep(CHECK_INTERVAL)


async def post_init(app):
    # Սկսում ենք ֆոնային ստուգումը՝ առանց հավելյալ գրադարանի։
    app.create_task(periodic_alerts(app))


# ==================================
# ՄԵԿՆԱՐԿ
# ==================================

def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN փոփոխականը կարգավորված չէ"
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("signal", signal_command))

    logger.info("Բոտը գործարկվում է...")
    app.run_polling()


if __name__ == "__main__":
    main()

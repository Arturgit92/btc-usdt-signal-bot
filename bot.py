
import os
import json
import logging
import urllib.parse
import urllib.request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API = "https://api.gateio.ws/api/v4/spot/candlesticks"

# Telegram chats that subscribed with /start
subscribers = set()
last_alert = None


def candles(interval):
    params = urllib.parse.urlencode({
        "currency_pair": "BTC_USDT",
        "interval": interval,
        "limit": 100,
    })
    req = urllib.request.Request(
        f"{API}?{params}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        rows = json.loads(response.read().decode())

    # Gate candle format: time, volume, close, high, low, open...
    rows.sort(key=lambda row: int(row[0]))

    # Exclude the newest candle, which may still be forming.
    rows = rows[:-1]

    return [
        {
            "time": int(row[0]),
            "close": float(row[2]),
            "high": float(row[3]),
            "low": float(row[4]),
        }
        for row in rows
    ]


def ema(values, period):
    if len(values) < period:
        raise ValueError("Not enough candles for EMA")
    result = sum(values[:period]) / period
    alpha = 2 / (period + 1)
    for value in values[period:]:
        result = value * alpha + result * (1 - alpha)
    return result


def rsi(values, period=14):
    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]
    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100 if avg_gain else 50

    return 100 - 100 / (1 + avg_gain / avg_loss)


def analyze():
    fast = candles("15m")
    slow = candles("1h")

    c15 = [x["close"] for x in fast]
    c1h = [x["close"] for x in slow]

    price = c15[-1]
    e9_15, e21_15 = ema(c15, 9), ema(c15, 21)
    e9_1h, e21_1h = ema(c1h, 9), ema(c1h, 21)
    momentum = rsi(c15)

    direction = "NO TRADE"
    reason = "15m և 1h միտումները բավարար չափով չեն հաստատում մուտքը։"

    if (
        e9_15 > e21_15
        and e9_1h > e21_1h
        and 50 < momentum < 68
    ):
        direction = "LONG"
        reason = "Երկու ժամային շրջանակներում EMA միտումը վերև է։"

    elif (
        e9_15 < e21_15
        and e9_1h < e21_1h
        and 32 < momentum < 50
    ):
        direction = "SHORT"
        reason = "Երկու ժամային շրջանակներում EMA միտումը ներքև է։"

    sl = tp = None
    if direction == "LONG":
        sl = price * 0.992
        tp = price * 1.016
    elif direction == "SHORT":
        sl = price * 1.008
        tp = price * 0.984

    message = (
        f"BTC/USDT — {direction}\n"
        f"Գին՝ {price:,.2f} USDT\n"
        f"RSI (15m)՝ {momentum:.1f}\n"
        f"EMA 9/21 (15m)՝ "
        f"{e9_15:,.2f} / {e21_15:,.2f}\n"
        f"EMA 9/21 (1h)՝ "
        f"{e9_1h:,.2f} / {e21_1h:,.2f}\n"
        f"Պատճառ՝ {reason}\n"
    )

    if sl is not None:
        message += (
            f"Մուտքի հղումային գին՝ {price:,.2f}\n"
            f"Stop Loss՝ {sl:,.2f}\n"
            f"Take Profit՝ {tp:,.2f}\n"
            "SL/TP-ն մոտավոր 0.8% / 1.6% են։"
        )

    message += "\n\nՍա փորձնական ազդանշան է, ոչ ֆինանսական երաշխիք։"
    return direction, fast[-1]["time"], message


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    await update.message.reply_text(
        "Bot is working! 🤖\n"
        "BTC/USDT signal bot is online.\n"
        "Ավտոմատ ստուգում՝ յուրաքանչյուր 15 րոպեն մեկ։\n"
        "Օգտագործիր /signal՝ հիմա ստուգելու համար։"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _, _, message = analyze()
        await update.message.reply_text(message)
    except Exception:
        log.exception("Signal analysis failed")
        await update.message.reply_text(
            "Չհաջողվեց ստանալ շուկայի տվյալները։ "
            "Փորձիր մի փոքր ուշ։"
        )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is running ✅")


async def scheduled_check(context: ContextTypes.DEFAULT_TYPE):
    global last_alert

    try:
        direction, candle_time, message = analyze()

        # Avoid repeated alerts for the same candle and direction.
        alert_key = (direction, candle_time)
        if direction not in ("LONG", "SHORT") or alert_key == last_alert:
            return

        last_alert = alert_key

        for chat_id in list(subscribers):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Ավտոմատ ազդանշան 📊\n\n" + message,
                )
            except Exception:
                log.exception("Could not send alert to a chat")

    except Exception:
        log.exception("Scheduled market check failed")


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("status", status))

    app.job_queue.run_repeating(
        scheduled_check,
        interval=900,
        first=10,
    )

    app.run_polling()


if __name__ == "__main__":
    main()

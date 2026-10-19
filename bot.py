
import os
import logging
import requests
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# Կարգավորումներ
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SYMBOL = "BTC_USDT"
BASE_URL = "https://api.gateio.ws/api/v4"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_candles(interval="15m", limit=100):
    url = f"{BASE_URL}/spot/candlesticks"
    response = requests.get(
        url,
        params={
            "currency_pair": SYMBOL,
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
    )
    response.raise_for_status()

    rows = response.json()
    df = pd.DataFrame(
        rows,
        columns=["time", "volume", "close", "high", "low", "open"]
    )
    df["close"] = pd.to_numeric(df["close"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def calculate_signal():
    df15 = get_candles("15m")
    df1h = get_candles("1h")

    for df in (df15, df1h):
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df15["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df15["rsi"] = 100 - (100 / (1 + rs))

    last = df15.iloc[-1]
    hour = df1h.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema9_15 = float(last["ema9"])
    ema21_15 = float(last["ema21"])
    ema9_1h = float(hour["ema9"])
    ema21_1h = float(hour["ema21"])

    long_ok = (
        ema9_15 > ema21_15
        and ema9_1h > ema21_1h
        and 50 <= rsi <= 68
        and price > ema9_15
    )

    short_ok = (
        ema9_15 < ema21_15
        and ema9_1h < ema21_1h
        and 32 <= rsi <= 50
        and price < ema9_15
    )

    if long_ok:
        signal = "LONG 🟢"
        reason = "EMA-ները աճող ուղղությամբ են, RSI-ն հաստատում է։"
    elif short_ok:
        signal = "SHORT 🔴"
        reason = "EMA-ները նվազող ուղղությամբ են, RSI-ն հաստատում է։"
    else:
        signal = "NO TRADE ⚪"
        reason = "Հստակ պայմաններ չկան։ Սպասում ենք։"

    return (
        f"BTC/USDT — {signal}\n"
        f"Գին՝ {price:,.2f} USDT\n\n"
        f"RSI (15m)՝ {rsi:.2f}\n"
        f"EMA 9 (15m)՝ {ema9_15:,.2f}\n"
        f"EMA 21 (15m)՝ {ema21_15:,.2f}\n"
        f"EMA 9 (1h)՝ {ema9_1h:,.2f}\n"
        f"EMA 21 (1h)՝ {ema21_1h:,.2f}\n\n"
        f"Մեկնաբանություն՝ {reason}\n\n"
        "⚠️ Սա ցուցիչների ազդանշան է, ոչ թե շահույթի երաշխիք։"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Բոտը աշխատում է։ ✅\n\n"
        "/signal — ստուգել BTC/USDT ազդանշանը\n"
        "/start — բացել բոտը"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = await update.message.reply_text(
        "Ստուգում եմ BTC/USDT շուկան..."
    )
    try:
        result = calculate_signal()
        await message.edit_text(result)
    except Exception:
        logging.exception("Signal calculation failed")
        await message.edit_text(
            "Չհաջողվեց ստանալ տվյալները։ Փորձիր մի փոքր ուշ։"
        )


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN-ը սահմանված չէ Secrets-ում։"
        )

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

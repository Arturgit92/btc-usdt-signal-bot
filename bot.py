
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

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SYMBOL = "BTC_USDT"
BASE_URL = "https://api.gateio.ws/api/v4"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_candles(interval="15m", limit=150):
    response = requests.get(
        f"{BASE_URL}/spot/candlesticks",
        params={
            "currency_pair": SYMBOL,
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
    )
    response.raise_for_status()

    df = pd.DataFrame(
        response.json(),
        columns=[
            "time", "volume", "close",
            "high", "low", "open"
        ],
    )

    for col in ["close", "high", "low", "open"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.sort_values("time").reset_index(drop=True)

    # Վերջին մոմը կարող է դեռ չփակված լինել։
    # Վերլուծության համար օգտագործում ենք փակված մոմերը։
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    return df.reset_index(drop=True)


def add_indicators(df):
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

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


def make_plan(direction, price, ema9, ema21, atr):
    # Մուտքի մոտավոր գոտի՝ EMA9-ի շուրջ։
    # Այն պատվեր տեղադրելու հրահանգ չէ։
    buffer = atr * 0.15
    entry_low = ema9 - buffer
    entry_high = ema9 + buffer

    # LONG-ի համար սպասում ենք, որ գինը
    # վերադարձից հետո հաստատի աճող շարժումը։
    # SHORT-ի համար՝ նվազող շարժումը։

    entry = (entry_low + entry_high) / 2

    if direction == "LONG":
        stop = entry - 1.2 * atr
        risk = entry - stop
        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        confirmation = (
            "Գինը պահպանում է մուտքի գոտին, "
            "15m մոմը փակում է EMA9-ից վեր, "
            "RSI-ն գերադասելի է 50-ից բարձր։"
        )
    else:
        stop = entry + 1.2 * atr
        risk = stop - entry
        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        confirmation = (
            "Գինը մերժվում է մուտքի գոտուց, "
            "15m մոմը փակում է EMA9-ից ներքև, "
            "RSI-ն գերադասելի է 50-ից ցածր։"
        )

    return (
        f"{direction} սցենար\n"
        f"Մուտքի գոտի՝ {entry_low:,.2f} – "
        f"{entry_high:,.2f}\n"
        f"SL՝ {stop:,.2f}\n"
        f"TP1՝ {tp1:,.2f}\n"
        f"TP2՝ {tp2:,.2f}\n"
        f"Ռիսկ/շահույթ՝ 1:1.5 և 1:2.5\n"
        f"Հաստատում՝ {confirmation}"
    )


def calculate_signal():
    df15 = add_indicators(get_candles("15m"))
    df1h = add_indicators(get_candles("1h"))

    last = df15.iloc[-1]
    hour = df1h.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema9 = float(last["ema9"])
    ema21 = float(last["ema21"])
    ema9h = float(hour["ema9"])
    ema21h = float(hour["ema21"])
    atr = float(last["atr"])

    if pd.isna(rsi) or pd.isna(atr) or atr <= 0:
        return "Բավարար տվյալներ չկան։ Փորձիր ավելի ուշ։"

    long_ok = (
        ema9 > ema21
        and ema9h > ema21h
        and 50 <= rsi <= 68
        and price > ema9
    )

    short_ok = (
        ema9 < ema21
        and ema9h < ema21h
        and 32 <= rsi <= 50
        and price < ema9
    )

    text = (
        f"BTC/USDT\n"
        f"Գին՝ {price:,.2f} USDT\n"
        f"RSI (15m)՝ {rsi:.1f}\n"
        f"EMA 9/21 (15m)՝ {ema9:,.2f} / "
        f"{ema21:,.2f}\n"
        f"EMA 9/21 (1h)՝ {ema9h:,.2f} / "
        f"{ema21h:,.2f}\n"
        f"ATR (15m)՝ {atr:,.2f}\n\n"
    )

    if long_ok:
        text += "ԱԶԴԱՆՇԱՆ՝ LONG 🟢\n\n"
        text += make_plan("LONG", price, ema9, ema21, atr)

    elif short_ok:
        text += "ԱԶԴԱՆՇԱՆ՝ SHORT 🔴\n\n"
        text += make_plan("SHORT", price, ema9, ema21, atr)

    else:
        text += (
            "ԱԶԴԱՆՇԱՆ՝ NO TRADE ⚪\n"
            "Մուտքի հաստատում դեռ չկա։\n\n"
            "LONG-ի պայմանական սցենար՝\n"
        )
        text += make_plan("LONG", price, ema9, ema21, atr)
        text += "\n\nSHORT-ի պայմանական սցենար՝\n"
        text += make_plan("SHORT", price, ema9, ema21, atr)

    text += (
        "\n\n⚠️ Մակարդակները ATR/EMA-ի հիման վրա "
        "հաշվարկային են, ոչ երաշխավորված։ "
        "Սպասիր հաստատման։ Բոտը գործարք չի բացում։"
    )

    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Բոտը աշխատում է։ ✅\n\n"
        "/signal — BTC/USDT վերլուծություն\n"
        "/start — օգնություն"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = await update.message.reply_text(
        "Ստուգում եմ շուկան..."
    )

    try:
        result = calculate_signal()
        await message.edit_text(result)
    except Exception:
        logging.exception("Signal error")
        await message.edit_text(
            "Տվյալների ստացման սխալ։ "
            "Ստուգիր ինտերնետը և փորձիր կրկին։"
        )


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN-ը բացակայում է Replit Secrets-ից։"
        )

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

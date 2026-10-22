
import os
import time
import asyncio
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GATE_URL = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
CONTRACT = "BTC_USDT"

FEE_RATE = float(os.getenv("FUTURES_FEE_RATE", "0.0005"))
CHECK_EVERY_SECONDS = 300  # Ստուգում՝ ամեն 5 րոպեն մեկ
ATR_MULTIPLIER = 1.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------
# GATE DATA
# -------------------------

def request_candles(interval, start_ts, end_ts):
    rows_all = []
    chunk_size = 5 * 24 * 60 * 60
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_size, end_ts)

        response = requests.get(
            GATE_URL,
            params={
                "contract": CONTRACT,
                "interval": interval,
                "from": int(cursor),
                "to": int(chunk_end),
            },
            timeout=25,
        )
        response.raise_for_status()
        rows = response.json()

        if not isinstance(rows, list):
            raise ValueError(f"Gate API պատասխան՝ {rows}")

        rows_all.extend(rows)
        cursor = chunk_end
        time.sleep(0.1)

    parsed = []

    for row in rows_all:
        try:
            if isinstance(row, dict):
                parsed.append({
                    "time": int(row["t"]),
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": float(row.get("v", 0)),
                })
            else:
                parsed.append({
                    "time": int(row[0]),
                    "volume": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "open": float(row[5]),
                })
        except (KeyError, IndexError, TypeError, ValueError):
            continue

    df = pd.DataFrame(parsed)
    if df.empty:
        raise ValueError(f"Gate-ից {interval} տվյալներ չկան")

    return (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )


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
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    return df


def get_market_data(days=10):
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - days * 86400

    df15 = add_indicators(request_candles("15m", start, now))
    df1h = add_indicators(request_candles("1h", start, now))

    return df15, df1h


# -------------------------
# STRICT RSI SIGNAL
# -------------------------

def get_strict_signal(df15, df1h, index):
    """
    Index-ը պետք է լինի վերջին ամբողջությամբ փակված 15m մոմը։
    1h EMA-ն օգտագործվում է միայն փակված ժամային մոմից։
    """
    if index < 30:
        return None

    row = df15.iloc[index]

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    close_time = int(row["time"]) + 15 * 60

    eligible = df1h[
        df1h["time"] + 3600 <= close_time
    ]
    if eligible.empty:
        return None

    hour = eligible.iloc[-1]

    rsi = float(row["rsi"])

    long_ok = (
        row["ema9"] > row["ema21"]
        and hour["ema9"] > hour["ema21"]
        and 53 <= rsi <= 65
    )

    short_ok = (
        row["ema9"] < row["ema21"]
        and hour["ema9"] < hour["ema21"]
        and 35 <= rsi <= 47
    )

    if long_ok:
        side = "LONG"
    elif short_ok:
        side = "SHORT"
    else:
        return None

    return {
        "side": side,
        "time": int(row["time"]),
        "price": float(row["close"]),
        "rsi": rsi,
        "atr": float(row["atr"]),
    }


def make_levels(entry, side, atr):
    risk = atr * ATR_MULTIPLIER

    if side == "LONG":
        sl = entry - risk
        tp1 = entry + 2 * risk
        tp2 = entry + 3 * risk
    else:
        sl = entry + risk
        tp1 = entry - 2 * risk
        tp2 = entry - 3 * risk

    return sl, tp1, tp2


def signal_message(signal):
    entry = signal["price"]
    side = signal["side"]
    sl, tp1, tp2 = make_levels(entry, side, signal["atr"])

    candle_time = datetime.fromtimestamp(
        signal["time"], timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"🚨 BTC/USDT FUTURES — {side}\n\n"
        f"15m փակ մոմ՝ {candle_time}\n"
        f"Հղումային գին՝ {entry:.2f} USDT\n"
        f"RSI (15m)՝ {signal['rsi']:.1f}\n\n"
        f"Stop Loss՝ {sl:.2f}\n"
        f"TP1՝ {tp1:.2f} (փակել 50%)\n"
        f"TP2՝ {tp2:.2f} (մնացած 50%)\n\n"
        "Ֆիլտր՝ խիստ RSI + EMA 9/21 (15m և 1h)\n"
        "⚠️ Սա ազդանշան է, ոչ թե ավտոմատ գործարք։ "
        "Հաշվի առ կոմիսիան, սահքը և շուկայի ռիսկը։"
    )


# -------------------------
# AUTOMATIC ALERTS
# -------------------------

async def check_automatic_signal(
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = context.application.bot_data.get("alert_chat_id")

    # Նախ պետք է Telegram-ում /start ուղարկվի։
    if not chat_id:
        return

    try:
        df15, df1h = await asyncio.to_thread(get_market_data, 10)

        # Վերջին մոմը կարող է դեռ ձևավորվել։
        # Վերցնում ենք նախորդ՝ փակված մոմը։
        index = len(df15) - 2
        if index < 30:
            return

        row = df15.iloc[index]
        candle_time = int(row["time"])

        # Յուրաքանչյուր փակ մոմ ստուգվում է միայն մեկ անգամ։
        last_checked = context.application.bot_data.get(
            "last_checked_candle"
        )

        if last_checked == candle_time:
            return

        # Նշում ենք ստուգված՝ նույնիսկ եթե ազդանշան չկա։
        context.application.bot_data[
            "last_checked_candle"
        ] = candle_time

        signal = get_strict_signal(df15, df1h, index)

        if signal is None:
            logger.info(
                "NO TRADE: վերջին փակ մոմը %s", candle_time
            )
            return

        # Նույն մոմի ազդանշանը կրկին չենք ուղարկում։
        last_alert = context.application.bot_data.get(
            "last_alert_candle"
        )
        if last_alert == candle_time:
            return

        await context.bot.send_message(
            chat_id=chat_id,
            text=signal_message(signal),
        )

        context.application.bot_data[
            "last_alert_candle"
        ] = candle_time

        logger.info(
            "Ավտոմատ ազդանշան ուղարկվեց՝ %s",
            signal["side"],
        )

    except Exception:
        logger.exception("Ավտոմատ ազդանշանի ստուգման սխալ")


# -------------------------
# TELEGRAM COMMANDS
# -------------------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id
    context.application.bot_data["alert_chat_id"] = chat_id

    await update.message.reply_text(
        "Բոտը աշխատում է։ Ավտոմատ ազդանշանները միացված են։\n\n"
        "Շուկա՝ BTC/USDT Futures\n"
        "Ֆիլտր՝ խիստ RSI + EMA 9/21\n"
        "Ստուգում՝ ամեն 5 րոպեն մեկ\n\n"
        "/signal — ձեռքով ստուգել շուկան\n"
        "/backtest 30 — բեքթեստի հրաման\n"
        "/stopalerts — անջատել ավտոմատ ազդանշանները\n\n"
        "Բոտը գործարքներ ինքնուրույն չի բացում։"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Ստուգում եմ BTC/USDT Futures-ը..."
    )

    try:
        df15, df1h = await asyncio.to_thread(get_market_data, 10)
        index = len(df15) - 2

        signal = get_strict_signal(df15, df1h, index)

        if signal is None:
            row = df15.iloc[index]
            await update.message.reply_text(
                "BTC/USDT — NO TRADE\n\n"
                f"Փակ մոմի գին՝ {row['close']:.2f}\n"
                f"RSI (15m)՝ {row['rsi']:.1f}\n"
                "Խիստ ֆիլտրի բոլոր պայմանները չեն կատարվել։"
            )
            return

        await update.message.reply_text(signal_message(signal))

    except Exception as exc:
        logger.exception("Ձեռքով ստուգման սխալ")
        await update.message.reply_text(
            f"Շուկայի տվյալների սխալ՝ {str(exc)[:250]}"
        )


async def stopalerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    context.application.bot_data.pop("alert_chat_id", None)

    await update.message.reply_text(
        "Ավտոմատ ազդանշաններն անջատված են։\n"
        "Նորից միացնելու համար ուղարկիր /start։"
    )


async def backtest_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Այս տարբերակում ավտոմատ ազդանշաններն են թարմացված։\n"
        "Բեքթեստի նախկին համեմատական կոդը պետք է պահել "
        "նախորդ տարբերակից առանձին, եթե այն դեռ օգտագործում ես։"
    )


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN միջավայրի փոփոխականը բացակայում է"
        )

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("stopalerts", stopalerts_command))

    # Աշխատում է, քանի դեռ հոսթինգում բոտի պրոցեսը միացված է։
    app.job_queue.run_repeating(
        check_automatic_signal,
        interval=CHECK_EVERY_SECONDS,
        first=10,
        name="btc_strict_rsi_alerts",
    )

    logger.info("BTC/USDT ավտոմատ ազդանշանների բոտը գործարկվում է")
    app.run_polling()


if __name__ == "__main__":
    main()

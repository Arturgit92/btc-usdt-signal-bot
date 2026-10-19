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

# ===== ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GATE_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAIR = "BTC_USDT"

CHECK_SECONDS = 60
ATR_SL_MULTIPLIER = 1.5
ATR_TP1_MULTIPLIER = 1.5
ATR_TP2_MULTIPLIER = 2.5

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_candles(interval, limit=100):
    response = requests.get(
        GATE_URL,
        params={
            "currency_pair": PAIR,
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
    )
    response.raise_for_status()
    raw = response.json()

    rows = []
    for item in raw:
        # Gate-ի մոմի դաշտերը՝
        # ժամանակ, quote volume, close, high, low, open, volume
        if len(item) < 6:
            continue

        rows.append({
            "time": int(item[0]),
            "close": float(item[2]),
            "high": float(item[3]),
            "low": float(item[4]),
            "open": float(item[5]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Gate-ը մոմերի տվյալներ չի վերադարձրել")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Վերջին մոմը կարող է դեռ չփակված լինել։
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < 30:
        raise ValueError("Բավարար մոմեր չկան հաշվարկի համար")

    return df.reset_index(drop=True)


def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    change = df["close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

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
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    return df


def analyze():
    m15 = add_indicators(get_candles("15m"))
    h1 = add_indicators(get_candles("1h"))

    # Վերջին երկու փակ մոմերը
    prev = m15.iloc[-2]
    cur = m15.iloc[-1]
    hour = h1.iloc[-1]

    price = float(cur["close"])
    atr = float(cur["atr"])
    rsi = float(cur["rsi"])

    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR-ի հաշվարկը պատրաստ չէ")

    bull_1h = hour["ema9"] > hour["ema21"]
    bear_1h = hour["ema9"] < hour["ema21"]

    # Փակ մոմի վրա EMA խաչման ստուգում
    cross_up = (
        prev["ema9"] <= prev["ema21"]
        and cur["ema9"] > cur["ema21"]
    )

    cross_down = (
        prev["ema9"] >= prev["ema21"]
        and cur["ema9"] < cur["ema21"]
    )

    side = None
    reason = ""

    if cross_up and bull_1h and 45 < rsi < 68:
        side = "LONG"
        reason = (
            "15m EMA9-ը խաչել է EMA21-ը վերևից, "
            "1h ուղղությունը աճող է, RSI-ն հաստատում է շարժումը։"
        )

    elif cross_down and bear_1h and 32 < rsi < 55:
        side = "SHORT"
        reason = (
            "15m EMA9-ը խաչել է EMA21-ը ներքևից, "
            "1h ուղղությունը նվազող է, RSI-ն հաստատում է շարժումը։"
        )

    # Մուտքի, Stop Loss-ի և Take Profit-ի հաշվարկ
    if side == "LONG":
        entry = price
        sl = entry - ATR_SL_MULTIPLIER * atr
        tp1 = entry + ATR_TP1_MULTIPLIER * atr
        tp2 = entry + ATR_TP2_MULTIPLIER * atr

    elif side == "SHORT":
        entry = price
        sl = entry + ATR_SL_MULTIPLIER * atr
        tp1 = entry - ATR_TP1_MULTIPLIER * atr
        tp2 = entry - ATR_TP2_MULTIPLIER * atr

    else:
        entry = sl = tp1 = tp2 = None

    return {
        "candle_time": int(cur["time"]),
        "side": side,
        "price": price,
        "rsi": rsi,
        "ema9": float(cur["ema9"]),
        "ema21": float(cur["ema21"]),
        "ema9_1h": float(hour["ema9"]),
        "ema21_1h": float(hour["ema21"]),
        "atr": atr,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "reason": reason,
    }


def format_signal(s):
    if not s["side"]:
        return (
            "BTC/USDT — NO TRADE\n"
            f"Փակ մոմի գին՝ {s['price']:.2f} USDT\n"
            f"RSI (15m)՝ {s['rsi']:.2f}\n"
            f"EMA 9/21 (15m)՝ "
            f"{s['ema9']:.2f} / {s['ema21']:.2f}\n"
            "Հստակ մուտքի պայման դեռ չի հաստատվել։"
        )

    return (
        f"🚨 BTC/USDT — {s['side']} ԱԶԴԱՆՇԱՆ\n\n"
        f"Մուտքի հաշվարկային գին՝ {s['entry']:.2f} USDT\n"
        f"Stop Loss՝ {s['sl']:.2f}\n"
        f"Take Profit 1՝ {s['tp1']:.2f}\n"
        f"Take Profit 2՝ {s['tp2']:.2f}\n\n"
        f"RSI (15m)՝ {s['rsi']:.2f}\n"
        f"ATR (15m)՝ {s['atr']:.2f}\n\n"
        f"Պատճառ՝ {s['reason']}\n\n"
        "⚠️ Սա ազդանշան է, ոչ թե երաշխավորված շահույթ։\n"
        "Գինը կարող է փոխվել մինչև պատվերի տեղադրումը։"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )
    await update.message.reply_text(
        "Բոտը միացված է։\n"
        "Ավտոմատ ստուգումները ակտիվ են։\n"
        "Օգտագործիր /signal՝ հիմա ստուգելու համար։"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )

    try:
        result = await asyncio.to_thread(analyze)
        await update.message.reply_text(format_signal(result))
    except Exception:
        log.exception("Signal check failed")
        await update.message.reply_text(
            "Տվյալները ստանալու սխալ է։ Փորձիր մի փոքր ուշ։"
        )


async def periodic_alerts(app: Application):
    # Մեկ անգամ պահում ենք վերջին ստուգված փակ մոմի ժամանակը,
    # որպեսզի վերագործարկումից հետո հին ազդանշանը չկրկնվի։
    last_seen_candle = None

    while True:
        try:
            result = await asyncio.to_thread(analyze)
            candle_time = result["candle_time"]

            if last_seen_candle is None:
                last_seen_candle = candle_time

            elif candle_time != last_seen_candle:
                last_seen_candle = candle_time

                if result["side"]:
                    message = format_signal(result)
                    chat_ids = list(
                        app.bot_data.get("chat_ids", set())
                    )

                    for chat_id in chat_ids:
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                            )
                        except Exception:
                            log.exception(
                                "Telegram alert send failed"
                            )

        except Exception:
            log.exception("Automatic signal check failed")

        await asyncio.sleep(CHECK_SECONDS)


async def post_init(app: Application):
    app.create_task(periodic_alerts(app))


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is missing"
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    app.run_polling()


if __name__ == "__main__":
    main()

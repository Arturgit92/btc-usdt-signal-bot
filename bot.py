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

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GATE_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAIR = "BTC_USDT"
CHECK_SECONDS = 60

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_candles(interval, limit=120):
    r = requests.get(
        GATE_URL,
        params={
            "currency_pair": PAIR,
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
    )
    r.raise_for_status()

    rows = []
    for x in r.json():
        if len(x) >= 6:
            rows.append({
                "time": int(x[0]),
                "close": float(x[2]),
                "high": float(x[3]),
                "low": float(x[4]),
                "open": float(x[5]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Gate-ից տվյալներ չկան")

    df = (
        df.drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Վերջին մոմը կարող է դեռ բաց լինել
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < 30:
        raise ValueError("Մոմերի քանակը բավարար չէ")

    return df.reset_index(drop=True)


def indicators(df):
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
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    return df


def analyze():
    m = indicators(get_candles("15m"))
    h = indicators(get_candles("1h"))

    prev = m.iloc[-2]
    cur = m.iloc[-1]
    hour = h.iloc[-1]

    price = float(cur["close"])
    rsi = float(cur["rsi"])
    atr = float(cur["atr"])

    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR-ը հաշվարկել չի հաջողվել")

    bull1h = hour["ema9"] > hour["ema21"]
    bear1h = hour["ema9"] < hour["ema21"]

    # Մոմի ուղղությունը և EMA-ի վերականգնումը
    bullish_candle = cur["close"] > cur["open"]
    bearish_candle = cur["close"] < cur["open"]

    reclaim_long = (
        prev["close"] <= prev["ema9"]
        and cur["close"] > cur["ema9"]
        and bullish_candle
    )

    reclaim_short = (
        prev["close"] >= prev["ema9"]
        and cur["close"] < cur["ema9"]
        and bearish_candle
    )

    # Խաչում կամ հետքաշումից վերականգնում
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

    if bull1h and 45 < rsi < 68 and (
        cross_up or reclaim_long
    ):
        side = "LONG"
        reason = (
            "1h աճող ուղղություն, 15m մոմը վերականգնվել է "
            "EMA 9-ից վերև, RSI-ն հաստատում է։"
        )

    elif bear1h and 32 < rsi < 55 and (
        cross_down or reclaim_short
    ):
        side = "SHORT"
        reason = (
            "1h նվազող ուղղություն, 15m մոմը իջել է "
            "EMA 9-ից ներքև, RSI-ն հաստատում է։"
        )

    if side == "LONG":
        entry = price
        sl = entry - 1.5 * atr
        tp1 = entry + 1.5 * atr
        tp2 = entry + 2.5 * atr
    elif side == "SHORT":
        entry = price
        sl = entry + 1.5 * atr
        tp1 = entry - 1.5 * atr
        tp2 = entry - 2.5 * atr
    else:
        entry = sl = tp1 = tp2 = None

    return {
        "time": int(cur["time"]),
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


def format_result(s):
    if not s["side"]:
        return (
            "BTC/USDT — NO TRADE\n"
            f"Փակ մոմի գին՝ {s['price']:.2f} USDT\n"
            f"RSI (15m)՝ {s['rsi']:.2f}\n"
            f"EMA 9/21 (15m)՝ "
            f"{s['ema9']:.2f} / {s['ema21']:.2f}\n"
            f"EMA 9/21 (1h)՝ "
            f"{s['ema9_1h']:.2f} / {s['ema21_1h']:.2f}\n"
            "Մուտքի հաստատում չկա։ Սպասում ենք։"
        )

    return (
        f"🚨 BTC/USDT — {s['side']}\n\n"
        f"Հաշվարկային մուտք՝ {s['entry']:.2f}\n"
        f"Stop Loss՝ {s['sl']:.2f}\n"
        f"Take Profit 1՝ {s['tp1']:.2f}\n"
        f"Take Profit 2՝ {s['tp2']:.2f}\n\n"
        f"RSI՝ {s['rsi']:.2f}\n"
        f"ATR՝ {s['atr']:.2f}\n"
        f"Պատճառ՝ {s['reason']}\n\n"
        "⚠️ Ազդանշանը երաշխավորված շահույթ չէ։ "
        "Մուտքի իրական գինը կարող է տարբերվել։"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )
    await update.message.reply_text(
        "Բոտը աշխատում է։ Ավտոմատ ազդանշանները միացված են։\n"
        "Հրաման՝ /signal"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )
    try:
        result = await asyncio.to_thread(analyze)
        await update.message.reply_text(format_result(result))
    except Exception:
        log.exception("Signal error")
        await update.message.reply_text(
            "Տվյալների սխալ։ Փորձիր ավելի ուշ։"
        )


async def periodic_alerts(app: Application):
    last_time = None

    while True:
        try:
            s = await asyncio.to_thread(analyze)

            if last_time is None:
                last_time = s["time"]

            elif s["time"] != last_time:
                last_time = s["time"]

                if s["side"]:
                    for chat_id in list(
                        app.bot_data.get("chat_ids", set())
                    ):
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=format_result(s),
                            )
                        except Exception:
                            log.exception("Telegram send error")

        except Exception:
            log.exception("Automatic check failed")

        await asyncio.sleep(CHECK_SECONDS)


async def post_init(app: Application):
    app.create_task(periodic_alerts(app))


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN-ը բացակայում է")

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

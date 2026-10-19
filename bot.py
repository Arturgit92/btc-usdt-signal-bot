
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
# GATE API - ՄՈՄԵՐԻ ՏՎՅԱԼՆԵՐ
# =========================

def get_candles(interval="15m", limit=100):
    params = {
        "currency_pair": PAIR,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        GATE_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("Gate API-ն դատարկ տվյալներ է վերադարձրել")

    rows = []

    for candle in data:
        # Gate-ի պատասխանը կարող է ունենալ 6-ից ավելի դաշտ։
        # Մեզ անհրաժեշտ են առաջին 6 դաշտերը։
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

    if len(rows) < 30:
        raise ValueError(
            f"Բավարար մոմեր չկան․ ստացվել է {len(rows)}"
        )

    df = pd.DataFrame(rows)

    # Հին մոմերը՝ առաջինը, նորերը՝ վերջում։
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    return df


# =========================
# ՏԵԽՆԻԿԱԿԱՆ ՑՈՒՑԱՆԻՇՆԵՐ
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

    relative_strength = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + relative_strength)
    )

    # RSI-ի հատուկ դեպքեր
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
# ԱԶԴԱՆՇԱՆԻ ՀԱՇՎԱՐԿ
# =========================

def analyze_market():
    candles_15m = get_candles("15m")
    candles_1h = get_candles("1h")

    df15 = calculate_indicators(candles_15m)
    df1h = calculate_indicators(candles_1h)

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
        raise ValueError("Ցուցանիշները դեռ պատրաստ չեն")

    if price <= 0 or atr <= 0:
        raise ValueError("Գնի կամ ATR-ի արժեքը սխալ է")

    signal = "NO TRADE"
    reason = "Ժամանակահատվածների ուղղությունները չեն հաստատվում"

    # LONG՝ երկու ժամանակահատվածներում աճող ուղղություն
    if (
        ema9_15 > ema21_15
        and ema9_1h > ema21_1h
        and 50 < rsi < 68
    ):
        signal = "LONG"
        reason = "EMA-ները աճող են, RSI-ն հաստատում է շարժումը"

    # SHORT՝ երկու ժամանակահատվածներում նվազող ուղղություն
    elif (
        ema9_15 < ema21_15
        and ema9_1h < ema21_1h
        and 32 < rsi < 50
    ):
        signal = "SHORT"
        reason = "EMA-ները նվազող են, RSI-ն հաստատում է շարժումը"

    # Նախնական մակարդակներ՝ ATR-ի հիման վրա
    if signal == "LONG":
        stop_loss = price - (1.5 * atr)
        take_profit = price + (2 * atr)

    elif signal == "SHORT":
        stop_loss = price + (1.5 * atr)
        take_profit = price - (2 * atr)

    else:
        stop_loss = None
        take_profit = None

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
    }


# =========================
# TELEGRAM ՀՐԱՄԱՆՆԵՐ
# =========================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text("Bot is working!")


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
            f"ATR (15m)՝ {result['atr']:.2f}\n"
            f"Պատճառ՝ {result['reason']}\n"
        )

        if result["signal"] != "NO TRADE":
            message += (
                f"Stop Loss՝ {result['stop_loss']:.2f}\n"
                f"Take Profit՝ {result['take_profit']:.2f}\n"
            )
        else:
            message += (
                "Մուտքի պայմանները հաստատված չեն։\n"
                "Մի՛ բացիր գործարք միայն այս ազդանշանի հիման վրա։"
            )

        await update.message.reply_text(message)

    except Exception:
        logger.exception("Signal-ի հաշվարկը ձախողվեց")

        await update.message.reply_text(
            "Տվյալների ստացման սխալ։ "
            "Ստուգիր Fadehost-ի Logs-ը։"
        )


# =========================
# ԳՈՐԾԱՐԿՈՒՄ
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


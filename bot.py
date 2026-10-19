
import os
import time
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

# ========== ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ ==========

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GATE_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAIR = "BTC_USDT"

CHECK_SECONDS = 60
ATR_MULTIPLIER = 1.5
TP2_MULTIPLIER = 2.5

# Մոտավոր մուտքի և ելքի վճարների գումարը՝ տոկոսով
ROUND_TRIP_FEE_PCT = 0.20

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ========== GATE ՏՎՅԱԼՆԵՐ ==========

def get_candles(interval, limit=120):
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

    rows = []
    for item in response.json():
        if len(item) >= 6:
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

    # Բաց մոմը չենք օգտագործում
    duration = 900 if interval == "15m" else 3600
    now = int(time.time())
    df = df[df["time"] + duration <= now]

    if len(df) < 30:
        raise ValueError("Բավարար փակ մոմեր չկան")

    return df.reset_index(drop=True)


def fetch_history(interval, days=30):
    """Ներբեռնում է պատմական մոմերը փոքր ժամանակահատվածներով."""
    duration = 900 if interval == "15m" else 3600

    now = int(time.time())
    end = (now // duration) * duration
    start = max(0, end - days * 86400)

    rows = []
    cursor = start

    # Յուրաքանչյուր հարցման համար առավելագույնը 800 մոմ
    chunk_seconds = duration * 800

    while cursor < end:
        chunk_end = min(cursor + chunk_seconds, end)

        response = requests.get(
            GATE_URL,
            params={
                "currency_pair": PAIR,
                "interval": interval,
                "from": cursor,
                "to": chunk_end,
                "limit": 1000,
            },
            timeout=25,
        )
        response.raise_for_status()
        batch = response.json()

        for item in batch:
            if len(item) >= 6:
                rows.append({
                    "time": int(item[0]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "open": float(item[5]),
                })

        cursor = chunk_end + 1
        time.sleep(0.08)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Պատմական տվյալներ չստացվեցին")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Հեռացնում ենք դեռ չփակված մոմերը
    now = int(time.time())
    df = df[df["time"] + duration <= now]

    if len(df) < 50:
        raise ValueError(
            "Պատմական մոմերը քիչ են։ Ստուգիր Gate API-ի տվյալները։"
        )

    return df.reset_index(drop=True)


# ========== ՑՈՒՑԱՆԻՇՆԵՐ ==========

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
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

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
        adjust=False,
        min_periods=14,
    ).mean()

    return df


# ========== ԱԶԴԱՆՇԱՆԻ ՏՐԱՄԱԲԱՆՈՒԹՅՈՒՆ ==========

def determine_signal(prev, cur, hour):
    bull1h = hour["ema9"] > hour["ema21"]
    bear1h = hour["ema9"] < hour["ema21"]

    cross_up = (
        prev["ema9"] <= prev["ema21"]
        and cur["ema9"] > cur["ema21"]
    )

    cross_down = (
        prev["ema9"] >= prev["ema21"]
        and cur["ema9"] < cur["ema21"]
    )

    reclaim_long = (
        prev["close"] <= prev["ema9"]
        and cur["close"] > cur["ema9"]
        and cur["close"] > cur["open"]
    )

    reclaim_short = (
        prev["close"] >= prev["ema9"]
        and cur["close"] < cur["ema9"]
        and cur["close"] < cur["open"]
    )

    rsi = float(cur["rsi"])

    if (
        bull1h
        and 45 < rsi < 68
        and (cross_up or reclaim_long)
    ):
        return (
            "LONG",
            "1h աճող ուղղություն, 15m վերականգնում, RSI-ի հաստատում",
        )

    if (
        bear1h
        and 32 < rsi < 55
        and (cross_down or reclaim_short)
    ):
        return (
            "SHORT",
            "1h նվազող ուղղություն, 15m անկում, RSI-ի հաստատում",
        )

    return None, ""


def analyze():
    m15 = indicators(get_candles("15m"))
    h1 = indicators(get_candles("1h"))

    prev = m15.iloc[-2]
    cur = m15.iloc[-1]
    hour = h1.iloc[-1]

    price = float(cur["close"])
    atr = float(cur["atr"])

    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR-ը հաշվարկել չի հաջողվել")

    side, reason = determine_signal(prev, cur, hour)

    entry = sl = tp1 = tp2 = None

    if side == "LONG":
        entry = price
        sl = entry - ATR_MULTIPLIER * atr
        tp1 = entry + ATR_MULTIPLIER * atr
        tp2 = entry + TP2_MULTIPLIER * atr

    elif side == "SHORT":
        entry = price
        sl = entry + ATR_MULTIPLIER * atr
        tp1 = entry - ATR_MULTIPLIER * atr
        tp2 = entry - TP2_MULTIPLIER * atr

    return {
        "time": int(cur["time"]),
        "side": side,
        "price": price,
        "rsi": float(cur["rsi"]),
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


# ========== ԱԶԴԱՆՇԱՆԻ ՏԵՔՍՏ ==========

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
            "Հստակ մուտքի պայման դեռ չկա։"
        )

    return (
        f"🚨 BTC/USDT — {s['side']} ԱԶԴԱՆՇԱՆ\n\n"
        f"Հաշվարկային մուտք՝ {s['entry']:.2f} USDT\n"
        f"Stop Loss՝ {s['sl']:.2f}\n"
        f"Take Profit 1՝ {s['tp1']:.2f}\n"
        f"Take Profit 2՝ {s['tp2']:.2f}\n\n"
        f"RSI (15m)՝ {s['rsi']:.2f}\n"
        f"ATR (15m)՝ {s['atr']:.2f}\n"
        f"Պատճառ՝ {s['reason']}\n\n"
        "⚠️ Սա ազդանշան է, ոչ թե երաշխավորված շահույթ։\n"
        "Իրական մուտքի գինը կարող է տարբերվել։"
    )


# ========== BACKTEST ==========

def run_backtest(days=30):
    if not 7 <= days <= 90:
        raise ValueError("Ընտրիր 7-ից 90 օր")

    m = indicators(fetch_history("15m", days))
    h = indicators(fetch_history("1h", days))

    # 1h մոմը հասանելի է միայն փակվելուց հետո
    h = h.copy()
    h["hour_closed"] = h["time"] + 3600
    h = h.rename(columns={
        "ema9": "hour_ema9",
        "ema21": "hour_ema21",
    })

    m["signal_time"] = m["time"] + 900

    data = pd.merge_asof(
        m.sort_values("signal_time"),
        h[
            ["hour_closed", "hour_ema9", "hour_ema21"]
        ].sort_values("hour_closed"),
        left_on="signal_time",
        right_on="hour_closed",
        direction="backward",
    ).reset_index(drop=True)

    trades = []
    i = 1

    while i < len(data) - 1:
        prev = data.iloc[i - 1]
        cur = data.iloc[i]

        needed = [
            cur["atr"],
            cur["rsi"],
            cur["hour_ema9"],
            cur["hour_ema21"],
        ]

        if any(pd.isna(v) for v in needed):
            i += 1
            continue

        # Նույն ազդանշանային կանոններն ենք օգտագործում
        hour = {
            "ema9": cur["hour_ema9"],
            "ema21": cur["hour_ema21"],
        }
        side, _ = determine_signal(prev, cur, hour)

        if not side:
            i += 1
            continue

        entry = float(cur["close"])
        atr = float(cur["atr"])
        distance = ATR_MULTIPLIER * atr

        if side == "LONG":
            sl = entry - distance
            tp = entry + distance
        else:
            sl = entry + distance
            tp = entry - distance

        outcome = None
        exit_price = None
        exit_index = None

        # Գործարքը ստուգում ենք հաջորդ փակ մոմերից սկսած
        for j in range(i + 1, len(data)):
            candle = data.iloc[j]

            if side == "LONG":
                hit_sl = candle["low"] <= sl
                hit_tp = candle["high"] >= tp
            else:
                hit_sl = candle["high"] >= sl
                hit_tp = candle["low"] <= tp

            # Եթե նույն մոմում երկու մակարդակն էլ հասանելի են,
            # պահպանողական հաշվարկով Stop Loss-ն է առաջինը։
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_index = j
                break

            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_index = j
                break

        if outcome is None:
            # Չփակված գործարքը չենք հաշվում և նոր դիրք չենք բացում
            break

        if side == "LONG":
            gross_pct = (exit_price - entry) / entry * 100
        else:
            gross_pct = (entry - exit_price) / entry * 100

        net_pct = gross_pct - ROUND_TRIP_FEE_PCT

        trades.append({
            "side": side,
            "outcome": outcome,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
        })

        # Հաջորդ գործարքը միայն նախորդի ավարտից հետո
        i = exit_index + 1

    total = len(trades)
    wins = sum(t["outcome"] == "WIN" for t in trades)
    losses = sum(t["outcome"] == "LOSS" for t in trades)

    gross_sum = sum(t["gross_pct"] for t in trades)
    net_sum = sum(t["net_pct"] for t in trades)

    win_rate = wins / total * 100 if total else 0

    long_count = sum(t["side"] == "LONG" for t in trades)
    short_count = sum(t["side"] == "SHORT" for t in trades)

    return (
        f"📊 BTC/USDT BACKTEST — {days} օր\n\n"
        f"Փակ գործարքներ՝ {total}\n"
        f"LONG՝ {long_count}\n"
        f"SHORT՝ {short_count}\n"
        f"Շահութաբեր՝ {wins}\n"
        f"Վնասով՝ {losses}\n"
        f"Հաջողության տոկոս՝ {win_rate:.1f}%\n\n"
        f"Համախառն արդյունքների գումար՝ {gross_sum:.2f}%\n"
        f"Մոտավոր արդյունք վճարներից հետո՝ {net_sum:.2f}%\n\n"
        "Մուտք՝ ազդանշանի մոմի փակման գնով։\n"
        "SL/TP՝ 1.5 ATR։ Նույն մոմում SL-ն ու TP-ն "
        "հասնելու դեպքում հաշվվում է SL։\n"
        f"Վճարների ենթադրություն՝ {ROUND_TRIP_FEE_PCT:.2f}% "
        "յուրաքանչյուր փակ գործարքի համար։\n"
        "Արդյունքների տոկոսները գումարված են, ոչ թե "
        "կապիտալի բարդ տոկոսով հաշվարկված։\n"
        "Սահքը և իրական կատարման ուշացումը ներառված չեն։"
    )


# ========== TELEGRAM ՀՐԱՄԱՆՆԵՐ ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )

    await update.message.reply_text(
        "Բոտը աշխատում է։\n"
        "Ավտոմատ ազդանշանները միացված են։\n\n"
        "/signal — ստուգել հիմա\n"
        "/backtest — վերջին 30 օրվա փորձարկում\n"
        "/backtest 7 — վերջին 7 օրվա փորձարկում"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("chat_ids", set()).add(
        update.effective_chat.id
    )

    try:
        result = await asyncio.to_thread(analyze)
        await update.message.reply_text(format_result(result))
    except Exception:
        log.exception("Signal check failed")
        await update.message.reply_text(
            "Տվյալների սխալ։ Ստուգիր Fadehost-ի Logs-ը։"
        )


async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        days = 30

        if context.args:
            days = int(context.args[0])

        if not 7 <= days <= 90:
            await update.message.reply_text(
                "Օգտագործիր 7-ից 90 օր։ Օրինակ՝ /backtest 30"
            )
            return

        await update.message.reply_text(
            f"Սկսեցի BTC/USDT-ի {days} օրվա backtest-ը։ "
            "Տվյալների ներբեռնումը կարող է մի փոքր տևել։"
        )

        result = await asyncio.to_thread(run_backtest, days)
        await update.message.reply_text(result)

    except ValueError as e:
        await update.message.reply_text(
            f"Սխալ պարամետր։ Օրինակ՝ /backtest 30\n{e}"
        )
    except Exception:
        log.exception("Backtest failed")
        await update.message.reply_text(
            "Backtest-ը ձախողվեց։ Fadehost-ի Logs-ում "
            "ստուգիր սխալի մանրամասները։"
        )


# ========== ԱՎՏՈՄԱՏ ԱԶԴԱՆՇԱՆՆԵՐ ==========

async def periodic_alerts(app: Application):
    last_time = None

    while True:
        try:
            result = await asyncio.to_thread(analyze)

            if last_time is None:
                # Սկզբում միայն գրանցում ենք վերջին փակ մոմը
                last_time = result["time"]

            elif result["time"] != last_time:
                last_time = result["time"]

                if result["side"]:
                    message = format_result(result)
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
                            log.exception("Telegram send failed")

        except Exception:
            log.exception("Automatic signal check failed")

        await asyncio.sleep(CHECK_SECONDS)


async def post_init(app: Application):
    app.create_task(periodic_alerts(app))


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable-ը բացակայում է"
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("backtest", backtest))

    app.run_polling()


if __name__ == "__main__":
    main()


import os
import time
import asyncio
import logging

import requests
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# BTC/USDT Futures Telegram signal bot.
# Signals only: the bot does NOT place real exchange orders.

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
FEE_RATE = float(os.getenv("FUTURES_FEE_RATE", "0.0005"))
LIVE_STRATEGY = os.getenv("LIVE_STRATEGY", "STRICT").strip().upper()

GATE_URL = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
CONTRACT = "BTC_USDT"

ATR_MULTIPLIER = 1.5
MIN_TP1_NET_PCT = 0.004
POLL_SECONDS = 60

CANDLE_SECONDS = {
    "15m": 900,
    "1h": 3600,
}

STRATEGIES = ("OLD", "PROTECT", "STRICT")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("btc-signal-bot")

CHAT_IDS = set()
LAST_ALERT_CANDLE = None


def gate_candles(interval, limit=300, start=None, end=None):
    params = {
        "contract": CONTRACT,
        "interval": interval,
    }

    if start is not None and end is not None:
        params.update({
            "from": int(start),
            "to": int(end),
        })
    else:
        params["limit"] = min(int(limit), 1000)

    response = requests.get(
        GATE_URL,
        params=params,
        timeout=25,
    )
    response.raise_for_status()

    rows = response.json()
    data = []

    for row in rows:
        try:
            data.append({
                "time": int(row["t"]),
                "open": float(row["o"]),
                "high": float(row["h"]),
                "low": float(row["l"]),
                "close": float(row["c"]),
                "volume": float(row.get("v", 0)),
            })
        except (KeyError, TypeError, ValueError):
            continue

    columns = [
        "time", "open", "high", "low", "close", "volume"
    ]

    if not data:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(data)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def closed_candles(interval, limit=300):
    df = gate_candles(interval, limit=limit)

    if df.empty:
        return df

    seconds = CANDLE_SECONDS[interval]
    now = int(time.time())

    return df[
        df["time"] + seconds <= now
    ].reset_index(drop=True)


def history_candles(interval, days):
    seconds = CANDLE_SECONDS[interval]
    now = int(time.time())
    cursor = now - int(days * 86400)
    frames = []

    while cursor < now:
        end = min(now, cursor + seconds * 900)

        df = gate_candles(
            interval,
            start=cursor,
            end=end,
        )

        if not df.empty:
            df = df[
                (df["time"] >= cursor)
                & (df["time"] <= end)
                & (df["time"] + seconds <= now)
            ]

            if not df.empty:
                frames.append(df)
                cursor = int(df["time"].max()) + seconds
            else:
                cursor = end + seconds
        else:
            cursor = end + seconds

        time.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=[
            "time", "open", "high", "low", "close", "volume"
        ])

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def indicators(df):
    df = df.copy()
    close = df["close"]

    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    delta = close.diff()
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
    rsi = 100 - 100 / (1 + rs)

    # Correct RSI edge cases:
    # no losses -> RSI 100; no gains -> RSI 0;
    # no gains and no losses -> RSI 50.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0), 50
    )

    df["rsi"] = rsi

    previous_close = close.shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


def signal_for(previous, current, hour, strategy):
    required = [
        previous["ema9"],
        previous["ema21"],
        current["ema9"],
        current["ema21"],
        current["rsi"],
        hour["ema9"],
        hour["ema21"],
    ]

    if any(pd.isna(value) for value in required):
        return "NO TRADE"

    cross_up = (
        previous["ema9"] <= previous["ema21"]
        and current["ema9"] > current["ema21"]
    )

    cross_down = (
        previous["ema9"] >= previous["ema21"]
        and current["ema9"] < current["ema21"]
    )

    reclaim_long = (
        previous["close"] <= previous["ema9"]
        and current["close"] > current["ema9"]
        and current["ema9"] > current["ema21"]
    )

    reclaim_short = (
        previous["close"] >= previous["ema9"]
        and current["close"] < current["ema9"]
        and current["ema9"] < current["ema21"]
    )

    if strategy == "STRICT":
        long_min, long_max = 53, 65
        short_min, short_max = 35, 47
    else:
        long_min, long_max = 50, 68
        short_min, short_max = 32, 50

    rsi = float(current["rsi"])

    if (
        hour["ema9"] > hour["ema21"]
        and long_min < rsi < long_max
        and (cross_up or reclaim_long)
    ):
        return "LONG"

    if (
        hour["ema9"] < hour["ema21"]
        and short_min < rsi < short_max
        and (cross_down or reclaim_short)
    ):
        return "SHORT"

    return "NO TRADE"


def calculate_levels(side, entry, atr):
    if entry <= 0 or atr <= 0:
        return None

    risk_distance = ATR_MULTIPLIER * atr

    if side == "LONG":
        sl = entry - risk_distance

        if sl <= 0:
            return None

        risk_fraction = (
            (entry - sl) + FEE_RATE * (entry + sl)
        ) / entry

        tp1_for_2r = (
            entry * (1 + FEE_RATE)
            + 2 * entry * risk_fraction
        ) / (1 - FEE_RATE)

        tp2_for_3r = (
            entry * (1 + FEE_RATE)
            + 3 * entry * risk_fraction
        ) / (1 - FEE_RATE)

        minimum_tp1 = (
            entry * (1 + FEE_RATE)
            + entry * MIN_TP1_NET_PCT
        ) / (1 - FEE_RATE)

        tp1 = max(tp1_for_2r, minimum_tp1)
        tp2 = max(
            tp2_for_3r,
            tp1 * (1 + MIN_TP1_NET_PCT),
        )

    else:
        sl = entry + risk_distance

        risk_fraction = (
            (sl - entry) + FEE_RATE * (entry + sl)
        ) / entry

        tp1_for_2r = (
            entry * (1 - FEE_RATE)
            - 2 * entry * risk_fraction
        ) / (1 + FEE_RATE)

        tp2_for_3r = (
            entry * (1 - FEE_RATE)
            - 3 * entry * risk_fraction
        ) / (1 + FEE_RATE)

        minimum_tp1 = (
            entry * (1 - FEE_RATE)
            - entry * MIN_TP1_NET_PCT
        ) / (1 + FEE_RATE)

        tp1 = min(tp1_for_2r, minimum_tp1)
        tp2 = min(
            tp2_for_3r,
            tp1 * (1 - MIN_TP1_NET_PCT),
        )

        if tp1 <= 0 or tp2 <= 0:
            return None

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_fraction": risk_fraction,
    }


def breakeven_price(side, entry):
    if side == "LONG":
        return entry * (1 + FEE_RATE) / (1 - FEE_RATE)

    return entry * (1 - FEE_RATE) / (1 + FEE_RATE)


def analyze_market():
    m15 = closed_candles("15m", 300)
    h1 = closed_candles("1h", 300)

    if len(m15) < 50 or len(h1) < 50:
        raise RuntimeError(
            "Gate-ից բավարար փակված մոմեր չեն ստացվել։"
        )

    m15 = indicators(m15)
    h1 = indicators(h1)

    current = m15.iloc[-1]
    previous = m15.iloc[-2]

    signal_close_time = int(current["time"]) + 900

    eligible_hours = h1[
        h1["time"] + 3600 <= signal_close_time
    ]

    if eligible_hours.empty:
        raise RuntimeError(
            "Չկա համապատասխան փակված 1h մոմ։"
        )

    hour = eligible_hours.iloc[-1]

    side = signal_for(
        previous, current, hour, LIVE_STRATEGY
    )

    result = {
        "signal": side,
        "entry": float(current["close"]),
        "rsi": float(current["rsi"]),
        "ema9_15": float(current["ema9"]),
        "ema21_15": float(current["ema21"]),
        "ema9_1h": float(hour["ema9"]),
        "ema21_1h": float(hour["ema21"]),
        "atr": float(current["atr"]),
        "candle_time": int(current["time"]),
    }

    if side in ("LONG", "SHORT"):
        level_data = calculate_levels(
            side, result["entry"], result["atr"]
        )

        if level_data:
            result.update(level_data)
        else:
            result["signal"] = "NO TRADE"

    return result


def format_signal(data):
    lines = [
        f"BTC/USDT Futures — {data['signal']}",
        f"Հղումային գին՝ {data['entry']:.2f} USDT",
        f"RSI (15m)՝ {data['rsi']:.2f}",
        (
            f"EMA 9/21 (15m)՝ "
            f"{data['ema9_15']:.2f} / {data['ema21_15']:.2f}"
        ),
        (
            f"EMA 9/21 (1h)՝ "
            f"{data['ema9_1h']:.2f} / {data['ema21_1h']:.2f}"
        ),
        f"ATR (15m)՝ {data['atr']:.2f}",
        f"Կոմիսիա՝ {FEE_RATE * 100:.4f}% մեկ կողմի համար",
        f"Ակտիվ ռազմավարություն՝ {LIVE_STRATEGY}",
    ]

    if data["signal"] in ("LONG", "SHORT"):
        lines.extend([
            "",
            f"SL՝ {data['sl']:.2f}",
            f"TP1՝ {data['tp1']:.2f} (դիրքի 50%)",
            f"TP2՝ {data['tp2']:.2f} (մնացած 50%)",
        ])
    else:
        lines.extend([
            "",
            "Պայմանները չեն համընկնում․ NO TRADE։",
        ])

    lines.extend([
        "",
        "Սա հղումային հաշվարկ է, ոչ երաշխավորված մուտք։",
        "Բոտը պատվերներ չի տեղադրում։",
    ])

    return "\n".join(lines)


def net_return(side, entry, exit_price):
    if side == "LONG":
        gross = (exit_price - entry) / entry
    else:
        gross = (entry - exit_price) / entry

    fees = FEE_RATE * (entry + exit_price) / entry
    return gross - fees


def simulate_trade(df, index, side, strategy):
    # Closed signal candle; enter at the next candle's open.
    if index + 1 >= len(df):
        return None

    entry = float(df.iloc[index + 1]["open"])
    atr = float(df.iloc[index]["atr"])

    level_data = calculate_levels(side, entry, atr)

    if not level_data:
        return None

    sl = level_data["sl"]
    tp1 = level_data["tp1"]
    tp2 = level_data["tp2"]
    risk = level_data["risk_fraction"]

    tp1_reached = False
    pnl = 0.0

    for j in range(index + 1, len(df)):
        candle = df.iloc[j]
        high = float(candle["high"])
        low = float(candle["low"])

        stop = (
            breakeven_price(side, entry)
            if tp1_reached and strategy == "PROTECT"
            else sl
        )

        if side == "LONG":
            stop_hit = low <= stop
            tp1_hit = high >= tp1
            tp2_hit = high >= tp2
        else:
            stop_hit = high >= stop
            tp1_hit = low <= tp1
            tp2_hit = low <= tp2

        # Conservative assumption: if stop and target are both touched
        # in one candle, process the stop first.
        if stop_hit:
            if tp1_reached:
                pnl += 0.5 * net_return(side, entry, stop)
                outcome = (
                    "TP1_BE"
                    if strategy == "PROTECT"
                    else "TP1_SL"
                )
            else:
                pnl += net_return(side, entry, stop)
                outcome = "SL"

            return {
                "side": side,
                "r": pnl / risk,
                "pct": pnl * 100,
                "outcome": outcome,
                "exit_index": j,
            }

        if tp2_hit:
            if not tp1_reached:
                pnl += 0.5 * net_return(side, entry, tp1)

            pnl += 0.5 * net_return(side, entry, tp2)

            return {
                "side": side,
                "r": pnl / risk,
                "pct": pnl * 100,
                "outcome": "TP2",
                "exit_index": j,
            }

        if tp1_hit and not tp1_reached:
            pnl += 0.5 * net_return(side, entry, tp1)
            tp1_reached = True

    return None


def max_drawdown_r(trades):
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for trade in trades:
        equity += trade["r"]
        peak = max(peak, equity)
        max_drawdown = max(
            max_drawdown, peak - equity
        )

    return max_drawdown


def run_backtest(days=30):
    days = max(7, min(int(days), 90))

    m15 = history_candles("15m", days + 10)
    h1 = history_candles("1h", days + 10)

    if len(m15) < 100 or len(h1) < 50:
        return "Բավարար պատմական տվյալներ չկան։"

    m15 = indicators(m15)
    h1 = indicators(h1)

    m15["close_time"] = m15["time"] + 900
    h1["hour_close_time"] = h1["time"] + 3600

    hour_columns = h1[
        ["hour_close_time", "ema9", "ema21"]
    ].rename(columns={
        "ema9": "ema9_1h",
        "ema21": "ema21_1h",
    })

    df = pd.merge_asof(
        m15.sort_values("close_time"),
        hour_columns.sort_values("hour_close_time"),
        left_on="close_time",
        right_on="hour_close_time",
        direction="backward",
    )

    cutoff = int(time.time()) - days * 86400
    reports = []

    for strategy in STRATEGIES:
        trades = []
        index = 1

        while index < len(df) - 1:
            row = df.iloc[index]

            if int(row["close_time"]) < cutoff:
                index += 1
                continue

            previous = df.iloc[index - 1]
            hour = {
                "ema9": row["ema9_1h"],
                "ema21": row["ema21_1h"],
            }

            side = signal_for(
                previous, row, hour, strategy
            )

            if side == "NO TRADE":
                index += 1
                continue

            trade = simulate_trade(
                df, index, side, strategy
            )

            if trade is None:
                break

            trades.append(trade)
            index = trade["exit_index"] + 1

        if not trades:
            reports.append(
                f"📊 {strategy}: ավարտված գործարքներ չեն գտնվել։"
            )
            continue

        wins = sum(t["r"] > 0 for t in trades)
        losses = sum(t["r"] < 0 for t in trades)
        longs = [t for t in trades if t["side"] == "LONG"]
        shorts = [t for t in trades if t["side"] == "SHORT"]

        reports.append(
            f"📊 {strategy} — {days} օր\n"
            f"Գործարքներ՝ {len(trades)} | "
            f"շահութաբեր՝ {wins} | կորուստով՝ {losses}\n"
            f"Հաղթանակների տոկոս՝ "
            f"{wins / len(trades) * 100:.2f}%\n"
            f"Միջին արդյունք՝ "
            f"{sum(t['r'] for t in trades) / len(trades):+.2f}R / գործարք\n"
            f"Ընդհանուր արդյունք՝ "
            f"{sum(t['r'] for t in trades):+.2f}R\n"
            f"Առավելագույն անկում՝ "
            f"{max_drawdown_r(trades):.2f}R\n"
            f"Գործարքների տոկոսների գումար՝ "
            f"{sum(t['pct'] for t in trades):+.2f}%\n"
            f"LONG՝ {len(longs)} | "
            f"շահութաբեր՝ {sum(t['r'] > 0 for t in longs)}\n"
            f"SHORT՝ {len(shorts)} | "
            f"շահութաբեր՝ {sum(t['r'] > 0 for t in shorts)}\n"
            f"TP2՝ {sum(t['outcome'] == 'TP2' for t in trades)} | "
            f"SL՝ {sum(t['outcome'] == 'SL' for t in trades)} | "
            f"TP1-ից հետո SL՝ "
            f"{sum(t['outcome'] == 'TP1_SL' for t in trades)} | "
            f"breakeven՝ "
            f"{sum(t['outcome'] == 'TP1_BE' for t in trades)}"
        )

    return (
        "\n\n".join(reports)
        + "\n\nԿոմիսիան մոտավոր ներառված է։"
        "\nՍայթաքումն ու funding-ը ներառված չեն։"
        "\nԲեքթեստը շահույթի երաշխիք չէ։"
    )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)

    await update.effective_message.reply_text(
        "Բոտը միացված է։ Այս զրույցը գրանցված է ազդանշանների համար։\n\n"
        "/signal — ընթացիկ վերլուծություն\n"
        "/backtest 30 — երեք տարբերակի բեքթեստ\n"
        "/status — բոտի վիճակ\n"
        "/stopalerts — դադարեցնել ազդանշանները\n\n"
        "Բոտը պատվերներ չի տեղադրում։"
    )


async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Ստուգում եմ Gate Futures տվյալները…"
    )

    try:
        data = await asyncio.to_thread(analyze_market)
        await update.effective_message.reply_text(
            format_signal(data)
        )
    except Exception as error:
        log.exception("Signal analysis failed")
        await update.effective_message.reply_text(
            f"Վերլուծության սխալ՝ {str(error)[:250]}"
        )


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = 30

    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            await update.effective_message.reply_text(
                "Օգտագործիր՝ /backtest 30 կամ /backtest 90"
            )
            return

    days = max(7, min(days, 90))

    await update.effective_message.reply_text(
        f"Հաշվում եմ վերջին {days} օրվա բեքթեստը…"
    )

    try:
        result = await asyncio.to_thread(run_backtest, days)

        for start in range(0, len(result), 3800):
            await update.effective_message.reply_text(
                result[start:start + 3800]
            )
    except Exception as error:
        log.exception("Backtest failed")
        await update.effective_message.reply_text(
            f"Բեքթեստի սխալ՝ {str(error)[:250]}"
        )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Բոտի գործընթացն աշխատում է։\n"
        f"Ակտիվ ֆիլտր՝ {LIVE_STRATEGY}\n"
        f"Գրանցված զրույցներ՝ {len(CHAT_IDS)}\n"
        f"Կոմիսիա մեկ կողմի համար՝ {FEE_RATE * 100:.4f}%"
    )


async def stopalerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.discard(update.effective_chat.id)

    await update.effective_message.reply_text(
        "Այս զրույցի ավտոմատ ազդանշաններն անջատված են։\n"
        "/start գրիր՝ կրկին միացնելու համար։"
    )


async def alert_loop(app):
    global LAST_ALERT_CANDLE

    while True:
        try:
            data = await asyncio.to_thread(analyze_market)
            candle_time = data["candle_time"]

            # On startup, wait for the next closed 15m candle.
            if LAST_ALERT_CANDLE is None:
                LAST_ALERT_CANDLE = candle_time

            elif candle_time != LAST_ALERT_CANDLE:
                LAST_ALERT_CANDLE = candle_time

                if data["signal"] in ("LONG", "SHORT"):
                    message = (
                        "🔔 ՆՈՐ ԱԶԴԱՆՇԱՆ\n\n"
                        + format_signal(data)
                    )

                    for chat_id in list(CHAT_IDS):
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                            )
                        except Exception:
                            log.exception(
                                "Telegram message send failed"
                            )

        except Exception:
            log.exception("Automatic signal check failed")

        await asyncio.sleep(POLL_SECONDS)


async def post_init(app):
    app.create_task(alert_loop(app))


def main():
    if not TOKEN:
        raise RuntimeError(
            "Fadehost-ում TELEGRAM_BOT_TOKEN փոփոխականը սահմանված չէ։"
        )

    if LIVE_STRATEGY not in STRATEGIES:
        raise ValueError(
            "LIVE_STRATEGY-ը պետք է լինի OLD, PROTECT կամ STRICT։"
        )

    if not 0 <= FEE_RATE < 0.1:
        raise ValueError(
            "FUTURES_FEE_RATE-ը սխալ է։ Օրինակ՝ 0.0005։"
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("stopalerts", stopalerts_cmd))

    log.info(
        "BTC/USDT signal bot starting; strategy=%s",
        LIVE_STRATEGY,
    )

    app.run_polling()


if __name__ == "__main__":
    main()


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

# ==================================================
# ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ
# ==================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Gate.io USDT Futures
GATE_URL = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
CONTRACT = "BTC_USDT"

# Կոմիսիա՝ ՄԵԿ կողմի համար։
# 0.0005 = 0.05%։ Փոխիր ըստ քո իրական Gate fee tier-ի։
FEE_RATE = float(os.getenv("FUTURES_FEE_RATE", "0.0005"))

# SL = ATR * այս գործակիցը
ATR_MULTIPLIER = 1.5

# TP-ները զուտ 2R և 3R են՝ երկու կողմի
# կոմիսիան հաշվելուց հետո։
TP1_R = 2.0
TP2_R = 3.0

# Բեքթեստում դիրքի 50%-ը փակվում է TP1-ում,
# մնացած 50%-ը՝ TP2-ում։
TP1_CLOSE_PERCENT = 0.50

POLL_SECONDS = 60
CANDLE_SECONDS = {
    "15m": 900,
    "1h": 3600,
}

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Սրանք հիշվում են միայն ծրագրի աշխատանքի ընթացքում։
CHAT_IDS = set()
LAST_ALERT_CANDLE = None


# ==================================================
# GATE FUTURES ՏՎՅԱԼՆԵՐ
# ==================================================

def request_candles(interval, limit=200, start=None, end=None):
    params = {
        "contract": CONTRACT,
        "interval": interval,
        "limit": min(int(limit), 1000),
    }

    if start is not None:
        params["from"] = int(start)

    if end is not None:
        params["to"] = int(end)

    response = requests.get(
        GATE_URL,
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("Gate-ը անսպասելի պատասխան է վերադարձրել")

    parsed = []

    for row in rows:
        try:
            parsed.append({
                "time": int(row["t"]),
                "open": float(row["o"]),
                "high": float(row["h"]),
                "low": float(row["l"]),
                "close": float(row["c"]),
                "volume": float(row.get("v", 0)),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not parsed:
        return pd.DataFrame(
            columns=[
                "time", "open", "high",
                "low", "close", "volume"
            ]
        )

    df = pd.DataFrame(parsed)
    df = df.drop_duplicates(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)


def get_closed_candles(interval, limit=200):
    """Վերադարձնում է միայն փակված մոմերը։"""
    df = request_candles(interval, limit=limit)

    if df.empty:
        return df

    now = int(time.time())
    seconds = CANDLE_SECONDS[interval]

    # Բաց մոմը չենք օգտագործում ազդանշանի համար։
    df = df[df["time"] + seconds <= now]

    return df.reset_index(drop=True)


def fetch_history(interval, days):
    """Պատմական մոմերի ներբեռնում՝ էջերով։"""
    seconds = CANDLE_SECONDS[interval]
    now = int(time.time())

    start_time = now - int(days * 86400)
    cursor = start_time
    all_frames = []

    while cursor < now:
        chunk_end = min(
            now,
            cursor + seconds * 999,
        )

        df = request_candles(
            interval,
            limit=1000,
            start=cursor,
            end=chunk_end,
        )

        if df.empty:
            cursor = chunk_end + seconds
            continue

        df = df[
            (df["time"] >= cursor)
            & (df["time"] <= chunk_end)
            & (df["time"] + seconds <= now)
        ]

        if not df.empty:
            all_frames.append(df)
            latest_time = int(df["time"].max())
            cursor = latest_time + seconds
        else:
            cursor = chunk_end + seconds

        # Պաշտպանություն՝ անվերջ ցիկլից։
        if cursor <= start_time:
            break

        time.sleep(0.08)

    if not all_frames:
        return pd.DataFrame(
            columns=[
                "time", "open", "high",
                "low", "close", "volume"
            ]
        )

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["time"])
    result = result.sort_values("time").reset_index(drop=True)

    return result


# ==================================================
# ՑՈՒՑԻՉՆԵՐ
# ==================================================

def indicators(df):
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

    # RSI-ի եզրային դեպքեր
    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0), "rsi"
    ] = 0

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


# ==================================================
# ԱԶԴԱՆՇԱՆԻ ՊԱՅՄԱՆՆԵՐ
# ==================================================

def determine_signal(prev, cur, hour):
    """Վերադարձնում է LONG, SHORT կամ NO TRADE։"""

    values = [
        prev.get("ema9"),
        prev.get("ema21"),
        cur.get("ema9"),
        cur.get("ema21"),
        cur.get("rsi"),
        hour.get("ema9"),
        hour.get("ema21"),
    ]

    if any(pd.isna(v) for v in values):
        return "NO TRADE"

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
        and cur["ema9"] > cur["ema21"]
    )

    reclaim_short = (
        prev["close"] >= prev["ema9"]
        and cur["close"] < cur["ema9"]
        and cur["ema9"] < cur["ema21"]
    )

    rsi = cur["rsi"]

    if (
        hour["ema9"] > hour["ema21"]
        and 45 < rsi < 68
        and (cross_up or reclaim_long)
    ):
        return "LONG"

    if (
        hour["ema9"] < hour["ema21"]
        and 32 < rsi < 55
        and (cross_down or reclaim_short)
    ):
        return "SHORT"

    return "NO TRADE"


# ==================================================
# SL, TP1, TP2՝ ԿՈՄԻՍԻԱՆ ՀԱՇՎԵԼՈՎ
# ==================================================

def calculate_levels(signal, entry, atr):
    """
    SL-ը ATR-ով է։

    R = մեկ միավորի զուտ կորուստը SL-ի դեպքում,
    ներառյալ մուտքի և ելքի կոմիսիան։

    TP1 = զուտ +2R
    TP2 = զուտ +3R

    Չեն ներառվում slippage-ը և funding-ը։
    """

    if signal not in ("LONG", "SHORT"):
        return None

    if entry <= 0 or atr <= 0:
        return None

    if not 0 <= FEE_RATE < 0.1:
        raise ValueError(
            "FUTURES_FEE_RATE-ը ստուգիր։ "
            "Օրինակ՝ 0.0005 նշանակում է 0.05%։"
        )

    distance = ATR_MULTIPLIER * atr
    fee = FEE_RATE

    if signal == "LONG":
        sl = entry - distance

        if sl <= 0:
            return None

        # SL-ի զուտ կորուստը՝ մեկ միավորի հաշվով։
        net_risk = (entry - sl) + fee * (entry + sl)

        # Զուտ շահույթը TP-ում = R բազմապատիկ։
        tp1 = (
            entry * (1 + fee) + TP1_R * net_risk
        ) / (1 - fee)

        tp2 = (
            entry * (1 + fee) + TP2_R * net_risk
        ) / (1 - fee)

    else:
        sl = entry + distance

        net_risk = (sl - entry) + fee * (entry + sl)

        tp1 = (
            entry * (1 - fee) - TP1_R * net_risk
        ) / (1 + fee)

        tp2 = (
            entry * (1 - fee) - TP2_R * net_risk
        ) / (1 + fee)

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "net_risk": net_risk,
    }


# ==================================================
# ԱՅԺՄՅԱՆ ԱԶԴԱՆՇԱՆԻ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ
# ==================================================

def analyze():
    m15 = get_closed_candles("15m", limit=200)
    h1 = get_closed_candles("1h", limit=200)

    if len(m15) < 40 or len(h1) < 40:
        raise ValueError(
            "Բավարար փակված մոմեր չկան։ Փորձիր մի փոքր ուշ։"
        )

    m15 = indicators(m15)
    h1 = indicators(h1)

    prev = m15.iloc[-2]
    cur = m15.iloc[-1]
    hour = h1.iloc[-1]

    signal = determine_signal(prev, cur, hour)

    entry = float(cur["close"])
    atr = float(cur["atr"])

    result = {
        "signal": signal,
        "entry": entry,
        "rsi": float(cur["rsi"]),
        "ema9_15": float(cur["ema9"]),
        "ema21_15": float(cur["ema21"]),
        "ema9_1h": float(hour["ema9"]),
        "ema21_1h": float(hour["ema21"]),
        "atr": atr,
        "candle_time": int(cur["time"]),
    }

    if signal in ("LONG", "SHORT"):
        levels = calculate_levels(signal, entry, atr)

        if levels is None:
            result["signal"] = "NO TRADE"
        else:
            result.update(levels)

    return result


def format_signal(data):
    lines = [
        f"BTC/USDT Futures — {data['signal']}",
        f"Մուտք՝ {data['entry']:.2f} USDT",
        f"RSI (15m)՝ {data['rsi']:.2f}",
        (
            f"EMA 9/21 (15m)՝ "
            f"{data['ema9_15']:.2f} / "
            f"{data['ema21_15']:.2f}"
        ),
        (
            f"EMA 9/21 (1h)՝ "
            f"{data['ema9_1h']:.2f} / "
            f"{data['ema21_1h']:.2f}"
        ),
        f"ATR (15m)՝ {data['atr']:.2f}",
        f"Կոմիսիա՝ {FEE_RATE * 100:.4f}% մեկ կողմի համար",
    ]

    if data["signal"] in ("LONG", "SHORT"):
        lines.extend([
            "",
            f"SL՝ {data['sl']:.2f}",
            f"TP1՝ {data['tp1']:.2f} (զուտ 2R)",
            f"TP2՝ {data['tp2']:.2f} (զուտ 3R)",
            "",
            "TP1-ում փակել դիրքի 50%-ը՝ ըստ ռազմավարության։",
            "TP2-ը նախատեսված է մնացած 50%-ի համար։",
        ])
    else:
        lines.extend([
            "",
            "Հիմա պայմանները չեն համընկնում։",
            "Մի բացիր գործարք միայն ազդանշան ստանալու համար։",
        ])

    lines.extend([
        "",
        "Տվյալները Gate USDT Futures-ից են։",
        "Բոտը պատվերներ չի տեղադրում։",
    ])

    return "\n".join(lines)


# ==================================================
# TELEGRAM ՀՐԱՄԱՆՆԵՐ
# ==================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id
    CHAT_IDS.add(chat_id)

    await update.message.reply_text(
        "Bot is working!\n\n"
        "Հրամաններ՝\n"
        "/signal — ընթացիկ վերլուծություն\n"
        "/backtest 30 — վերջին 30 օրվա բեքթեստ\n\n"
        "Բոտը կարող է նաև ուղարկել նոր LONG/SHORT "
        "ազդանշաններ։ Այն ինքնուրույն գործարք չի բացում։"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Ստուգում եմ Gate Futures տվյալները..."
    )

    try:
        data = await asyncio.to_thread(analyze)
        await update.message.reply_text(format_signal(data))
    except Exception as exc:
        logger.exception("Signal analysis failed")
        await update.message.reply_text(
            "Վերլուծության սխալ։ Փորձիր կրկին։\n"
            f"Պատճառը՝ {str(exc)[:250]}"
        )


# ==================================================
# ԲԵՔԹԵՍՏ
# ==================================================

def run_backtest(days=30):
    days = max(7, min(int(days), 90))

    now = int(time.time())
    target_start = now - days * 86400

    # Լրացուցիչ պատմություն՝ EMA-ների հաշվարկի համար։
    history_days = days + 10

    m15 = fetch_history("15m", history_days)
    h1 = fetch_history("1h", history_days)

    if len(m15) < 100 or len(h1) < 40:
        return (
            "Բավարար պատմական տվյալներ չկան։ "
            "Փորձիր ավելի ուշ։"
        )

    m15 = indicators(m15)
    h1 = indicators(h1)

    # 15m ազդանշանը հայտնի է մոմի փակվելուց հետո։
    m15["signal_time"] = m15["time"] + 900

    # 1h EMA-ները հասանելի են միայն ժամային մոմի փակումից հետո։
    h1["hour_close_time"] = h1["time"] + 3600

    hour_data = h1[
        ["hour_close_time", "ema9", "ema21"]
    ].rename(columns={
        "ema9": "ema9_1h",
        "ema21": "ema21_1h",
    })

    merged = pd.merge_asof(
        m15.sort_values("signal_time"),
        hour_data.sort_values("hour_close_time"),
        left_on="signal_time",
        right_on="hour_close_time",
        direction="backward",
    )

    trades = []
    i = 1

    while i < len(merged) - 1:
        prev = merged.iloc[i - 1]
        cur = merged.iloc[i]

        if int(cur["signal_time"]) < target_start:
            i += 1
            continue

        hour = {
            "ema9": cur["ema9_1h"],
            "ema21": cur["ema21_1h"],
        }

        signal = determine_signal(prev, cur, hour)

        if signal == "NO TRADE":
            i += 1
            continue

        entry = float(cur["close"])
        atr = float(cur["atr"])

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        levels = calculate_levels(signal, entry, atr)

        if not levels:
            i += 1
            continue

        sl = levels["sl"]
        tp1 = levels["tp1"]
        tp2 = levels["tp2"]
        net_risk = levels["net_risk"]

        first_tp_hit = False
        outcome_r = None
        exit_index = None

        for j in range(i + 1, len(merged)):
            candle = merged.iloc[j]

            high = float(candle["high"])
            low = float(candle["low"])

            if signal == "LONG":
                sl_hit = low <= sl
                tp1_hit = high >= tp1
                tp2_hit = high >= tp2
            else:
                sl_hit = high >= sl
                tp1_hit = low <= tp1
                tp2_hit = low <= tp2

            # Եթե նույն մոմում SL և TP են հատվել,
            # պահպանողական հաշվարկով SL-ը նախորդում է։
            if sl_hit:
                if first_tp_hit:
                    # 50%-ը TP1-ում փակվել է,
                    # մնացած 50%-ը՝ SL-ում։
                    outcome_r = (
                        TP1_R * TP1_CLOSE_PERCENT
                        - (1 - TP1_CLOSE_PERCENT)
                    )
                else:
                    outcome_r = -1.0

                exit_index = j
                break

            if tp2_hit:
                # Կեսը՝ TP1-ում, կեսը՝ TP2-ում։
                outcome_r = (
                    TP1_R * TP1_CLOSE_PERCENT
                    + TP2_R * (1 - TP1_CLOSE_PERCENT)
                )
                exit_index = j
                break

            if tp1_hit and not first_tp_hit:
                first_tp_hit = True

        # Չփակված գործարքը չենք հաշվում որպես ավարտված։
        if exit_index is None:
            break

        risk_pct = (net_risk / entry) * 100
        net_pct = outcome_r * risk_pct

        trades.append({
            "signal": signal,
            "outcome_r": outcome_r,
            "net_pct": net_pct,
        })

        # Նոր գործարք՝ միայն նախորդի ավարտից հետո։
        i = exit_index + 1

    if not trades:
        return (
            f"Բեքթեստ՝ {days} օր\n"
            "Ավարտված գործարքներ չեն գտնվել։\n"
            "Սա չի նշանակում, որ ռազմավարությունը շահութաբեր է "
            "կամ անշահավետ․ ստուգիր տվյալները և պայմանները։"
        )

    total = len(trades)
    wins = sum(t["outcome_r"] > 0 for t in trades)
    losses = sum(t["outcome_r"] < 0 for t in trades)

    longs = sum(t["signal"] == "LONG" for t in trades)
    shorts = sum(t["signal"] == "SHORT" for t in trades)

    win_rate = wins / total * 100
    total_r = sum(t["outcome_r"] for t in trades)
    total_pct = sum(t["net_pct"] for t in trades)

    return (
        f"BTC/USDT Futures — Բեքթեստ ({days} օր)\n\n"
        f"Ավարտված գործարքներ՝ {total}\n"
        f"LONG՝ {longs} | SHORT՝ {shorts}\n"
        f"Շահութաբեր՝ {wins} | Կորուստով՝ {losses}\n"
        f"Հաղթանակների տոկոս՝ {win_rate:.2f}%\n"
        f"Ընդհանուր արդյունք՝ {total_r:+.2f}R\n"
        f"Գործարքների տոկոսների գումար՝ {total_pct:+.2f}%\n\n"
        "TP1-ում փակվում է 50%, TP2-ում՝ մնացած 50%-ը։\n"
        "Կոմիսիան ներառված է մոտավոր հաշվարկում։\n"
        "Չեն ներառվում slippage-ը, funding-ը և կապիտալի "
        "բարդ տոկոսադրույքով աճը։\n"
        "Անցյալի արդյունքը ապագա շահույթ չի երաշխավորում։"
    )


async def backtest_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    days = 30

    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "Օգտագործիր՝ /backtest 30"
            )
            return

    days = max(7, min(days, 90))

    await update.message.reply_text(
        f"Հաշվում եմ վերջին {days} օրվա բեքթեստը։ "
        "Սա կարող է մի փոքր ժամանակ տևել..."
    )

    try:
        result = await asyncio.to_thread(
            run_backtest, days
        )
        await update.message.reply_text(result)
    except Exception as exc:
        logger.exception("Backtest failed")
        await update.message.reply_text(
            "Բեքթեստի ժամանակ սխալ առաջացավ։\n"
            f"Պատճառը՝ {str(exc)[:250]}"
        )


# ==================================================
# ԱՎՏՈՄԱՏ ԱԶԴԱՆՇԱՆՆԵՐ
# ==================================================

async def periodic_alerts(app: Application):
    global LAST_ALERT_CANDLE

    while True:
        try:
            data = await asyncio.to_thread(analyze)
            candle_time = data["candle_time"]

            if LAST_ALERT_CANDLE is None:
                # Սկզբում չենք ուղարկում հին ազդանշանը։
                LAST_ALERT_CANDLE = candle_time

            elif candle_time != LAST_ALERT_CANDLE:
                LAST_ALERT_CANDLE = candle_time

                if data["signal"] in ("LONG", "SHORT"):
                    message = (
                        "ՆՈՐ ԱԶԴԱՆՇԱՆ\n\n"
                        + format_signal(data)
                    )

                    for chat_id in list(CHAT_IDS):
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                            )
                        except Exception:
                            logger.exception(
                                "Telegram send failed for %s",
                                chat_id,
                            )

        except Exception:
            logger.exception("Automatic alert check failed")

        await asyncio.sleep(POLL_SECONDS)


async def post_init(app: Application):
    app.create_task(periodic_alerts(app))


# ==================================================
# ՄԵԿՆԱՐԿ
# ==================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN-ը սահմանված չէ։ "
            "Ավելացրու այն Fadehost-ի Environment Variables-ում։"
        )

    if not 0 <= FEE_RATE < 0.1:
        raise ValueError(
            "FUTURES_FEE_RATE-ը սխալ է։ "
            "Օրինակ՝ 0.0005 = 0.05%։"
        )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start_command)
    )
    app.add_handler(
        CommandHandler("signal", signal_command)
    )
    app.add_handler(
        CommandHandler("backtest", backtest_command)
    )

    logger.info("BTC/USDT Futures bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()


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

# =========================================================
# CONFIGURATION
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GATE_URL = (
    "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
)
CONTRACT = "BTC_USDT"

FEE_RATE = float(os.getenv("FUTURES_FEE_RATE", "0.0005"))
ATR_MULTIPLIER = 1.5

TP1_R = 2.0
TP2_R = 3.0
TP1_CLOSE_PERCENT = 0.50
MIN_NET_TP1_PCT = 0.004

AUTO_CHECK_SECONDS = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

http = requests.Session()
http.headers.update({"User-Agent": "BTC-USDT-Signal-Bot/1.0"})

subscribed_chats = set()
last_alert_candle = None


# =========================================================
# HTTP AND GATE DATA
# =========================================================

def get_json(url, params=None, timeout=20):
    response = http.get(
        url,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def normalize_candles(data):
    rows = []

    for item in data:
        if isinstance(item, dict):
            rows.append({
                "time": int(item.get("t", item.get("time", 0))),
                "open": float(item["o"]),
                "high": float(item["h"]),
                "low": float(item["l"]),
                "close": float(item["c"]),
                "volume": float(item.get("v", 0)),
            })

        elif isinstance(item, (list, tuple)) and len(item) >= 6:
            # Gate-ի սովորական futures candle list-ի հերթականությունը:
            # time, volume, close, high, low, open
            rows.append({
                "time": int(item[0]),
                "volume": float(item[1]),
                "close": float(item[2]),
                "high": float(item[3]),
                "low": float(item[4]),
                "open": float(item[5]),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Gate API-ն մոմեր չի վերադարձրել։")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def request_candles(interval="15m", limit=200):
    """Վերջին մոմերը՝ ընթացիկ ազդանշանի համար."""

    data = get_json(
        GATE_URL,
        params={
            "contract": CONTRACT,
            "interval": interval,
            "limit": limit,
        },
    )

    return normalize_candles(data)


def request_candles_range(start_ts, end_ts, interval):
    """
    Պատմական մոմերը վերցնում է փոքր ժամանակահատվածներով։
    from/to-ի հետ limit չի ուղարկվում։
    """

    chunk_seconds = 5 * 24 * 60 * 60
    all_data = []
    chunk_start = int(start_ts)

    while chunk_start < end_ts:
        chunk_end = min(
            chunk_start + chunk_seconds,
            int(end_ts),
        )

        data = get_json(
            GATE_URL,
            params={
                "contract": CONTRACT,
                "interval": interval,
                "from": chunk_start,
                "to": chunk_end,
            },
        )

        if not isinstance(data, list):
            raise ValueError(
                f"Gate API-ի անսպասելի պատասխանը՝ {data}"
            )

        all_data.extend(data)

        # Հաջորդ հարցումը սկսվում է հաջորդ վայրկյանից։
        chunk_start = chunk_end + 1
        time.sleep(0.15)

    if not all_data:
        raise ValueError(
            f"Gate-ը {interval} ժամանակահատվածի մոմեր չի վերադարձրել։"
        )

    return normalize_candles(all_data)


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False,
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

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


# =========================================================
# SL / TP CALCULATION
# =========================================================

def calculate_levels(signal, entry, atr):
    """
    TP1-ի մոտավոր զուտ թիրախը՝ առնվազն 0.4%,
    կամ 2R՝ որն ավելի հեռու է։

    TP2-ը՝ առնվազն 3R և TP1-ից ավելի հեռու։

    R-ի չափման հիմքը սկզբնական գնի ռիսկն է՝ entry-ից SL։
    Մաքուր արդյունքի հաշվարկում ներառվում են մուտքի ու ելքի վճարները։
    """

    if signal not in ("LONG", "SHORT"):
        return None

    if entry <= 0 or atr <= 0:
        return None

    if not 0 <= FEE_RATE < 0.1:
        raise ValueError("FUTURES_FEE_RATE-ի արժեքը սխալ է։")

    risk_distance = ATR_MULTIPLIER * atr

    if signal == "LONG":
        sl = entry - risk_distance

        if sl <= 0:
            return None

        # Լուծում ենք՝ exit-entry-fee*(entry+exit) = entry*0.004
        min_tp1 = (
            entry * (1 + FEE_RATE + MIN_NET_TP1_PCT)
            / (1 - FEE_RATE)
        )

        tp1_2r = entry + TP1_R * risk_distance
        tp1 = max(tp1_2r, min_tp1)

        tp2_3r = entry + TP2_R * risk_distance
        tp2 = max(
            tp2_3r,
            tp1 + entry * MIN_NET_TP1_PCT,
        )

    else:
        sl = entry + risk_distance

        min_tp1 = (
            entry * (1 - FEE_RATE - MIN_NET_TP1_PCT)
            / (1 + FEE_RATE)
        )

        tp1_2r = entry - TP1_R * risk_distance
        tp1 = min(tp1_2r, min_tp1)

        tp2_3r = entry - TP2_R * risk_distance
        tp2 = min(
            tp2_3r,
            tp1 - entry * MIN_NET_TP1_PCT,
        )

    if min(sl, tp1, tp2) <= 0:
        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_distance": risk_distance,
    }


# =========================================================
# SIGNAL CONDITIONS
# =========================================================

def signal_for_closed_candle(df15, df1h, index):
    """
    Օգտագործում է միայն արդեն փակված 15m և 1h մոմերը։
    index-ը այն 15m մոմի դիրքն է, որտեղից սկսվում է մուտքի ստուգումը։
    """

    signal_idx = index - 1

    if signal_idx < 0:
        return None

    candle = df15.iloc[signal_idx]
    signal_time = int(candle["time"])

    # Ընտրում ենք միայն այն 1h մոմերը, որոնք փակված են
    # ազդանշանի մոմի փակման պահին։
    signal_close_time = signal_time + 15 * 60

    closed_1h = df1h[
        df1h["time"] + 60 * 60 <= signal_close_time
    ]

    if closed_1h.empty:
        return None

    h1 = closed_1h.iloc[-1]

    required = [
        candle["ema9"],
        candle["ema21"],
        candle["rsi"],
        candle["atr"],
        h1["ema9"],
        h1["ema21"],
    ]

    if any(pd.isna(value) for value in required):
        return None

    long_condition = (
        candle["ema9"] > candle["ema21"]
        and h1["ema9"] > h1["ema21"]
        and 50 <= candle["rsi"] <= 68
    )

    short_condition = (
        candle["ema9"] < candle["ema21"]
        and h1["ema9"] < h1["ema21"]
        and 32 <= candle["rsi"] <= 50
    )

    if long_condition:
        side = "LONG"
    elif short_condition:
        side = "SHORT"
    else:
        return None

    return {
        "side": side,
        "signal_time": signal_time,
        "rsi": float(candle["rsi"]),
        "ema9": float(candle["ema9"]),
        "ema21": float(candle["ema21"]),
        "atr": float(candle["atr"]),
    }


def analyze_current_market():
    df15 = add_indicators(request_candles("15m", 200))
    df1h = add_indicators(request_candles("1h", 200))

    # Վերջին մոմը կարող է դեռ չփակված լինել։
    idx = len(df15) - 1

    if idx < 30:
        raise ValueError("15m տվյալները բավարար չեն։")

    candle = df15.iloc[-2]

    signal = signal_for_closed_candle(df15, df1h, idx)

    result = {
        "side": signal["side"] if signal else "NO TRADE",
        "price": float(candle["close"]),
        "rsi": float(candle["rsi"]),
        "ema9": float(candle["ema9"]),
        "ema21": float(candle["ema21"]),
        "candle_time": int(candle["time"]),
        "reason": (
            "Պայմանները չեն հաստատվել"
            if signal is None else ""
        ),
    }

    if signal:
        levels = calculate_levels(
            signal["side"],
            result["price"],
            signal["atr"],
        )

        if levels is None:
            result["side"] = "NO TRADE"
            result["reason"] = "SL/TP մակարդակները հաշվարկել չհաջողվեց։"
        else:
            result.update(levels)

    return result


# =========================================================
# BACKTEST
# =========================================================

def net_pnl_per_unit(side, entry, exit_price):
    """Մեկ միավորի զուտ PnL՝ մուտքի և ելքի վճարներով։"""

    if side == "LONG":
        gross = exit_price - entry
    else:
        gross = entry - exit_price

    fees = FEE_RATE * (entry + exit_price)
    return gross - fees


def simulate_trade(df15, start_index, side, entry, levels):
    """
    Գործարքը ստուգում է ապագա մոմերով։
    Եթե նույն մոմում SL և TP են հասնում, նախընտրում է SL-ը։
    TP1-ում փակում է 50%, TP2-ում՝ մնացած 50%-ը։
    """

    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    risk = levels["risk_distance"]

    first_tp_hit = False

    for j in range(start_index, len(df15)):
        candle = df15.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        if side == "LONG":
            sl_hit = low <= sl
            tp1_hit = high >= tp1
            tp2_hit = high >= tp2
        else:
            sl_hit = high >= sl
            tp1_hit = low <= tp1
            tp2_hit = low <= tp2

        # Եթե SL-ն ու TP-ն նույն մոմում են՝ SL առաջինը։
        if sl_hit:
            if first_tp_hit:
                pnl = (
                    TP1_CLOSE_PERCENT
                    * net_pnl_per_unit(side, entry, tp1)
                    + (1 - TP1_CLOSE_PERCENT)
                    * net_pnl_per_unit(side, entry, sl)
                )
            else:
                pnl = net_pnl_per_unit(side, entry, sl)

            return {
                "exit_index": j,
                "exit_reason": "SL_AFTER_TP1" if first_tp_hit else "SL",
                "pnl_per_unit": pnl,
                "R": pnl / risk,
                "net_pct": pnl / entry * 100,
                "tp1_hit": first_tp_hit,
            }

        if tp2_hit:
            pnl = (
                TP1_CLOSE_PERCENT
                * net_pnl_per_unit(side, entry, tp1)
                + (1 - TP1_CLOSE_PERCENT)
                * net_pnl_per_unit(side, entry, tp2)
            )

            return {
                "exit_index": j,
                "exit_reason": "TP2",
                "pnl_per_unit": pnl,
                "R": pnl / risk,
                "net_pct": pnl / entry * 100,
                "tp1_hit": True,
            }

        if tp1_hit and not first_tp_hit:
            first_tp_hit = True

    # Վերջին մոմին դեռ բաց գործարքը չենք ներկայացնում որպես ավարտված։
    return None


def run_backtest(days=30):
    if days < 1 or days > 365:
        raise ValueError("Օրերի միջակայքը պետք է լինի 1-ից 365։")

    now = int(time.time())
    start = now - days * 24 * 60 * 60

    # Լրացուցիչ պատմություն՝ EMA/RSI/ATR ցուցիչների տաքացման համար։
    warmup_start = start - 5 * 24 * 60 * 60

    df15 = request_candles_range(
        warmup_start, now, "15m"
    )
    df1h = request_candles_range(
        warmup_start, now, "1h"
    )

    df15 = add_indicators(df15).reset_index(drop=True)
    df1h = add_indicators(df1h).reset_index(drop=True)

    trades = []
    i = 1

    while i < len(df15) - 1:
        entry_candle = df15.iloc[i]
        entry_time = int(entry_candle["time"])

        if entry_time < start:
            i += 1
            continue

        # Մուտք՝ ազդանշանի մոմին հաջորդող մոմի բացման գնով։
        signal = signal_for_closed_candle(df15, df1h, i)

        if signal is None:
            i += 1
            continue

        side = signal["side"]
        entry = float(entry_candle["open"])

        levels = calculate_levels(
            side,
            entry,
            signal["atr"],
        )

        if levels is None:
            i += 1
            continue

        outcome = simulate_trade(
            df15,
            i,
            side,
            entry,
            levels,
        )

        if outcome is None:
            # Բաց գործարքը ավարտվածների մեջ չենք հաշվում։
            break

        trades.append({
            "side": side,
            "entry_time": entry_time,
            "entry": entry,
            "sl": levels["sl"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            **outcome,
        })

        # Չենք թույլատրում մի քանի համընկնող դիրք։
        i = outcome["exit_index"] + 1

    return trades


def format_backtest(trades, days):
    total = len(trades)

    if total == 0:
        return (
            f"BTC/USDT Futures — Բեքթեստ ({days} օր)\n\n"
            "Ավարտված գործարքներ չկան։ Ստուգիր տվյալները "
            "կամ ընտրված ժամանակահատվածը։"
        )

    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]

    winners = [t for t in trades if t["pnl_per_unit"] > 0]
    losers = [t for t in trades if t["pnl_per_unit"] <= 0]

    total_r = sum(t["R"] for t in trades)
    total_pct = sum(t["net_pct"] for t in trades)
    win_rate = len(winners) / total * 100

    long_wins = sum(t["pnl_per_unit"] > 0 for t in longs)
    short_wins = sum(t["pnl_per_unit"] > 0 for t in shorts)

    avg_r = total_r / total
    tp2_count = sum(t["exit_reason"] == "TP2" for t in trades)
    sl_count = sum(t["exit_reason"] == "SL" for t in trades)
    sl_after_tp1_count = sum(
        t["exit_reason"] == "SL_AFTER_TP1" for t in trades
    )

    return (
        f"BTC/USDT Futures — Բեքթեստ ({days} օր)\n\n"
        f"Ավարտված գործարքներ՝ {total}\n"
        f"LONG՝ {len(longs)} | SHORT՝ {len(shorts)}\n"
        f"LONG հաղթանակներ՝ {long_wins}/{len(longs) if longs else 0}\n"
        f"SHORT հաղթանակներ՝ {short_wins}/{len(shorts) if shorts else 0}\n\n"
        f"Շահութաբեր՝ {len(winners)} | Կորուստով՝ {len(losers)}\n"
        f"Հաղթանակների տոկոս՝ {win_rate:.2f}%\n"
        f"Միջին արդյունք՝ {avg_r:+.2f}R / գործարք\n"
        f"Ընդհանուր արդյունք՝ {total_r:+.2f}R\n"
        f"Գործարքների տոկոսների գումար՝ {total_pct:+.2f}%\n\n"
        f"TP2 ելքեր՝ {tp2_count}\n"
        f"SL ելքեր՝ {sl_count}\n"
        f"TP1-ից հետո SL՝ {sl_after_tp1_count}\n\n"
        f"TP1-ում փակվում է 50%, TP2-ում՝ մնացած 50%-ը։\n"
        f"Կոմիսիա՝ {FEE_RATE * 100:.3f}% մեկ կողմի համար։\n"
        "Slippage-ը և funding-ը ներառված չեն։\n"
        "R-ը հաշվարկվում է սկզբնական գնի ռիսկի նկատմամբ։\n"
        "Տոկոսների գումարը հաշվի կապիտալի եկամտաբերությունը չէ։\n"
        "Սա պատմական թեստ է, շահույթի երաշխիք չէ։"
    )


# =========================================================
# TELEGRAM COMMANDS
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribed_chats.add(update.effective_chat.id)

    await update.message.reply_text(
        "Bot is working!\n\n"
        "BTC/USDT Futures ազդանշանների բոտ։\n\n"
        "/signal — ընթացիկ ազդանշան\n"
        "/backtest — վերջին 30 օրվա բեքթեստ\n"
        "/backtest 7 — վերջին 7 օրվա բեքթեստ\n"
        "/stopalerts — անջատել ավտոմատ ծանուցումները\n\n"
        "Բոտը Gate-ում գործարքներ ինքնուրույն չի բացում։"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text("Վերլուծում եմ BTC/USDT-ը…")

    try:
        s = await asyncio.to_thread(analyze_current_market)

        lines = [
            f"BTC/USDT Futures — {s['side']}",
            f"Գին՝ {s['price']:.2f} USDT",
            f"RSI (15m)՝ {s['rsi']:.2f}",
            f"EMA 9 (15m)՝ {s['ema9']:.2f}",
            f"EMA 21 (15m)՝ {s['ema21']:.2f}",
        ]

        if s["side"] in ("LONG", "SHORT"):
            lines += [
                "",
                f"Entry՝ {s['entry']:.2f}",
                f"Stop Loss՝ {s['sl']:.2f}",
                f"TP1՝ {s['tp1']:.2f}",
                f"TP2՝ {s['tp2']:.2f}",
                "",
                "TP1-ը 0.4% զուտ թիրախ է կամ 2R՝ որն ավելի հեռու է։",
                "TP1-ում փակել 50%, TP2-ում՝ մնացած 50%-ը։",
            ]
        else:
            lines.append("Պատճառ՝ " + s["reason"])

        lines += [
            "",
            "Կոմիսիան մոտավոր հաշվարկված է։",
            "Սա ազդանշան է, ոչ թե շահույթի երաշխիք։",
        ]

        await update.message.reply_text("\n".join(lines))

    except Exception as exc:
        logger.exception("Signal error")
        await update.message.reply_text(
            "Չհաջողվեց վերլուծել շուկան։\n"
            f"Սխալ՝ {str(exc)[:300]}"
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
                "Օգտագործիր /backtest կամ /backtest 30"
            )
            return

    if not 1 <= days <= 365:
        await update.message.reply_text(
            "Օրերի քանակը պետք է լինի 1-ից 365։"
        )
        return

    await update.message.reply_text(
        f"Սկսում եմ {days} օրվա բեքթեստը։ "
        "Տվյալների ներբեռնումը կարող է տևել։"
    )

    try:
        trades = await asyncio.to_thread(run_backtest, days)
        message = format_backtest(trades, days)
        await update.message.reply_text(message)

    except Exception as exc:
        logger.exception("Backtest failed")
        await update.message.reply_text(
            "Բեքթեստը ձախողվեց։ Ստուգիր Fadehost Logs-ը։\n"
            f"Սխալ՝ {str(exc)[:300]}"
        )


async def stop_alerts(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    subscribed_chats.discard(update.effective_chat.id)

    await update.message.reply_text(
        "Ավտոմատ ծանուցումներն անջատված են։ "
        "/start՝ նորից միացնելու համար։"
    )


# =========================================================
# AUTOMATIC ALERTS
# =========================================================

async def auto_alert_loop(app: Application):
    global last_alert_candle

    while True:
        try:
            if subscribed_chats:
                s = await asyncio.to_thread(analyze_current_market)

                if (
                    s["side"] in ("LONG", "SHORT")
                    and s["candle_time"] != last_alert_candle
                ):
                    lines = [
                        f"BTC/USDT Futures — {s['side']}",
                        f"Գին՝ {s['price']:.2f}",
                        f"RSI՝ {s['rsi']:.2f}",
                        f"Entry՝ {s['entry']:.2f}",
                        f"SL՝ {s['sl']:.2f}",
                        f"TP1՝ {s['tp1']:.2f}",
                        f"TP2՝ {s['tp2']:.2f}",
                        "",
                        "TP1-ում՝ 50%, TP2-ում՝ մնացած 50%։",
                        "Սա ազդանշան է, ոչ թե շահույթի երաշխիք։",
                    ]

                    message = "\n".join(lines)

                    for chat_id in list(subscribed_chats):
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                            )
                        except Exception:
                            logger.exception(
                                "Alert could not be sent to %s",
                                chat_id,
                            )

                    last_alert_candle = s["candle_time"]

        except Exception:
            logger.exception("Automatic signal check failed")

        await asyncio.sleep(AUTO_CHECK_SECONDS)


async def post_init(app: Application):
    app.create_task(auto_alert_loop(app))


# =========================================================
# MAIN
# =========================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN-ը բացակայում է։ "
            "Սահմանիր այն Fadehost Environment Variables-ում։"
        )

    if not 0 <= FEE_RATE < 0.1:
        raise ValueError("FUTURES_FEE_RATE-ի արժեքը սխալ է։")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("stopalerts", stop_alerts))

    logger.info("BTC/USDT signal bot started")
    app.run_polling()


if __name__ == "__main__":
    main()

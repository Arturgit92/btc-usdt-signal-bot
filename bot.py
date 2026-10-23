
import os
import json
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
# SETTINGS
# ==================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GATE_URL = (
    "https://api.gateio.ws/api/v4/"
    "futures/usdt/candlesticks"
)
CONTRACT = "BTC_USDT"

FEE_RATE = float(
    os.getenv("FUTURES_FEE_RATE", "0.0005")
)

CHECK_SECONDS = 300          # Ստուգում՝ 5 րոպեն մեկ
ATR_MULTIPLIER = 1.5
TP1_R = 2.0
TP2_R = 3.0
TP1_PERCENT = 0.50

STATE_FILE = "bot_state.json"

STRATEGIES = {
    "BASELINE": "Հին պայմաններ",
    "BREAKEVEN": "TP1-ից հետո պաշտպանություն",
    "STRICT": "Խիստ RSI ֆիլտր",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================
# SAVED BOT STATE
# ==================================================

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(app):
    state = {
        "alert_chat_id": app.bot_data.get("alert_chat_id"),
        "last_checked_candle": app.bot_data.get(
            "last_checked_candle"
        ),
        "last_alert_candle": app.bot_data.get(
            "last_alert_candle"
        ),
    }

    try:
        with open(STATE_FILE, "w", encoding="utf-8") as file:
            json.dump(state, file)
    except OSError:
        logger.exception("Չհաջողվեց պահպանել bot_state.json")


def restore_state(app):
    state = load_state()

    for key in (
        "alert_chat_id",
        "last_checked_candle",
        "last_alert_candle",
    ):
        if state.get(key) is not None:
            app.bot_data[key] = state[key]


# ==================================================
# GATE FUTURES CANDLES
# ==================================================

def request_candles(interval, start_ts, end_ts):
    all_rows = []
    chunk_seconds = 5 * 24 * 60 * 60
    cursor = int(start_ts)

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)

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
            raise ValueError(
                f"Gate API-ն անսպասելի պատասխան է տվել: {rows}"
            )

        all_rows.extend(rows)
        cursor = chunk_end
        time.sleep(0.1)

    parsed = []

    for row in all_rows:
        try:
            # Gate futures array:
            # [timestamp, volume, close, high, low, open]
            if isinstance(row, dict):
                parsed.append({
                    "time": int(row["t"]),
                    "volume": float(row.get("v", 0)),
                    "close": float(row["c"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "open": float(row["o"]),
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
        raise ValueError(
            f"Gate-ից {interval} մոմեր չեն ստացվել"
        )

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
        min_periods=14,
        adjust=False,
    ).mean()

    return df


def get_market_data(days=10):
    now = int(datetime.now(timezone.utc).timestamp())

    # Նախորդ 7 օրը օգտագործվում է ցուցիչները հաշվարկելու համար։
    start = now - (days + 7) * 86400

    df15 = request_candles("15m", start, now)
    df1h = request_candles("1h", start, now)

    # Հեռացնում ենք դեռ չփակված մոմերը։
    df15 = df15[
        df15["time"] + 15 * 60 <= now
    ].copy()

    df1h = df1h[
        df1h["time"] + 60 * 60 <= now
    ].copy()

    df15 = add_indicators(df15).reset_index(drop=True)
    df1h = add_indicators(df1h).reset_index(drop=True)

    if len(df15) < 40 or len(df1h) < 30:
        raise ValueError("Ցուցիչների համար բավարար տվյալներ չկան")

    return df15, df1h


# ==================================================
# SIGNAL CONDITIONS
# ==================================================

def get_signal(df15, df1h, signal_idx, strategy):
    """
    signal_idx-ը փակված 15m մոմի ինդեքսն է։
    Բեքթեստում մուտքը մոդելավորվում է հաջորդ մոմի բացման գնով։
    """

    if signal_idx < 30 or signal_idx >= len(df15):
        return None

    row15 = df15.iloc[signal_idx]

    if pd.isna(row15["atr"]) or row15["atr"] <= 0:
        return None

    close_time = int(row15["time"]) + 15 * 60

    # Օգտագործում ենք միայն արդեն փակված ժամային մոմը։
    eligible = df1h[
        df1h["time"] + 3600 <= close_time
    ]

    if eligible.empty:
        return None

    row1h = eligible.iloc[-1]

    if pd.isna(row1h["ema9"]) or pd.isna(row1h["ema21"]):
        return None

    trend_up = (
        row15["ema9"] > row15["ema21"]
        and row1h["ema9"] > row1h["ema21"]
    )

    trend_down = (
        row15["ema9"] < row15["ema21"]
        and row1h["ema9"] < row1h["ema21"]
    )

    rsi = float(row15["rsi"])

    if strategy == "STRICT":
        long_ok = trend_up and 53 <= rsi <= 65
        short_ok = trend_down and 35 <= rsi <= 47
    else:
        long_ok = trend_up and 50 <= rsi <= 68
        short_ok = trend_down and 32 <= rsi <= 50

    if long_ok:
        side = "LONG"
    elif short_ok:
        side = "SHORT"
    else:
        return None

    return {
        "side": side,
        "time": int(row15["time"]),
        "price": float(row15["close"]),
        "rsi": rsi,
        "atr": float(row15["atr"]),
    }


# ==================================================
# STOP LOSS / TAKE PROFIT / FEES
# ==================================================

def calculate_levels(entry, side, atr):
    risk = atr * ATR_MULTIPLIER

    if risk <= 0:
        raise ValueError("ATR ռիսկը վավեր չէ")

    if side == "LONG":
        sl = entry - risk

        # TP1-ի նվազագույն թիրախը՝ մոտ 0.4% զուտ շահույթ,
        # եթե մուտքի և ելքի գնի վրա նույն fee-ն է կիրառվում։
        min_tp1 = (
            entry * (1 + FEE_RATE + 0.004)
            / (1 - FEE_RATE)
        )

        tp1 = max(
            entry + TP1_R * risk,
            min_tp1,
        )

        tp2 = max(
            entry + TP2_R * risk,
            tp1 + entry * 0.004,
        )

    else:
        sl = entry + risk

        min_tp1 = (
            entry * (1 - FEE_RATE - 0.004)
            / (1 + FEE_RATE)
        )

        tp1 = min(
            entry - TP1_R * risk,
            min_tp1,
        )

        tp2 = min(
            entry - TP2_R * risk,
            tp1 - entry * 0.004,
        )

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
    }


def leg_pnl(entry, exit_price, side, fraction):
    if side == "LONG":
        gross = exit_price - entry
    else:
        gross = entry - exit_price

    fees = (entry + exit_price) * FEE_RATE

    return fraction * (gross - fees)


def fee_adjusted_breakeven(entry, side):
    if side == "LONG":
        return entry * (1 + FEE_RATE) / (1 - FEE_RATE)

    return entry * (1 - FEE_RATE) / (1 + FEE_RATE)


# ==================================================
# TRADE SIMULATION FOR BACKTEST
# ==================================================

def simulate_trade(df15, entry_idx, signal, strategy):
    entry_candle = df15.iloc[entry_idx]
    entry = float(entry_candle["open"])
    side = signal["side"]

    levels = calculate_levels(
        entry,
        side,
        signal["atr"],
    )

    original_sl = levels["sl"]
    active_sl = original_sl
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    risk = levels["risk"]

    tp1_hit = False
    pnl = 0.0
    outcome = "OPEN"
    exit_idx = entry_idx

    for j in range(entry_idx, len(df15)):
        candle = df15.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])
        exit_idx = j

        if side == "LONG":
            sl_hit = low <= active_sl
            tp1_hit_now = high >= tp1
            tp2_hit_now = high >= tp2
        else:
            sl_hit = high >= active_sl
            tp1_hit_now = low <= tp1
            tp2_hit_now = low <= tp2

        # Մինչև TP1. եթե նույն մոմում SL և TP են հասել,
        # պահպանողական հաշվարկով SL-ն առաջնահերթ է։
        if not tp1_hit:
            if sl_hit:
                pnl += leg_pnl(entry, active_sl, side, 1.0)
                outcome = "SL"
                break

            if tp2_hit_now:
                pnl += leg_pnl(
                    entry, tp1, side, TP1_PERCENT
                )
                pnl += leg_pnl(
                    entry, tp2, side, 1 - TP1_PERCENT
                )
                outcome = "TP2"
                break

            if tp1_hit_now:
                pnl += leg_pnl(
                    entry, tp1, side, TP1_PERCENT
                )
                tp1_hit = True

                if strategy == "BREAKEVEN":
                    active_sl = fee_adjusted_breakeven(
                        entry, side
                    )

                # Նույն մոմում TP1-ից հետո stop-ի հասնելը
                # համարում ենք stop-ի ելք՝ պահպանողական ձևով։
                if side == "LONG":
                    new_stop_hit = low <= active_sl
                else:
                    new_stop_hit = high >= active_sl

                if new_stop_hit:
                    pnl += leg_pnl(
                        entry,
                        active_sl,
                        side,
                        1 - TP1_PERCENT,
                    )

                    outcome = (
                        "TP1_BE"
                        if strategy == "BREAKEVEN"
                        else "TP1_SL"
                    )
                    break

                if tp2_hit_now:
                    pnl += leg_pnl(
                        entry,
                        tp2,
                        side,
                        1 - TP1_PERCENT,
                    )
                    outcome = "TP2"
                    break

        else:
            # TP1-ն արդեն հասել է. մնացել է դիրքի 50%-ը։
            if sl_hit:
                pnl += leg_pnl(
                    entry,
                    active_sl,
                    side,
                    1 - TP1_PERCENT,
                )

                outcome = (
                    "TP1_BE"
                    if strategy == "BREAKEVEN"
                    else "TP1_SL"
                )
                break

            if tp2_hit_now:
                pnl += leg_pnl(
                    entry,
                    tp2,
                    side,
                    1 - TP1_PERCENT,
                )
                outcome = "TP2"
                break

    # Չավարտված գործարքը արդյունքների մեջ չենք ներառում։
    if outcome == "OPEN":
        return None

    return {
        "side": side,
        "entry": entry,
        "pnl": pnl,
        "R": pnl / risk,
        "outcome": outcome,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
    }


def run_backtest(df15, df1h, days, strategy):
    now = int(datetime.now(timezone.utc).timestamp())
    start_ts = now - days * 86400

    trades = []
    i = 40

    while i < len(df15) - 1:
        if int(df15.iloc[i]["time"]) < start_ts:
            i += 1
            continue

        # Մուտքի պահին օգտագործում ենք նախորդ փակված մոմը։
        signal = get_signal(
            df15,
            df1h,
            i - 1,
            strategy,
        )

        if signal is None:
            i += 1
            continue

        trade = simulate_trade(
            df15,
            i,
            signal,
            strategy,
        )

        if trade is None:
            break

        trades.append(trade)

        # Միաժամանակ միայն մեկ դիրք է մոդելավորվում։
        i = trade["exit_idx"] + 1

    return trades


def summarize(trades):
    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_R": 0.0,
            "avg_R": 0.0,
            "max_drawdown_R": 0.0,
            "long_count": 0,
            "long_wins": 0,
            "short_count": 0,
            "short_wins": 0,
            "tp2": 0,
            "sl": 0,
            "tp1_sl": 0,
            "tp1_be": 0,
            "sum_pct": 0.0,
        }

    wins = sum(t["pnl"] > 0 for t in trades)
    losses = len(trades) - wins

    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for trade in trades:
        cumulative += trade["R"]
        peak = max(peak, cumulative)
        max_drawdown = max(
            max_drawdown,
            peak - cumulative,
        )

    total_R = sum(t["R"] for t in trades)

    # Սա գործարքների տոկոսների գումարն է, ոչ թե հաշվի եկամտաբերությունը։
    sum_pct = sum(
        t["pnl"] / t["entry"] * 100
        for t in trades
    )

    return {
        "count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100,
        "total_R": total_R,
        "avg_R": total_R / len(trades),
        "max_drawdown_R": max_drawdown,
        "long_count": len(longs),
        "long_wins": sum(t["pnl"] > 0 for t in longs),
        "short_count": len(shorts),
        "short_wins": sum(t["pnl"] > 0 for t in shorts),
        "tp2": sum(t["outcome"] == "TP2" for t in trades),
        "sl": sum(t["outcome"] == "SL" for t in trades),
        "tp1_sl": sum(
            t["outcome"] == "TP1_SL" for t in trades
        ),
        "tp1_be": sum(
            t["outcome"] == "TP1_BE" for t in trades
        ),
        "sum_pct": sum_pct,
    }


def format_backtest(name, days, s):
    return (
        f"📊 {name} — {days} օր\n\n"
        f"Ավարտված գործարքներ՝ {s['count']}\n"
        f"Շահութաբեր՝ {s['wins']} | Կորուստով՝ {s['losses']}\n"
        f"Հաղթանակների տոկոս՝ {s['win_rate']:.2f}%\n"
        f"Միջին արդյունք՝ {s['avg_R']:.2f}R / գործարք\n"
        f"Ընդհանուր արդյունք՝ {s['total_R']:.2f}R\n"
        f"Առավելագույն անկում՝ {s['max_drawdown_R']:.2f}R\n"
        f"Գործարքների տոկոսների գումար՝ {s['sum_pct']:.2f}%\n\n"
        f"LONG՝ {s['long_count']} | շահութաբեր՝ {s['long_wins']}\n"
        f"SHORT՝ {s['short_count']} | շահութաբեր՝ {s['short_wins']}\n\n"
        f"TP2՝ {s['tp2']}\n"
        f"SL՝ {s['sl']}\n"
        f"TP1-ից հետո SL՝ {s['tp1_sl']}\n"
        f"TP1-ից հետո breakeven՝ {s['tp1_be']}\n\n"
        "Կոմիսիան հաշվարկված է մոտավոր՝ մուտքի և ելքի համար։\n"
        "Slippage-ը և funding-ը ներառված չեն։"
    )


# ==================================================
# TELEGRAM COMMANDS
# ==================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    context.application.bot_data["alert_chat_id"] = chat_id
    save_state(context.application)

    await update.message.reply_text(
        "Բոտը աշխատում է։ 🤖\n\n"
        "Ավտոմատ ազդանշաններ՝ միացված\n"
        "Շուկա՝ BTC/USDT Futures\n"
        "Ֆիլտր՝ խիստ RSI + EMA 9/21\n"
        "Ստուգում՝ ամեն 5 րոպեն մեկ\n\n"
        "/signal — ձեռքով ստուգում\n"
        "/backtest 30 — համեմատել 3 ռազմավարությունը\n"
        "/status — բոտի կարգավիճակ\n"
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
        df15, df1h = await asyncio.to_thread(
            get_market_data, 10
        )

        index = len(df15) - 1
        signal = get_signal(
            df15, df1h, index, "STRICT"
        )

        if signal is None:
            row = df15.iloc[index]
            await update.message.reply_text(
                "BTC/USDT — NO TRADE\n\n"
                f"Փակ մոմի գին՝ {row['close']:.2f} USDT\n"
                f"RSI (15m)՝ {row['rsi']:.1f}\n"
                "Խիստ ֆիլտրի պայմանները չեն համընկնում։"
            )
            return

        await update.message.reply_text(
            format_signal(signal)
        )

    except Exception as exc:
        logger.exception("/signal հրամանի սխալ")
        await update.message.reply_text(
            f"Տվյալների սխալ՝ {str(exc)[:250]}"
        )


def format_signal(signal):
    entry = signal["price"]
    side = signal["side"]

    levels = calculate_levels(
        entry,
        side,
        signal["atr"],
    )

    candle_time = datetime.fromtimestamp(
        signal["time"],
        timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"🚨 BTC/USDT FUTURES — {side}\n\n"
        f"Փակված 15m մոմ՝ {candle_time}\n"
        f"Հղումային գին՝ {entry:.2f} USDT\n"
        f"RSI (15m)՝ {signal['rsi']:.1f}\n\n"
        f"Stop Loss՝ {levels['sl']:.2f}\n"
        f"TP1՝ {levels['tp1']:.2f} (փակել 50%)\n"
        f"TP2՝ {levels['tp2']:.2f} (մնացած 50%)\n\n"
        "Ֆիլտր՝ խիստ RSI + EMA 9/21 (15m և 1h)\n"
        "⚠️ Սա ազդանշան է, ոչ թե կատարված գործարք։"
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
                "Ճիշտ հրաման՝ /backtest 30"
            )
            return

    if not 7 <= days <= 90:
        await update.message.reply_text(
            "Օրերի քանակը պետք է լինի 7-ից 90։"
        )
        return

    await update.message.reply_text(
        f"Բեռնում եմ BTC/USDT տվյալները՝ {days} օրվա համար։\n"
        "Խնդրում եմ սպասել։"
    )

    try:
        df15, df1h = await asyncio.to_thread(
            get_market_data, days
        )

        results = []

        for key, label in STRATEGIES.items():
            trades = await asyncio.to_thread(
                run_backtest,
                df15,
                df1h,
                days,
                key,
            )
            stats = summarize(trades)
            results.append((key, label, stats))

            await update.message.reply_text(
                format_backtest(label, days, stats)
            )

        lines = ["📌 ՀԱՄԵՄԱՏՈՒԹՅՈՒՆ\n"]

        for key, label, s in results:
            lines.append(
                f"{label}: {s['total_R']:.2f}R | "
                f"շահումներ՝ {s['win_rate']:.2f}% | "
                f"գործարքներ՝ {s['count']} | "
                f"անկում՝ {s['max_drawdown_R']:.2f}R"
            )

        lines.append(
            "\nՍա պատմական մոդելավորում է, ոչ թե շահույթի երաշխիք։"
        )
        lines.append(
            "Մոմի ներսում գնի շարժման հերթականությունը, "
            "slippage-ը և funding-ը լիարժեք չեն մոդելավորվում։"
        )
        lines.append(
            "Գործարքների տոկոսների գումարը հաշվի իրական "
            "եկամտաբերությունը չէ։"
        )

        await update.message.reply_text("\n".join(lines))

    except Exception as exc:
        logger.exception("/backtest հրամանի սխալ")
        await update.message.reply_text(
            "Բեքթեստը չհաջողվեց։\n"
            f"Սխալ՝ {str(exc)[:300]}"
        )


async def status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = context.application.bot_data.get("alert_chat_id")
    last_alert = context.application.bot_data.get(
        "last_alert_candle"
    )

    await update.message.reply_text(
        "BTC/USDT բոտի կարգավիճակ՝ ԱՇԽԱՏՈՒՄ Է ✅\n"
        f"Ավտոմատ ազդանշանների չաթ՝ "
        f"{'միացված' if chat_id else 'չգրանցված'}\n"
        f"Վերջին ազդանշանի մոմ՝ {last_alert or 'դեռ չկա'}\n"
        "Ֆիլտր՝ խիստ RSI"
    )


async def stopalerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    context.application.bot_data.pop("alert_chat_id", None)
    save_state(context.application)

    await update.message.reply_text(
        "Ավտոմատ ազդանշաններն անջատված են։\n"
        "Միացնելու համար ուղարկիր /start։"
    )


# ==================================================
# AUTOMATIC SIGNAL JOB
# ==================================================

async def automatic_signal_job(
    context: ContextTypes.DEFAULT_TYPE,
):
    app = context.application
    chat_id = app.bot_data.get("alert_chat_id")

    if not chat_id:
        return

    try:
        df15, df1h = await asyncio.to_thread(
            get_market_data, 10
        )

        # Վերջին տողը վերջին փակված մոմն է։
        index = len(df15) - 1

        if index < 30:
            return

        candle_time = int(df15.iloc[index]["time"])

        last_checked = app.bot_data.get(
            "last_checked_candle"
        )

        if last_checked == candle_time:
            return

        # Մոմը նշում ենք որպես ստուգված, որպեսզի
        # նույն մոմը նորից չստուգվի։
        app.bot_data["last_checked_candle"] = candle_time
        save_state(app)

        signal = get_signal(
            df15,
            df1h,
            index,
            "STRICT",
        )

        if signal is None:
            logger.info(
                "NO TRADE: փակված մոմ %s", candle_time
            )
            return

        last_alert = app.bot_data.get("last_alert_candle")

        if last_alert == candle_time:
            return

        await context.bot.send_message(
            chat_id=chat_id,
            text=format_signal(signal),
        )

        app.bot_data["last_alert_candle"] = candle_time
        save_state(app)

        logger.info(
            "Ավտոմատ %s ազդանշան ուղարկվեց",
            signal["side"],
        )

    except Exception:
        logger.exception("Ավտոմատ ստուգման սխալ")


# ==================================================
# START BOT
# ==================================================

def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN փոփոխականը բացակայում է։ "
            "Այն ավելացրու Fadehost-ի Environment Settings-ում։"
        )

    app = Application.builder().token(TOKEN).build()

    restore_state(app)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("stopalerts", stopalerts_command))

    if app.job_queue is None:
        raise RuntimeError(
            "JobQueue-ը հասանելի չէ։ Ստուգիր "
            "requirements.txt-ում python-telegram-bot[job-queue] տողը։"
        )

    app.job_queue.run_repeating(
        automatic_signal_job,
        interval=CHECK_SECONDS,
        first=10,
        name="btc_usdt_automatic_signals",
    )

    logger.info(
        "BTC/USDT Futures բոտը գործարկվում է։"
    )

    app.run_polling()


if __name__ == "__main__":
    main()

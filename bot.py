
import os
import time
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

# =========================
# SETTINGS
# =========================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GATE_URL = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
CONTRACT = "BTC_USDT"

FEE_RATE = float(os.getenv("FUTURES_FEE_RATE", "0.0005"))

ATR_MULTIPLIER = 1.5
TP1_R = 2.0
TP2_R = 3.0
TP1_CLOSE_PERCENT = 0.50

DEFAULT_BACKTEST_DAYS = 30

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================
# GATE CANDLES
# =========================

def request_candles(interval, start_ts, end_ts):
    """Download candles in small chunks to avoid API range limits."""
    all_rows = []
    chunk_seconds = 5 * 24 * 60 * 60
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)

        params = {
            "contract": CONTRACT,
            "interval": interval,
            "from": int(cursor),
            "to": int(chunk_end),
        }

        response = requests.get(
            GATE_URL,
            params=params,
            timeout=25,
        )
        response.raise_for_status()
        rows = response.json()

        if not isinstance(rows, list):
            raise ValueError(f"Unexpected Gate API response: {rows}")

        all_rows.extend(rows)
        cursor = chunk_end

        time.sleep(0.12)

    if not all_rows:
        raise ValueError(f"No candles received for {interval}")

    parsed = []

    for row in all_rows:
        try:
            # Gate futures candle array:
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
        raise ValueError(f"Could not parse candles for {interval}")

    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def download_market_data(days=30):
    now = int(datetime.now(timezone.utc).timestamp())

    # Extra history is needed to initialize EMA and RSI.
    warmup = 7 * 24 * 60 * 60
    start = now - days * 24 * 60 * 60 - warmup

    df15 = request_candles("15m", start, now)
    df1h = request_candles("1h", start, now)

    df15 = add_indicators(df15)
    df1h = add_indicators(df1h)

    return df15, df1h


# =========================
# INDICATORS
# =========================

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
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


# =========================
# SIGNAL CONDITIONS
# =========================

STRATEGIES = {
    "BASELINE": "Հին պայմաններ",
    "BREAKEVEN": "TP1-ից հետո պաշտպանություն",
    "STRICT": "Խիստ RSI ֆիլտր",
}


def get_signal(df15, df1h, signal_idx, strategy):
    """
    signal_idx is a CLOSED 15m candle.
    The entry is made at the NEXT candle open in backtesting.
    """

    if signal_idx < 30:
        return None

    row15 = df15.iloc[signal_idx]

    if pd.isna(row15["atr"]) or row15["atr"] <= 0:
        return None

    signal_close_time = int(row15["time"]) + 15 * 60

    # Use only hourly candles that have fully closed.
    eligible = df1h[
        df1h["time"] + 60 * 60 <= signal_close_time
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
        long_signal = trend_up and 53 <= rsi <= 65
        short_signal = trend_down and 35 <= rsi <= 47
    else:
        long_signal = trend_up and 50 <= rsi <= 68
        short_signal = trend_down and 32 <= rsi <= 50

    if long_signal:
        side = "LONG"
    elif short_signal:
        side = "SHORT"
    else:
        return None

    return {
        "side": side,
        "signal_price": float(row15["close"]),
        "atr": float(row15["atr"]),
        "rsi": rsi,
        "time": int(row15["time"]),
    }


# =========================
# PRICE LEVELS AND PNL
# =========================

def calculate_levels(entry, side, atr):
    risk_distance = atr * ATR_MULTIPLIER

    if risk_distance <= 0:
        raise ValueError("Invalid risk distance")

    if side == "LONG":
        sl = entry - risk_distance

        # At least 0.4% net target after entry/exit fees.
        min_tp1 = entry * (1 + FEE_RATE + 0.004) / (
            1 - FEE_RATE
        )

        tp1 = max(
            entry + TP1_R * risk_distance,
            min_tp1,
        )

        tp2 = max(
            entry + TP2_R * risk_distance,
            tp1 + entry * 0.004,
        )

    else:
        sl = entry + risk_distance

        min_tp1 = entry * (1 - FEE_RATE - 0.004) / (
            1 + FEE_RATE
        )

        tp1 = min(
            entry - TP1_R * risk_distance,
            min_tp1,
        )

        tp2 = min(
            entry - TP2_R * risk_distance,
            tp1 - entry * 0.004,
        )

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_distance": risk_distance,
    }


def leg_pnl(entry, exit_price, side, fraction):
    """Net PnL per 1 unit of starting position, including fees."""
    if side == "LONG":
        gross = exit_price - entry
    else:
        gross = entry - exit_price

    fees = (entry + exit_price) * FEE_RATE

    return fraction * (gross - fees)


def breakeven_stop(entry, side):
    """Exit level that approximately offsets entry and exit fees."""
    if side == "LONG":
        return entry * (1 + FEE_RATE) / (1 - FEE_RATE)

    return entry * (1 - FEE_RATE) / (1 + FEE_RATE)


# =========================
# TRADE SIMULATOR
# =========================

def simulate_trade(df15, entry_idx, signal, strategy):
    entry_candle = df15.iloc[entry_idx]
    entry = float(entry_candle["open"])
    side = signal["side"]

    levels = calculate_levels(entry, side, signal["atr"])

    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    risk_distance = levels["risk_distance"]

    tp1_hit = False
    pnl = 0.0
    outcome = "OPEN"
    exit_idx = entry_idx

    # Simulate future candles only.
    for j in range(entry_idx, len(df15)):
        candle = df15.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])
        exit_idx = j

        if side == "LONG":
            sl_hit = low <= sl
            tp1_hit_now = high >= tp1
            tp2_hit_now = high >= tp2
        else:
            sl_hit = high >= sl
            tp1_hit_now = low <= tp1
            tp2_hit_now = low <= tp2

        # Conservative assumption: if SL and target are both
        # inside the same candle, count SL first.
        if not tp1_hit:
            if sl_hit:
                pnl += leg_pnl(entry, sl, side, 1.0)
                outcome = "SL"
                break

            if tp2_hit_now:
                pnl += leg_pnl(
                    entry, tp1, side, TP1_CLOSE_PERCENT
                )
                pnl += leg_pnl(
                    entry, tp2, side, 1 - TP1_CLOSE_PERCENT
                )
                tp1_hit = True
                outcome = "TP2"
                break

            if tp1_hit_now:
                pnl += leg_pnl(
                    entry, tp1, side, TP1_CLOSE_PERCENT
                )
                tp1_hit = True

                if strategy == "BREAKEVEN":
                    sl = breakeven_stop(entry, side)

                # Remaining half may hit the adjusted stop
                # within the same candle.
                if side == "LONG":
                    adjusted_stop_hit = low <= sl
                else:
                    adjusted_stop_hit = high >= sl

                if adjusted_stop_hit:
                    pnl += leg_pnl(
                        entry, sl, side, 1 - TP1_CLOSE_PERCENT
                    )
                    outcome = (
                        "TP1_BE" if strategy == "BREAKEVEN"
                        else "TP1_SL"
                    )
                    break

                # If TP2 is also inside this candle, order is
                # ambiguous. Prefer the protective stop first.
                if tp2_hit_now:
                    pnl += leg_pnl(
                        entry, tp2, side, 1 - TP1_CLOSE_PERCENT
                    )
                    outcome = "TP2"
                    break

        else:
            # After TP1, only the remaining half is active.
            if sl_hit:
                pnl += leg_pnl(
                    entry, sl, side, 1 - TP1_CLOSE_PERCENT
                )
                outcome = (
                    "TP1_BE" if strategy == "BREAKEVEN"
                    else "TP1_SL"
                )
                break

            if tp2_hit_now:
                pnl += leg_pnl(
                    entry, tp2, side, 1 - TP1_CLOSE_PERCENT
                )
                outcome = "TP2"
                break

    if outcome == "OPEN":
        # Exclude unfinished trades from completed-trade stats.
        return None

    return {
        "side": side,
        "entry": entry,
        "sl": levels["sl"],
        "tp1": tp1,
        "tp2": tp2,
        "pnl": pnl,
        "R": pnl / risk_distance,
        "outcome": outcome,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "time": int(entry_candle["time"]),
    }


# =========================
# BACKTEST
# =========================

def run_backtest(df15, df1h, days, strategy):
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - days * 24 * 60 * 60

    trades = []
    i = 40

    while i < len(df15) - 1:
        candle_time = int(df15.iloc[i]["time"])

        if candle_time < start_ts:
            i += 1
            continue

        # Signal from the previous, fully closed candle.
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
            # Do not count a trade that has not finished.
            break

        trades.append(trade)

        # Only one position at a time; don't open another
        # trade while the previous one is still running.
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

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)

    long_trades = [t for t in trades if t["side"] == "LONG"]
    short_trades = [t for t in trades if t["side"] == "SHORT"]

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for trade in trades:
        cumulative += trade["R"]
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    return {
        "count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100,
        "total_R": sum(t["R"] for t in trades),
        "avg_R": sum(t["R"] for t in trades) / len(trades),
        "max_drawdown_R": max_drawdown,
        "long_count": len(long_trades),
        "long_wins": sum(1 for t in long_trades if t["pnl"] > 0),
        "short_count": len(short_trades),
        "short_wins": sum(1 for t in short_trades if t["pnl"] > 0),
        "tp2": sum(1 for t in trades if t["outcome"] == "TP2"),
        "sl": sum(1 for t in trades if t["outcome"] == "SL"),
        "tp1_sl": sum(1 for t in trades if t["outcome"] == "TP1_SL"),
        "tp1_be": sum(1 for t in trades if t["outcome"] == "TP1_BE"),
        "sum_pct": sum(
            t["pnl"] / t["entry"] * 100 for t in trades
        ),
    }


def format_backtest(name, days, stats):
    return (
        f"📊 {name} — {days} օր\n\n"
        f"Ավարտված գործարքներ՝ {stats['count']}\n"
        f"Շահութաբեր՝ {stats['wins']} | "
        f"Կորուստով՝ {stats['losses']}\n"
        f"Հաղթանակների տոկոս՝ {stats['win_rate']:.2f}%\n"
        f"Միջին արդյունք՝ {stats['avg_R']:.2f}R / գործարք\n"
        f"Ընդհանուր արդյունք՝ {stats['total_R']:.2f}R\n"
        f"Առավելագույն անկում՝ {stats['max_drawdown_R']:.2f}R\n"
        f"Գործարքների տոկոսների գումար՝ {stats['sum_pct']:.2f}%\n\n"
        f"LONG՝ {stats['long_count']} | "
        f"շահութաբեր՝ {stats['long_wins']}\n"
        f"SHORT՝ {stats['short_count']} | "
        f"շահութաբեր՝ {stats['short_wins']}\n\n"
        f"TP2՝ {stats['tp2']}\n"
        f"SL՝ {stats['sl']}\n"
        f"TP1-ից հետո SL՝ {stats['tp1_sl']}\n"
        f"TP1-ից հետո breakeven՝ {stats['tp1_be']}\n"
    )


# =========================
# TELEGRAM COMMANDS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Բոտը աշխատում է։\n\n"
        "Շուկա՝ BTC/USDT Futures\n"
        "Բոտը գործարքներ ինքնուրույն չի բացում։\n\n"
        "/signal — վերջին փակ մոմի ազդանշան\n"
        "/backtest — համեմատել 3 ռազմավարությունը\n"
        "/backtest 30 — ստուգել վերջին 30 օրը"
    )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text("Ստուգում եմ BTC/USDT Futures-ը...")

    try:
        now = int(datetime.now(timezone.utc).timestamp())
        start_ts = now - 10 * 24 * 60 * 60

        df15 = add_indicators(
            request_candles("15m", start_ts, now)
        )
        df1h = add_indicators(
            request_candles("1h", start_ts, now)
        )

        # -2 is the last fully closed 15m candle;
        # -1 may still be forming.
        signal_idx = len(df15) - 2
        row = df15.iloc[signal_idx]

        signal = get_signal(
            df15, df1h, signal_idx, "BASELINE"
        )

        current_price = float(row["close"])
        current_rsi = float(row["rsi"])

        if signal is None:
            await update.message.reply_text(
                "BTC/USDT — NO TRADE\n\n"
                f"Վերջին փակ մոմի գին՝ {current_price:.2f} USDT\n"
                f"RSI (15m)՝ {current_rsi:.1f}\n"
                "Մուտքի պայմանները լիարժեք չեն։"
            )
            return

        # Signal levels are indicative; this is not an order.
        entry = current_price
        levels = calculate_levels(
            entry,
            signal["side"],
            signal["atr"],
        )

        await update.message.reply_text(
            f"BTC/USDT Futures — {signal['side']}\n\n"
            f"Վերջին փակ մոմի գին՝ {current_price:.2f}\n"
            f"RSI (15m)՝ {current_rsi:.1f}\n"
            f"Մոտավոր մուտքի հղում՝ {entry:.2f}\n"
            f"Stop Loss՝ {levels['sl']:.2f}\n"
            f"TP1՝ {levels['tp1']:.2f} (փակել 50%)\n"
            f"TP2՝ {levels['tp2']:.2f} (մնացած 50%)\n\n"
            "Սա ազդանշան է, ոչ թե կատարված գործարք։ "
            "Գինը կարող է փոխվել մինչև մուտքը։"
        )

    except Exception as exc:
        logger.exception("Signal command failed")
        await update.message.reply_text(
            f"Տվյալների ստացման սխալ։\n"
            f"Մանրամասներ՝ {str(exc)[:250]}"
        )


async def backtest_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    days = DEFAULT_BACKTEST_DAYS

    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "Օգտագործիր՝ /backtest 30"
            )
            return

    if not 7 <= days <= 90:
        await update.message.reply_text(
            "Օրերի քանակը պետք է լինի 7-ից 90։"
        )
        return

    await update.message.reply_text(
        f"Բեռնում եմ BTC/USDT Futures տվյալները՝ "
        f"{days} օրվա համար։ Սա կարող է մի քիչ տևել։"
    )

    try:
        df15, df1h = download_market_data(days)

        results = []

        for key, label in STRATEGIES.items():
            trades = run_backtest(
                df15, df1h, days, key
            )
            stats = summarize(trades)
            results.append((key, label, stats))

        # Send the three strategies separately for readability.
        for key, label, stats in results:
            await update.message.reply_text(
                format_backtest(label, days, stats)
            )

        lines = ["📌 ՀԱՄԵՄԱՏՈՒԹՅՈՒՆ\n"]

        for key, label, stats in results:
            lines.append(
                f"{label}: {stats['total_R']:.2f}R | "
                f"շահումներ՝ {stats['win_rate']:.1f}% | "
                f"գործարքներ՝ {stats['count']} | "
                f"անկում՝ {stats['max_drawdown_R']:.2f}R"
            )

        lines.append(
            "\nԿարևոր՝ սա պատմական մոդելավորում է։ "
            "Slippage-ը և funding-ը ներառված չեն։ "
            "Մոմի ներսում գնի շարժման ճշգրիտ հերթականությունը "
            "չի երևում, ուստի արդյունքները մոտավոր են։"
        )

        lines.append(
            "\nՏարբերակները համեմատվում են նույն տվյալների վրա։ "
            "Դրական բեքթեստը շահույթի երաշխիք չէ։"
        )

        await update.message.reply_text("\n".join(lines))

    except Exception as exc:
        logger.exception("Backtest failed")
        await update.message.reply_text(
            "Բեքթեստը չհաջողվեց։ Ստուգիր լոգերը։\n"
            f"Սխալ՝ {str(exc)[:350]}"
        )


async def stopalerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Այս տարբերակում ավտոմատ ազդանշանների ցիկլ չկա։ "
        "Բոտը պատասխանում է միայն հրամաններին։"
    )


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is missing"
        )

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("stopalerts", stopalerts_command))

    logger.info("BTC/USDT Futures bot started")
    app.run_polling()


if __name__ == "__main__":
    main()

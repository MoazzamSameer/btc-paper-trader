from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    buy_threshold: float = 0.58
    sell_threshold: float = 0.46
    fee_rate: float = 0.001
    max_hold_days: int = 10


def generate_positions(prob_up: pd.Series, config: BacktestConfig) -> pd.Series:
    position = []
    holding = 0
    state = 0

    for prob in prob_up:
        if state == 0:
            if prob >= config.buy_threshold:
                state = 1
                holding = 1
            else:
                holding = 0
        else:
            holding += 1
            if prob <= config.sell_threshold or holding >= config.max_hold_days:
                state = 0
                holding = 0
        position.append(state)

    return pd.Series(position, index=prob_up.index, name="position")


def backtest(prices: pd.Series, prob_up: pd.Series, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aligned = pd.DataFrame({"close": prices, "prob_up": prob_up}).dropna().copy()
    aligned["position"] = generate_positions(aligned["prob_up"], config)
    aligned["market_return"] = aligned["close"].pct_change().fillna(0.0)
    aligned["strategy_return"] = aligned["position"].shift(1).fillna(0.0) * aligned["market_return"]
    aligned["turnover"] = aligned["position"].diff().abs().fillna(0.0)
    aligned["strategy_return"] -= aligned["turnover"] * config.fee_rate
    aligned["equity_curve"] = (1 + aligned["strategy_return"]).cumprod()
    aligned["buy_hold_curve"] = (1 + aligned["market_return"]).cumprod()

    trades = _build_trades(aligned, config)
    summary = _summarize(aligned, trades)
    return aligned, trades, summary


def _build_trades(aligned: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    trades = []
    entry_idx = None
    entry_price = None

    position = aligned["position"].astype(int)
    close = aligned["close"]

    for i, (idx, pos) in enumerate(position.items()):
        prev = int(position.iloc[i - 1]) if i > 0 else 0
        if prev == 0 and pos == 1:
            entry_idx = idx
            entry_price = close.loc[idx]
        elif prev == 1 and pos == 0 and entry_idx is not None:
            exit_price = close.loc[idx]
            gross = exit_price / entry_price - 1
            net = gross - 2 * config.fee_rate
            trades.append(
                {
                    "entry_date": entry_idx,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross,
                    "net_return": net,
                    "winner": int(net > 0),
                    "holding_days": (idx - entry_idx).days,
                }
            )
            entry_idx = None
            entry_price = None

    return pd.DataFrame(trades)


def _summarize(aligned: pd.DataFrame, trades: pd.DataFrame) -> dict:
    equity = aligned["equity_curve"]
    peak = equity.cummax()
    drawdown = equity / peak - 1
    win_rate = float(trades["winner"].mean()) if not trades.empty else 0.0
    total_return = float(equity.iloc[-1] - 1) if not equity.empty else 0.0
    annualized = float(equity.iloc[-1] ** (365.25 / max((aligned.index[-1] - aligned.index[0]).days, 1)) - 1) if len(aligned) > 1 else 0.0
    return {
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 1.0,
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "buy_hold_return": float(aligned["buy_hold_curve"].iloc[-1] - 1) if not aligned.empty else 0.0,
    }


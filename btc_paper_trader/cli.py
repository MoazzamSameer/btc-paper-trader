from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .backtest import BacktestConfig, backtest
from .data import download_btc_usd
from .features import build_feature_frame
from .model import probability_series, select_model


def _split_frame(frame: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15):
    n = len(frame)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train = frame.iloc[:train_end].copy()
    valid = frame.iloc[train_end:valid_end].copy()
    test = frame.iloc[valid_end:].copy()
    return train, valid, test


def _threshold_candidates(probabilities: pd.Series) -> tuple[list[float], list[float]]:
    """Generate a blend of fixed and data-driven threshold candidates."""

    p = probabilities.dropna()
    fixed_buy = [round(x, 2) for x in np.arange(0.50, 0.76, 0.01)]
    fixed_sell = [round(x, 2) for x in np.arange(0.30, 0.56, 0.01)]
    quant_buy = [round(float(q), 2) for q in p.quantile([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]).tolist()]
    quant_sell = [round(float(q), 2) for q in p.quantile([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]).tolist()]
    buy_grid = sorted(set(fixed_buy + quant_buy))
    sell_grid = sorted(set(fixed_sell + quant_sell))
    return buy_grid, sell_grid


def _score_summary(
    summary: dict,
    buy_threshold: float,
    sell_threshold: float,
    max_hold_days: int,
) -> float:
    """Prefer a healthy win band instead of chasing perfect validation streaks."""

    win_rate = float(summary.get("win_rate", 0.0))
    total_return = float(summary.get("total_return", 0.0))
    max_drawdown = abs(float(summary.get("max_drawdown", 0.0)))
    trade_count = float(summary.get("trade_count", 0.0))

    target_win_rate = 0.70
    target_trades = 6.0
    target_gap = 0.12

    win_score = max(0.0, 1.0 - abs(win_rate - target_win_rate))
    return_score = max(0.0, min(1.0, (total_return + 0.5) / 2.5))
    drawdown_score = max(0.0, min(1.0, 1.0 - max_drawdown))
    trade_score = max(0.0, min(1.0, 1.0 - abs(trade_count - target_trades) / 8.0))
    gap_score = max(0.0, min(1.0, 1.0 - abs((buy_threshold - sell_threshold) - target_gap) / target_gap))

    return (
        0.45 * win_score
        + 0.10 * return_score
        + 0.15 * drawdown_score
        + 0.15 * trade_score
        + 0.15 * gap_score
        + 0.08 * max(0.0, min(1.0, max_hold_days / 30.0))
    )


def _tune_config(prices: pd.Series, probabilities: pd.Series, fee_rate: float) -> tuple[BacktestConfig, dict, pd.DataFrame]:
    buy_grid, sell_grid = _threshold_candidates(probabilities)
    hold_grid = [3, 5, 7, 10, 15, 21, 30]

    rows: list[dict] = []
    best_config = BacktestConfig(fee_rate=fee_rate)
    best_summary = {"win_rate": -1.0, "total_return": -1.0, "trade_count": 0}
    best_score = float("-inf")
    best_tiebreak = (float("-inf"), float("-inf"), float("-inf"))
    tie_epsilon = 0.02

    for buy in buy_grid:
        for sell in sell_grid:
            if sell >= buy:
                continue
            for hold in hold_grid:
                config = BacktestConfig(
                    buy_threshold=buy,
                    sell_threshold=sell,
                    fee_rate=fee_rate,
                    max_hold_days=hold,
                )
                _, _, summary = backtest(prices, probabilities, config)
                if summary["trade_count"] < 5:
                    continue
                score = _score_summary(summary, buy, sell, hold)
                row = {
                    "buy_threshold": buy,
                    "sell_threshold": sell,
                    "max_hold_days": hold,
                    "score": score,
                    **summary,
                }
                rows.append(row)
                tiebreak = (buy, -sell, hold)
                if score > best_score + 1e-12:
                    best_score = score
                    best_tiebreak = tiebreak
                    best_config = config
                    best_summary = summary
                elif abs(score - best_score) <= tie_epsilon and tiebreak > best_tiebreak:
                    best_tiebreak = tiebreak
                    best_config = config
                    best_summary = summary

    candidates = pd.DataFrame(rows).sort_values(["score", "win_rate", "total_return"], ascending=False)
    return best_config, best_summary, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and paper trade a BTC model on 10 years of data.")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--buy-threshold", type=float, default=0.58)
    parser.add_argument("--sell-threshold", type=float, default=0.46)
    parser.add_argument("--max-hold-days", type=int, default=10)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    today = date.today()
    start_date = f"{today.year - args.years}-01-01"

    market = download_btc_usd(start=start_date, cache_path=args.artifacts / "market_data.csv")
    frame = build_feature_frame(market.frame)
    frame.to_csv(args.artifacts / "feature_frame.csv", index_label="Date")

    feature_cols = frame.attrs.get("feature_columns", [])
    feature_cols = [c for c in feature_cols if c in frame.columns]
    train, valid, test = _split_frame(frame)

    train_x = train[feature_cols]
    train_y = train["target"]
    valid_x = valid[feature_cols]
    valid_y = valid["target"]
    test_x = test[feature_cols]
    valid_close = market.frame.loc[valid.index, "Close"]
    test_close = market.frame.loc[test.index, "Close"]

    chosen = select_model(train_x, train_y, valid_x, valid_y)
    valid_probabilities = probability_series(chosen.model, valid_x)
    tuned_config, tuning_summary, candidates = _tune_config(valid_close, valid_probabilities, args.fee_rate)
    if args.buy_threshold != 0.58 or args.sell_threshold != 0.46 or args.max_hold_days != 10:
        tuned_config = BacktestConfig(
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            fee_rate=args.fee_rate,
            max_hold_days=args.max_hold_days,
        )

    full_train_x = pd.concat([train_x, valid_x], axis=0)
    full_train_y = pd.concat([train_y, valid_y], axis=0)
    chosen.model.fit(full_train_x, full_train_y)

    probabilities = probability_series(chosen.model, test_x)
    aligned, trades, summary = backtest(test_close, probabilities, tuned_config)
    summary["model_name"] = chosen.name
    summary["validation_accuracy"] = chosen.validation_accuracy
    summary["validation_best_win_rate"] = tuning_summary.get("win_rate", 0.0)
    summary["validation_best_total_return"] = tuning_summary.get("total_return", 0.0)
    summary["validation_best_score"] = _score_summary(
        tuning_summary,
        tuned_config.buy_threshold,
        tuned_config.sell_threshold,
        tuned_config.max_hold_days,
    )
    summary["source"] = market.source
    summary["start_date"] = start_date
    summary["train_rows"] = int(len(train))
    summary["valid_rows"] = int(len(valid))
    summary["test_rows"] = int(len(test))
    summary["buy_threshold"] = tuned_config.buy_threshold
    summary["sell_threshold"] = tuned_config.sell_threshold
    summary["max_hold_days"] = tuned_config.max_hold_days

    preds = pd.DataFrame(
        {
            "Close": test_close,
            "prob_up": probabilities,
            "position": aligned["position"],
            "equity_curve": aligned["equity_curve"],
            "buy_hold_curve": aligned["buy_hold_curve"],
        }
    )
    preds.to_csv(args.artifacts / "predictions.csv", index_label="Date")
    trades.to_csv(args.artifacts / "trades.csv", index=False)
    if not candidates.empty:
        candidates.head(50).to_csv(args.artifacts / "threshold_candidates.csv", index=False)
    (args.artifacts / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    joblib.dump(
        {
            "model": chosen.model,
            "feature_cols": feature_cols,
            "config": summary,
        },
        args.artifacts / "model.joblib",
    )

    print(json.dumps(summary, indent=2))
    return 0

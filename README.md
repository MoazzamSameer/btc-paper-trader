# BTC Paper Trader
Im kinda late to the party. Most people did this like 5 years ago. 
This project builds a Python-based Bitcoin trading agent that:

- downloads the last 10 years of BTC-USD data
- engineers technical features
- trains a supervised model
- runs walk-forward paper trading on historical data
- reports win rate, returns, and drawdown

## Important note

This is a research and simulation tool. It cannot guarantee a 70%+ success rate in live markets.
The code is set up to search for a strong historical configuration and report the real out-of-sample
win rate from paper trading.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m btc_paper_trader --years 10
```

## Output

The run writes artifacts to `artifacts/`:

- `market_data.csv`
- `predictions.csv`
- `trades.csv`
- `summary.json`
- `model.joblib`


"""
Replay paper trader on saved layer_outputs with different TP/SL parameters.

No model retraining needed — just re-runs the trading logic on cached signals.

Usage:
    python -m screener.replay_trader
"""

import pickle
import numpy as np
import pandas as pd
from screener.config import ScreenerConfig
from screener.paper_trader import PaperTrader
from screener.data_pipeline import init_data, _get_ohlcv_cache, load_raw_ohlcv


def replay(
    layer_outputs: list[dict],
    ohlcv: dict[str, pd.DataFrame],
    cfg: ScreenerConfig,
) -> dict:
    """Replay paper trader on pre-computed layer outputs.

    Applies the same T+1 signal delay as the real backtester:
    pipeline output from day T is used to trade on day T+1.
    """
    trader = PaperTrader(cfg)
    pending_symbols: list[str] = []

    for out in layer_outputs:
        date = out["date"]

        # Collect OHLCV for pending symbols + held position
        trade_symbols = set(pending_symbols)
        if trader.position:
            trade_symbols.add(trader.position.symbol)

        ohlcv_today = {}
        ohlcv_prev = {}
        for sym in trade_symbols:
            df = ohlcv.get(sym)
            if df is None:
                continue
            if date in df.index:
                ohlcv_today[sym] = df.loc[date]
            prev_dates = df.index[df.index < date]
            if len(prev_dates) > 0:
                ohlcv_prev[sym] = df.loc[prev_dates[-1]]

        trader.daily_update(
            date=date,
            ranked_symbols=pending_symbols,
            ohlcv_today=ohlcv_today,
            ohlcv_prev=ohlcv_prev,
        )

        pending_symbols = out.get("layer3", [])

    return trader.get_metrics()


def sweep(
    layer_outputs: list[dict],
    ohlcv: dict[str, pd.DataFrame],
    tp_range: list[float],
    sl_range: list[float],
    hold_range: list[int] | None = None,
) -> pd.DataFrame:
    """Grid search over TP/SL/hold_days combinations."""
    hold_range = hold_range or [5]
    results = []

    for tp in tp_range:
        for sl in sl_range:
            for hold in hold_range:
                cfg = ScreenerConfig()
                cfg.tp_pct = tp
                cfg.sl_pct = sl
                cfg.max_hold_days = hold

                metrics = replay(layer_outputs, ohlcv, cfg)
                metrics["tp_pct"] = tp
                metrics["sl_pct"] = sl
                metrics["max_hold_days"] = hold
                results.append(metrics)

    df = pd.DataFrame(results)
    # Reorder columns for readability
    param_cols = ["tp_pct", "sl_pct", "max_hold_days"]
    metric_cols = [c for c in df.columns if c not in param_cols]
    return df[param_cols + metric_cols]


if __name__ == "__main__":
    print("Loading data…")
    init_data()
    ohlcv_cache = _get_ohlcv_cache(ScreenerConfig())

    # Load OHLCV for the backtest range
    cfg = ScreenerConfig()
    all_syms = list(ohlcv_cache.keys())
    ohlcv = load_raw_ohlcv(all_syms, cfg.train_start, cfg.backtest_end, cfg)
    print(f"OHLCV: {len(ohlcv)} stocks")

    # Load saved layer outputs
    with open("backtest_results/backtest_results0227.pkl", "rb") as f:
        res = pickle.load(f)
    layer_outputs = res["layer_outputs"]
    print(f"Layer outputs: {len(layer_outputs)} days\n")

    # Sweep TP/SL combinations
    tp_range = [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    sl_range = [0.02, 0.03, 0.04, 0.05]
    hold_range = [3, 5, 7, 10]

    print(f"Sweeping {len(tp_range)}×{len(sl_range)}×{len(hold_range)} = "
          f"{len(tp_range)*len(sl_range)*len(hold_range)} combos…\n")

    df = sweep(layer_outputs, ohlcv, tp_range, sl_range, hold_range)

    # Sort by Sharpe
    df = df.sort_values("sharpe", ascending=False)

    print("=" * 90)
    print("TOP 20 BY SHARPE")
    print("=" * 90)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df.head(20).to_string(index=False))

    print(f"\n{'='*90}")
    print("BOTTOM 5 (worst)")
    print("=" * 90)
    print(df.tail(5).to_string(index=False))

    # Save full results
    df.to_csv("backtest_results/tp_sl_sweep.csv", index=False)
    print("\nFull results saved → backtest_results/tp_sl_sweep.csv")

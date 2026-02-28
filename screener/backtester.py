"""
Walk-Forward Backtester

Runs the full 4-layer pipeline with rolling-window initial training and
quarterly fine-tuning (warm-start XGBoost).  No lookahead bias.
Reports layer attribution and comparison benchmarks.
"""

import json
import os
import pickle
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.data_pipeline import (
    init_data,
    load_alpha158_factors,
    load_alpha158_labels,
    load_alpha158_year,
    load_industry_mapping,
    load_market_regime_features,
    load_raw_ohlcv,
    get_calendar,
    _get_ohlcv_cache,
    _get_benchmark_cache,
)
from screener.factor_timing_model import FactorTimingModel
from screener.technical_ranker import TechnicalRanker
from screener.paper_trader import PaperTrader


class WalkForwardBacktester:
    """Walk-forward backtester with quarterly retraining windows."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()

        # Models
        self.layer1 = FactorTimingModel(self.cfg)
        self.layer2 = TechnicalRanker(self.cfg)
        self.layer3 = None  # lazy-loaded when run_kronos=True
        self.trader = PaperTrader(self.cfg)

        # Data caches
        self._alpha158: pd.DataFrame | None = None
        self._labels: pd.Series | None = None
        self._regime: pd.DataFrame | None = None
        self._ohlcv: dict[str, pd.DataFrame] | None = None
        self._calendar: pd.DatetimeIndex | None = None
        self._upside_returns: pd.DataFrame | None = None
        self._downside_returns: pd.DataFrame | None = None
        self._combined_returns: pd.DataFrame | None = None

    # ── Data Loading ─────────────────────────────────────────────────────

    def load_data(self):
        """Load all required data."""
        print("=" * 60)
        print("Loading data…")
        print("=" * 60)
        init_data(self.cfg)

        # Alpha158 is loaded year-by-year inside Layer 1 to limit memory.
        # We set it to None here; Layer 1 handles its own loading.
        self._alpha158 = None
        self._labels = None
        self._regime = load_market_regime_features(self.cfg)

        self._calendar = get_calendar(
            self.cfg, self.cfg.train_start, self.cfg.backtest_end
        )

        # Load OHLCV for all universe stocks (needed for Layer 2 features + paper trading)
        all_symbols = list(_get_ohlcv_cache(self.cfg).keys())
        print(f"Loading OHLCV for {len(all_symbols)} symbols…")
        self._ohlcv = load_raw_ohlcv(
            all_symbols, self.cfg.train_start, self.cfg.backtest_end, self.cfg
        )
        print(f"OHLCV loaded for {len(self._ohlcv)} symbols.")

        # Precompute Layer 2 technical features for all stocks (cache for fast lookup)
        self.layer2.precompute_features(self._ohlcv)

        # Load CSRC industry classification for Layer 2
        industry_map = load_industry_mapping(self.cfg)
        self.layer2.set_industry_mapping(industry_map)

        # Cache forward returns once (used by Layer 2 training in every window)
        self._upside_returns, self._downside_returns, self._combined_returns = (
            self._compute_forward_returns()
        )

        # Precompute lagged IC + forward IC for all years (Layer 1)
        train_start_year = pd.Timestamp(self.cfg.train_start).year
        backtest_end_year = pd.Timestamp(self.cfg.backtest_end).year
        self.layer1.precompute_ic(train_start_year, backtest_end_year)

    # ── Retraining Windows ───────────────────────────────────────────────

    def _generate_retrain_windows(self) -> list[dict]:
        """Generate quarterly rolling-window schedule.

        Window 1: initial full training on [backtest_start - train_years, backtest_start - 1d]
        Windows 2+: fine-tune on previous quarter's data only.

        Returns list of dicts with keys:
          train_start, train_end, test_start, test_end, mode
        """
        cfg = self.cfg
        windows = []
        backtest_start = pd.Timestamp(cfg.backtest_start)
        backtest_end = pd.Timestamp(cfg.backtest_end)

        quarters = pd.date_range(backtest_start, backtest_end, freq="QS")
        if len(quarters) == 0:
            quarters = pd.DatetimeIndex([backtest_start])

        for i, q_start in enumerate(quarters):
            q_end = (
                quarters[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(quarters)
                else backtest_end
            )

            if i == 0:
                # Initial: full training on [backtest_start - train_years, backtest_start - 1d]
                train_start = q_start - pd.DateOffset(years=cfg.train_years)
                train_end = q_start - pd.Timedelta(days=1)
                mode = "initial"
            else:
                # Fine-tune: train on previous quarter's data only
                train_start = quarters[i - 1]
                train_end = q_start - pd.Timedelta(days=1)
                mode = "finetune"

            windows.append({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": q_start.strftime("%Y-%m-%d"),
                "test_end": q_end.strftime("%Y-%m-%d"),
                "mode": mode,
            })

        return windows

    # ── Layer 1+2 Training ───────────────────────────────────────────────

    def _train_layer1(self, train_start: str, train_end: str):
        """Full initial training for Layer 1."""
        print(f"\n  Training Layer 1 ({train_start} → {train_end})…")
        X, Y = self.layer1.build_training_data(
            self._alpha158, self._regime,
            train_start=train_start, train_end=train_end,
        )
        self.layer1.train(X, Y, train_start=train_start, train_end=train_end)

    def _finetune_layer1(self, train_start: str, train_end: str):
        """Fine-tune Layer 1 on new quarter's data."""
        print(f"\n  Fine-tuning Layer 1 ({train_start} → {train_end})…")
        X, Y = self.layer1.build_training_data(
            self._alpha158, self._regime,
            train_start=train_start, train_end=train_end,
        )
        self.layer1.finetune(X, Y)

    def _get_layer2_dates(self, train_start: str, train_end: str) -> list:
        """Get training dates for Layer 2 with leakage trimming and subsampling."""
        start_ts = pd.Timestamp(train_start)
        end_ts = pd.Timestamp(train_end)
        train_dates = self._calendar[
            (self._calendar >= start_ts) & (self._calendar <= end_ts)
        ]
        # Leakage trim: drop last N dates whose forward returns peek into test
        trim = self.cfg.forward_horizon_days
        if len(train_dates) > trim:
            train_dates = train_dates[:-trim]
        # Subsample for speed (every 5th trading day)
        train_dates = train_dates[::5]
        return list(train_dates)

    def _train_layer2(self, train_start: str, train_end: str):
        """Full initial training for Layer 2."""
        print(f"\n  Training Layer 2 ({train_start} → {train_end})…")
        train_dates = self._get_layer2_dates(train_start, train_end)
        X, y, w = self.layer2.build_training_data(
            self._ohlcv, train_dates, self._upside_returns, self._downside_returns
        )
        if len(X) > 0:
            self.layer2.train(X, y, sample_weight=w)
        else:
            print("  Warning: no training data for Layer 2")

    def _finetune_layer2(self, train_start: str, train_end: str):
        """Fine-tune Layer 2 on new quarter's data."""
        print(f"\n  Fine-tuning Layer 2 ({train_start} → {train_end})…")
        train_dates = self._get_layer2_dates(train_start, train_end)
        X, y, w = self.layer2.build_training_data(
            self._ohlcv, train_dates, self._upside_returns, self._downside_returns
        )
        if len(X) > 0:
            self.layer2.finetune(X, y, sample_weight=w)
        else:
            print("  Warning: no fine-tuning data for Layer 2")

    def _compute_forward_returns(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute forward upside, downside, and combined returns.

        Returns:
            (upside_df, downside_df, combined_df) — each datetime × symbol.
        """
        fwd = self.cfg.layer2_forward_days  # 5
        upside_dict, downside_dict = {}, {}
        for sym, df in self._ohlcv.items():
            close, high, low = df["close"], df["high"], df["low"]
            fwd_max = high[::-1].rolling(fwd, min_periods=fwd).max()[::-1].shift(-1)
            fwd_min = low[::-1].rolling(fwd, min_periods=fwd).min()[::-1].shift(-1)
            upside_dict[sym] = fwd_max / close - 1
            downside_dict[sym] = fwd_min / close - 1
        upside_df = pd.DataFrame(upside_dict)
        downside_df = pd.DataFrame(downside_dict)
        combined_df = upside_df + downside_df
        return upside_df, downside_df, combined_df

    def _compute_forward_close_returns(self) -> pd.DataFrame:
        """Compute 5-day forward close-to-close returns (for attribution only)."""
        close_dict = {}
        for sym, df in self._ohlcv.items():
            close_dict[sym] = df["close"]
        close_df = pd.DataFrame(close_dict)
        fwd = close_df.shift(-self.cfg.layer2_forward_days) / close_df - 1
        return fwd

    # ── Daily Pipeline ───────────────────────────────────────────────────

    def _run_daily_pipeline(
        self,
        date: pd.Timestamp,
        run_kronos: bool = True,
    ) -> dict:
        """Execute the full 4-layer pipeline for one trading day.

        Returns dict with layer outputs for attribution analysis.
        """
        result = {"date": date, "layer1": [], "layer2": [], "layer3": []}

        # Layer 1: Factor Timing → top 200
        try:
            regime_row = self._regime.loc[date] if date in self._regime.index else None
            layer1_picks = self.layer1.select_top(
                date, alpha158_df=self._alpha158, regime_row=regime_row
            )
        except Exception as e:
            print(f"  Layer 1 failed on {date}: {e}")
            layer1_picks = []
        result["layer1"] = layer1_picks

        if not layer1_picks:
            return result

        # Layer 2: Technical Ranking → top 30
        try:
            layer2_picks = self.layer2.select_top(
                self._ohlcv, layer1_picks, date, include_news=False  # no news in backtest
            )
        except Exception as e:
            print(f"  Layer 2 failed on {date}: {e}")
            layer2_picks = layer1_picks[:self.cfg.layer2_top_n]
        result["layer2"] = layer2_picks

        # Layer 3: Kronos → top 5
        layer3_picks = layer2_picks[:self.cfg.layer3_top_n]
        kronos_preds = {}
        if run_kronos and layer2_picks and self.layer3 is not None:
            try:
                scores = self.layer3.screen_stocks(
                    self._ohlcv, layer2_picks, date
                )
                if not scores.empty:
                    layer3_picks = list(scores.head(self.cfg.layer3_top_n).index)
                    for sym in layer3_picks:
                        pred = self.layer3.get_prediction(sym)
                        if pred and "pred_df" in pred:
                            kronos_preds[sym] = pred["pred_df"]
            except Exception as e:
                print(f"  Layer 3 (Kronos) failed on {date}: {e}")
        result["layer3"] = layer3_picks
        result["kronos_preds"] = kronos_preds

        return result

    # ── Full Backtest ────────────────────────────────────────────────────

    def run(
        self,
        run_kronos: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Run the full walk-forward backtest.

        Returns:
            Dict with keys: metrics, nav_series, trade_log, layer_attribution.
        """
        if self._ohlcv is None:
            self.load_data()

        windows = self._generate_retrain_windows()
        print(f"\nBacktest windows: {len(windows)}")
        for w in windows:
            print(f"  [{w['mode']:>8}] Train:{w['train_start']}→{w['train_end']}  "
                  f"Test:{w['test_start']}→{w['test_end']}")

        # Check for existing checkpoint to resume from
        ckpt = self._load_checkpoint()
        if ckpt is not None:
            resume_from = ckpt["completed_window"] + 1
            pending_symbols = ckpt["pending_symbols"]
            layer_outputs = ckpt["layer_outputs"]
            self._restore_from_checkpoint(ckpt)
            print(f"Skipping windows 1..{resume_from}, resuming at window {resume_from+1}")
        else:
            resume_from = 0
            self.trader.reset()
            layer_outputs = []
            pending_symbols = []

        # Bug fix: signal on day T, trade on day T+1 to avoid look-ahead bias.
        # Features (Alpha158, technicals) use T's close, so we can't trade at T's open.

        for wi, window in enumerate(windows):
            if wi < resume_from:
                continue
            print(f"\n{'='*60}")
            print(f"Window {wi+1}/{len(windows)} [{window['mode']}]: "
                  f"test {window['test_start']} → {window['test_end']}")
            print(f"{'='*60}")

            # Train or fine-tune models (skip if cached from a previous run)
            if not self._load_window_models(wi):
                if window["mode"] == "initial":
                    self._train_layer1(window["train_start"], window["train_end"])
                    self._train_layer2(window["train_start"], window["train_end"])
                else:
                    self._finetune_layer1(window["train_start"], window["train_end"])
                    self._finetune_layer2(window["train_start"], window["train_end"])
                self._save_window_models(wi)

            # Ensure lagged IC covers the test period for Layer 1 inference
            test_start_year = pd.Timestamp(window["test_start"]).year
            test_end_year = pd.Timestamp(window["test_end"]).year
            self.layer1.ensure_lagged_ic(range(test_start_year, test_end_year + 1))

            # Load Kronos model once per window (lazy import)
            if run_kronos:
                try:
                    if self.layer3 is None:
                        from screener.kronos_screener import KronosScreener
                        self.layer3 = KronosScreener(self.cfg)
                    self.layer3.load_model()
                except Exception as e:
                    print(f"  Kronos model load failed: {e}")
                    run_kronos = False

            # Test period
            test_dates = self._calendar[
                (self._calendar >= pd.Timestamp(window["test_start"]))
                & (self._calendar <= pd.Timestamp(window["test_end"]))
            ]

            for di, date in enumerate(test_dates):
                if verbose and di % 20 == 0:
                    print(f"  Day {di+1}/{len(test_dates)}: {date.date()}")

                # Run pipeline — generates TODAY's signal (used TOMORROW)
                pipeline_out = self._run_daily_pipeline(date, run_kronos=run_kronos)
                layer_outputs.append(pipeline_out)

                # Collect OHLCV for paper trader: pending symbols (yesterday's
                # picks, traded at today's open) + held position
                trade_symbols = set(pending_symbols)
                if self.trader.position:
                    trade_symbols.add(self.trader.position.symbol)

                ohlcv_today = {}
                ohlcv_prev = {}
                for sym in trade_symbols:
                    df = self._ohlcv.get(sym)
                    if df is None:
                        continue
                    if date in df.index:
                        ohlcv_today[sym] = df.loc[date]
                    prev_dates = df.index[df.index < date]
                    if len(prev_dates) > 0:
                        ohlcv_prev[sym] = df.loc[prev_dates[-1]]

                # Paper trader uses YESTERDAY's signal to buy at today's open
                # (avoids look-ahead: features used T's close, trade at T+1 open)
                self.trader.daily_update(
                    date=date,
                    ranked_symbols=pending_symbols,
                    ohlcv_today=ohlcv_today,
                    ohlcv_prev=ohlcv_prev,
                )

                # Store today's signal for tomorrow's trading
                pending_symbols = pipeline_out.get("layer3", [])

            # Unload Kronos after each window to save GPU memory
            if run_kronos and self.layer3 is not None:
                self.layer3.unload_model()

            # Save checkpoint after each completed window
            self._save_checkpoint(wi, pending_symbols, layer_outputs)

        # ── Results ──────────────────────────────────────────────────────
        metrics = self.trader.get_metrics()
        nav = self.trader.get_nav_series()

        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        # Layer attribution
        attribution = self._compute_layer_attribution(layer_outputs)

        # Run complete — remove checkpoint
        self._delete_checkpoint()

        return {
            "metrics": metrics,
            "nav_series": nav,
            "trade_log": self.trader.trade_log,
            "layer_attribution": attribution,
            "layer_outputs": layer_outputs,
        }

    # ── Layer Attribution ────────────────────────────────────────────────

    def _compute_layer_attribution(self, layer_outputs: list[dict]) -> dict:
        """Measure marginal alpha contribution of each layer.

        Computes the average 5-day forward return of stocks at each layer's
        cutoff to see how much each layer improves selection.
        """
        fwd_ret = self._compute_forward_close_returns()

        layer_returns = {"layer1": [], "layer2": [], "layer3": [], "universe": []}

        for out in layer_outputs:
            date = out["date"]
            if date not in fwd_ret.index:
                continue

            day_ret = fwd_ret.loc[date].dropna()
            if day_ret.empty:
                continue

            # Universe average
            layer_returns["universe"].append(day_ret.mean())

            for layer_name in ["layer1", "layer2", "layer3"]:
                picks = out.get(layer_name, [])
                if picks:
                    common = [s for s in picks if s in day_ret.index]
                    if common:
                        layer_returns[layer_name].append(day_ret.loc[common].mean())

        attribution = {}
        for name, rets in layer_returns.items():
            if rets:
                attribution[name] = {
                    "mean_5d_return": float(np.mean(rets)),
                    "std": float(np.std(rets)),
                    "n_days": len(rets),
                }
            else:
                attribution[name] = {"mean_5d_return": 0.0, "std": 0.0, "n_days": 0}

        print("\nLayer Attribution (avg 5-day forward return of selected stocks):")
        for name, stats in attribution.items():
            print(f"  {name:>10}: {stats['mean_5d_return']*100:.3f}% "
                  f"(±{stats['std']*100:.3f}%, n={stats['n_days']})")

        return attribution

    # ── Checkpointing ───────────────────────────────────────────────────

    def _checkpoint_path(self) -> str:
        return os.path.join(self.cfg.run_dir, "checkpoint.pkl")

    def _save_checkpoint(
        self,
        wi: int,
        pending_symbols: list[str],
        layer_outputs: list[dict],
    ):
        """Save progress after a completed quarterly window."""
        ckpt = {
            "completed_window": wi,
            "pending_symbols": pending_symbols,
            "layer_outputs": layer_outputs,
            # PaperTrader state
            "trader_cash": self.trader.cash,
            "trader_position": self.trader.position,
            "trader_trade_log": self.trader.trade_log,
            "trader_daily_nav": self.trader.daily_nav,
            # Model state
            "layer1_model": self.layer1.model,
            "layer1_feature_names": self.layer1.feature_names,
            "layer2_model": self.layer2.model,
        }
        path = self._checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved (window {wi+1}) → {path}")

    def _load_checkpoint(self) -> dict | None:
        """Load checkpoint if one exists. Returns checkpoint dict or None."""
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"\n*** Resuming from checkpoint (completed window {ckpt['completed_window']+1}) ***")
        return ckpt

    def _restore_from_checkpoint(self, ckpt: dict):
        """Restore trader and model state from checkpoint dict."""
        self.trader.cash = ckpt["trader_cash"]
        self.trader.position = ckpt["trader_position"]
        self.trader.trade_log = ckpt["trader_trade_log"]
        self.trader.daily_nav = ckpt["trader_daily_nav"]
        self.layer1.model = ckpt["layer1_model"]
        self.layer1.feature_names = ckpt["layer1_feature_names"]
        self.layer2.model = ckpt["layer2_model"]

    def _delete_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        path = self._checkpoint_path()
        if os.path.exists(path):
            os.remove(path)
            print(f"Checkpoint deleted (run complete) → {path}")

    # ── Per-Window Model Cache ────────────────────────────────────────
    # Persists L1+L2 model state per window in cache_dir so that
    # repeated runs (or run_rl after run) skip redundant training.

    def _window_model_cache_path(self, wi: int) -> str:
        return os.path.join(
            self.cfg.cache_dir, "l1l2_models", f"window_{wi}.pkl"
        )

    def _save_window_models(self, wi: int):
        path = self._window_model_cache_path(wi)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "layer1_model": self.layer1.model,
                "layer1_feature_names": self.layer1.feature_names,
                "layer2_model": self.layer2.model,
            }, f)
        print(f"  L1+L2 models cached (window {wi+1}) → {path}")

    def _load_window_models(self, wi: int) -> bool:
        """Load cached L1+L2 models for a window. Returns True if loaded."""
        path = self._window_model_cache_path(wi)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.layer1.model = data["layer1_model"]
        self.layer1.feature_names = data["layer1_feature_names"]
        self.layer2.model = data["layer2_model"]
        print(f"  L1+L2 models loaded from cache (window {wi+1})")
        return True

    # ── Per-Window Signal Cache ──────────────────────────────────────
    # Persists L1+L2 daily signals per window so that repeated RL runs
    # skip the expensive signal generation step entirely.

    def _signal_cache_path(self, wi: int, split: str) -> str:
        return os.path.join(
            self.cfg.cache_dir, "signals", f"window_{wi}_{split}.pkl"
        )

    def _load_cached_signals(self, wi: int, split: str) -> list[dict] | None:
        """Load cached signals for a window/split. Returns None if not cached."""
        path = self._signal_cache_path(wi, split)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_signals(self, signals: list[dict], wi: int, split: str):
        """Save signals to cache."""
        path = self._signal_cache_path(wi, split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(signals, f)

    # ── RL Checkpointing ─────────────────────────────────────────────

    def _rl_checkpoint_path(self) -> str:
        return os.path.join(self.cfg.run_dir, "rl_checkpoint.pkl")

    def _save_rl_checkpoint(
        self,
        wi: int,
        all_nav: list[tuple],
        window_results: list[dict],
        running_capital: float = 0.0,
    ):
        """Save RL backtest progress after a completed quarterly window."""
        ckpt = {
            "completed_window": wi,
            "all_nav": all_nav,
            "window_results": window_results,
            "running_capital": running_capital,
            # L1+L2 model state (needed to generate signals on resume)
            "layer1_model": self.layer1.model,
            "layer1_feature_names": self.layer1.feature_names,
            "layer2_model": self.layer2.model,
        }
        path = self._rl_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  RL checkpoint saved (window {wi+1}) → {path}")

    def _load_rl_checkpoint(self) -> dict | None:
        """Load RL checkpoint if one exists."""
        path = self._rl_checkpoint_path()
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        print(
            f"\n*** Resuming RL backtest from checkpoint "
            f"(completed window {ckpt['completed_window']+1}) ***"
        )
        return ckpt

    def _delete_rl_checkpoint(self):
        """Remove RL checkpoint file after successful completion."""
        path = self._rl_checkpoint_path()
        if os.path.exists(path):
            os.remove(path)
            print(f"RL checkpoint deleted (run complete) → {path}")

    def _save_rl_results_json(
        self,
        window_results: list[dict],
        metrics: dict | None = None,
    ):
        """Write RL results to a JSON file in the run directory.

        Called after each window so results survive Colab disconnects.
        """
        path = os.path.join(self.cfg.run_dir, "rl_results.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"window_results": window_results}
        if metrics is not None:
            payload["metrics"] = metrics
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  RL results saved → {path}")

    # ── Signal Precomputation ────────────────────────────────────────

    def precompute_signals(self, verbose: bool = True):
        """Generate and cache all per-window L1+L2 signals to disk.

        Run this once before iterating on RL hyperparameters. Subsequent
        ``run_rl()`` calls will load cached signals and skip L1+L2 entirely.
        """
        if self._ohlcv is None:
            self.load_data()

        windows = self._generate_retrain_windows()
        print(f"\nPrecomputing signals for {len(windows)} windows")

        total_days = 0
        total_bytes = 0

        for wi, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(
                f"Window {wi+1}/{len(windows)} [{window['mode']}]: "
                f"train {window['train_start']} → {window['train_end']}  "
                f"test {window['test_start']} → {window['test_end']}"
            )
            print(f"{'='*60}")

            # Check if both splits already cached
            train_cached = self._load_cached_signals(wi, "train")
            test_cached = self._load_cached_signals(wi, "test")
            if train_cached is not None and test_cached is not None:
                n = len(train_cached) + len(test_cached)
                print(f"  Already cached ({n} days), skipping")
                total_days += n
                total_bytes += (
                    os.path.getsize(self._signal_cache_path(wi, "train"))
                    + os.path.getsize(self._signal_cache_path(wi, "test"))
                )
                continue

            # Train / fine-tune L1+L2 (skip if model cache exists)
            if not self._load_window_models(wi):
                if window["mode"] == "initial":
                    self._train_layer1(window["train_start"], window["train_end"])
                    self._train_layer2(window["train_start"], window["train_end"])
                else:
                    self._finetune_layer1(window["train_start"], window["train_end"])
                    self._finetune_layer2(window["train_start"], window["train_end"])
                self._save_window_models(wi)

            # Ensure lagged IC covers train+test period
            train_start_year = pd.Timestamp(window["train_start"]).year
            test_end_year = pd.Timestamp(window["test_end"]).year
            self.layer1.ensure_lagged_ic(range(train_start_year, test_end_year + 1))

            # Generate and cache train signals
            if train_cached is None:
                print("\n  Generating train signals…")
                train_signals = self._generate_daily_signals(
                    window["train_start"], window["train_end"], verbose=verbose
                )
                self._save_signals(train_signals, wi, "train")
                print(f"  Cached {len(train_signals)} train days")
            else:
                train_signals = train_cached
                print(f"  Train signals already cached ({len(train_signals)} days)")

            # Generate and cache test signals
            if test_cached is None:
                print("  Generating test signals…")
                test_signals = self._generate_daily_signals(
                    window["test_start"], window["test_end"], verbose=verbose
                )
                self._save_signals(test_signals, wi, "test")
                print(f"  Cached {len(test_signals)} test days")
            else:
                test_signals = test_cached
                print(f"  Test signals already cached ({len(test_signals)} days)")

            n = len(train_signals) + len(test_signals)
            total_days += n
            total_bytes += (
                os.path.getsize(self._signal_cache_path(wi, "train"))
                + os.path.getsize(self._signal_cache_path(wi, "test"))
            )

        print(f"\n{'='*60}")
        print(f"Signal precomputation complete")
        print(f"  Total days: {total_days}")
        print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")
        print(f"  Cache path: {os.path.join(self.cfg.cache_dir, 'signals')}")
        print(f"{'='*60}")

    # ── RL Backtest ─────────────────────────────────────────────────────

    def _generate_daily_signals(
        self, start_date: str, end_date: str, verbose: bool = False,
    ) -> list[dict]:
        """Run L1→L2 pipeline for each trading day and capture signals.

        Returns list of dicts with keys:
            date, l2_scores (Series), l2_ranking (list[str]),
            l2_features (DataFrame symbol × tech features).
        """
        dates = self._calendar[
            (self._calendar >= pd.Timestamp(start_date))
            & (self._calendar <= pd.Timestamp(end_date))
        ]
        signals: list[dict] = []
        for di, date in enumerate(dates):
            if verbose and di % 50 == 0:
                print(f"  Signals {di+1}/{len(dates)}: {date.date()}")

            # Layer 1 — factor timing → top 200
            try:
                regime_row = (
                    self._regime.loc[date] if date in self._regime.index else None
                )
                l1_picks = self.layer1.select_top(
                    date, alpha158_df=self._alpha158, regime_row=regime_row
                )
            except Exception:
                l1_picks = []

            if not l1_picks:
                # Still emit a signal (agent may be holding stocks)
                empty = pd.Series(dtype=float)
                signals.append({
                    "date": date,
                    "l2_scores": empty,
                    "l2_upside": empty,
                    "l2_downside": empty,
                    "l2_ranking": [],
                    "l2_features": pd.DataFrame(),
                })
                continue

            # Layer 2 — ranking + features
            try:
                l2_combined, l2_upside, l2_downside = self.layer2.rank_stocks(
                    self._ohlcv, l1_picks, date, include_news=False
                )
                l2_ranking = list(l2_combined.index)  # sorted desc
                l2_features = self.layer2.compute_features_for_stocks(
                    self._ohlcv, date=date, symbols=l1_picks, include_news=False
                )
            except Exception:
                l2_combined = pd.Series(dtype=float)
                l2_upside = pd.Series(dtype=float)
                l2_downside = pd.Series(dtype=float)
                l2_ranking = l1_picks[: self.cfg.layer2_top_n]
                l2_features = pd.DataFrame()

            signals.append({
                "date": date,
                "l2_scores": l2_combined,
                "l2_upside": l2_upside,
                "l2_downside": l2_downside,
                "l2_ranking": l2_ranking,
                "l2_features": l2_features,
            })

        return signals

    def run_rl(self, verbose: bool = True) -> dict:
        """Walk-forward backtest using the RL portfolio agent.

        Same window schedule as run() but replaces the paper trader with
        a MaskablePPO agent trained on PortfolioEnv.  Checkpoints after each
        completed window so the run can resume after disconnects.

        Returns dict with keys: metrics, nav_series, window_results.
        """
        import warnings

        # Suppress noisy warnings from jupyter_client / gym / xgboost
        warnings.filterwarnings("ignore", module="jupyter_client")
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
        warnings.filterwarnings(
            "ignore", message=".*Falling back to prediction using DMatrix.*"
        )

        from screener.portfolio_env import PortfolioEnv
        from screener.rl_trader import RLTrader

        if self._ohlcv is None:
            self.load_data()

        windows = self._generate_retrain_windows()
        print(f"\nRL Backtest windows: {len(windows)}")
        for w in windows:
            print(
                f"  [{w['mode']:>8}] Train:{w['train_start']}→{w['train_end']}  "
                f"Test:{w['test_start']}→{w['test_end']}"
            )

        benchmark_df = _get_benchmark_cache(self.cfg)
        rl_trader = RLTrader(self.cfg)

        # Check for existing checkpoint to resume from
        rl_ckpt = self._load_rl_checkpoint()
        if rl_ckpt is not None:
            resume_from = rl_ckpt["completed_window"] + 1
            all_nav = rl_ckpt["all_nav"]
            window_results = rl_ckpt["window_results"]
            running_capital = rl_ckpt["running_capital"]
            self.layer1.model = rl_ckpt["layer1_model"]
            self.layer1.feature_names = rl_ckpt["layer1_feature_names"]
            self.layer2.model = rl_ckpt["layer2_model"]
            print(f"Skipping windows 1..{resume_from}, resuming at window {resume_from+1}")
        else:
            resume_from = 0
            all_nav: list[tuple[pd.Timestamp, float]] = []
            window_results: list[dict] = []
            running_capital: float = float(self.cfg.initial_capital)

        for wi, window in enumerate(windows):
            if wi < resume_from:
                continue

            print(f"\n{'='*60}")
            print(
                f"Window {wi+1}/{len(windows)} [{window['mode']}]: "
                f"test {window['test_start']} → {window['test_end']}"
            )
            print(f"{'='*60}")

            # ── Load cached signals or generate on-the-fly ─────────
            train_signals = self._load_cached_signals(wi, "train")
            test_signals = self._load_cached_signals(wi, "test")

            if train_signals is not None and test_signals is not None:
                # Cached path — skip L1+L2 entirely
                print(
                    f"  Loaded cached signals: "
                    f"{len(train_signals)} train, {len(test_signals)} test days"
                )
            else:
                # Fall back to generating on-the-fly (backwards compatible)
                if not self._load_window_models(wi):
                    if window["mode"] == "initial":
                        self._train_layer1(window["train_start"], window["train_end"])
                        self._train_layer2(window["train_start"], window["train_end"])
                    else:
                        self._finetune_layer1(window["train_start"], window["train_end"])
                        self._finetune_layer2(window["train_start"], window["train_end"])
                    self._save_window_models(wi)

                # Ensure lagged IC covers test period
                test_start_year = pd.Timestamp(window["test_start"]).year
                test_end_year = pd.Timestamp(window["test_end"]).year
                self.layer1.ensure_lagged_ic(range(test_start_year, test_end_year + 1))

                print("\n  Generating training signals…")
                train_signals = self._generate_daily_signals(
                    window["train_start"], window["train_end"], verbose=verbose
                )
                print(f"  Training signals: {len(train_signals)} days")

                print("  Generating test signals…")
                test_signals = self._generate_daily_signals(
                    window["test_start"], window["test_end"], verbose=verbose
                )
                print(f"  Test signals: {len(test_signals)} days")

            if len(train_signals) < 20:
                print("  Skipping window (insufficient training signals)")
                self._save_rl_checkpoint(wi, all_nav, window_results, running_capital)
                continue

            # ── Train PPO ─────────────────────────────────────────────
            print("\n  Training PPO agent…")
            train_env = PortfolioEnv(
                self.cfg, train_signals, self._ohlcv,
                benchmark_df, training_mode=True,
            )
            model = rl_trader.train(train_env)

            # Save model
            model_path = os.path.join(
                self.cfg.run_dir, f"rl_model_window_{wi}"
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            rl_trader.save(model, model_path)
            print(f"  Model saved → {model_path}")

            # ── Inference on test period ──────────────────────────────
            if len(test_signals) == 0:
                print("  Skipping inference (no test signals)")
                self._save_rl_checkpoint(wi, all_nav, window_results, running_capital)
                continue

            print("\n  Running inference…")
            from sb3_contrib.common.wrappers import ActionMasker
            from sb3_contrib.common.maskable.utils import get_action_masks

            test_env = PortfolioEnv(
                self.cfg, test_signals, self._ohlcv,
                benchmark_df, training_mode=False,
            )
            masked_test_env = ActionMasker(test_env, lambda e: e.action_masks())
            obs, _ = masked_test_env.reset()
            total_reward = 0.0
            blocked_count = 0
            sub_count = 0

            for step_i in range(len(test_signals) - 1):
                masks = get_action_masks(masked_test_env)
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=masks
                )
                obs, reward, terminated, truncated, info = masked_test_env.step(
                    int(action)
                )
                total_reward += reward
                blocked_count += len(info.get("blocked_trades", []))
                sub_count += len(info.get("substituted_trades", []))
                if terminated or truncated:
                    break

            # Collect NAV — scale to running capital for continuity
            test_nav = test_env._nav_history
            test_dates = [s["date"] for s in test_signals]
            scale = (
                running_capital / test_nav[0] if test_nav[0] > 0 else 1.0
            )
            for j, nav_val in enumerate(test_nav):
                if j < len(test_dates):
                    all_nav.append((test_dates[j], nav_val * scale))

            test_return = (
                test_nav[-1] / test_nav[0] - 1 if len(test_nav) > 1 else 0.0
            )
            running_capital = test_nav[-1] * scale  # carry over
            wr = {
                "window": wi,
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "test_return": float(test_return),
                "total_reward": float(total_reward),
                "blocked_trades": blocked_count,
                "substituted_trades": sub_count,
                "final_nav": float(running_capital),
            }
            window_results.append(wr)
            print(
                f"  Window return: {test_return*100:.2f}%  "
                f"Final NAV: {running_capital:,.0f}  "
                f"Blocked: {blocked_count}  Substituted: {sub_count}"
            )

            # Persist results incrementally so they survive disconnects
            self._save_rl_results_json(window_results)

            # Checkpoint after each completed window
            self._save_rl_checkpoint(wi, all_nav, window_results, running_capital)

        # ── Aggregate results ─────────────────────────────────────────
        if all_nav:
            nav_series = pd.Series(
                {d: v for d, v in all_nav}, name="nav"
            ).sort_index()
        else:
            nav_series = pd.Series(dtype=float, name="nav")

        metrics = self._compute_rl_metrics(nav_series)

        print(f"\n{'='*60}")
        print("RL BACKTEST RESULTS")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(
                f"  {k:>20}: {v:.4f}"
                if isinstance(v, float)
                else f"  {k:>20}: {v}"
            )

        # Final save with aggregate metrics
        self._save_rl_results_json(window_results, metrics)

        # Run complete — remove checkpoint
        self._delete_rl_checkpoint()

        return {
            "metrics": metrics,
            "nav_series": nav_series,
            "window_results": window_results,
        }

    @staticmethod
    def _compute_rl_metrics(nav_series: pd.Series) -> dict:
        """Compute performance metrics from a NAV series."""
        if len(nav_series) < 2:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_days": len(nav_series),
            }

        total_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)
        daily_returns = nav_series.pct_change().dropna()

        if daily_returns.std() > 0:
            sharpe = float(
                daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = float(drawdown.min())

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "n_days": len(nav_series),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save_results(self, results: dict, path: str | None = None):
        """Save backtest results and run config to disk."""
        run_dir = self.cfg.run_dir
        path = path or os.path.join(run_dir, "backtest_results.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved → {path}")

        # Save run config for provenance
        config_path = os.path.join(run_dir, "run_config.json")
        cfg_dict = asdict(self.cfg)
        # Convert non-serialisable values
        for k, v in cfg_dict.items():
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                continue
            cfg_dict[k] = str(v)
        with open(config_path, "w") as f:
            json.dump(cfg_dict, f, indent=2)
        print(f"Config saved → {config_path}")

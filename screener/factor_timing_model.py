"""
Layer 1: Factor Timing Model  (CSI1000 → ~200 stocks)

Learns which Alpha158 factor *categories* are predictive of returns given
the current calendar / market regime.  Outputs a per-stock composite score
by weighting factor categories according to predicted importance.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from screener.config import ScreenerConfig
from screener.utils import FACTOR_CATEGORIES, robust_zscore, group_features_by_category, compute_daily_category_ic
from screener.data_pipeline import (
    load_alpha158_factors,
    load_alpha158_labels,
    load_alpha158_year,
    alpha158_year_range,
    load_market_regime_features,
    compute_lagged_factor_ic,
)


class FactorTimingModel:
    """XGBoost multi-output regressor that predicts per-category forward IC."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.model: MultiOutputRegressor | None = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = [f"fwd_ic_{c}" for c in FACTOR_CATEGORIES]
        self._alpha158_df: pd.DataFrame | None = None  # cached
        self._lagged_ic_df: pd.DataFrame | None = None  # for inference lookup
        self._forward_ic_df: pd.DataFrame | None = None  # precomputed forward IC

    # ── IC Precomputation ─────────────────────────────────────────────────

    def precompute_ic(self, start_year: int, end_year: int):
        """Precompute lagged IC and forward IC for all years, with disk caching.

        Populates self._lagged_ic_df and self._forward_ic_df so that
        build_training_data() and ensure_lagged_ic() become simple slices.
        """
        import gc

        cfg = self.cfg
        cache_dir = cfg.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        realize_lag = cfg.forward_horizon_days

        lagged_parts = []
        forward_parts = []

        print(f"  Precomputing IC for years {start_year}–{end_year}…")
        for year in range(start_year, end_year + 1):
            lagged_path = os.path.join(cache_dir, f"_lagged_ic_{year}.pkl")
            forward_path = os.path.join(cache_dir, f"_forward_ic_{year}.pkl")

            if os.path.exists(lagged_path) and os.path.exists(forward_path):
                print(f"    {year}… (cached)", flush=True)
                with open(lagged_path, "rb") as f:
                    lagged_parts.append(pickle.load(f))
                with open(forward_path, "rb") as f:
                    forward_parts.append(pickle.load(f))
                continue

            print(f"    {year}…", end=" ", flush=True)
            year_df = load_alpha158_year(cfg, year)
            if year_df.empty:
                print("(skip)")
                continue
            year_labels = load_alpha158_labels(cfg, f"{year}-01-01", f"{year}-12-31")

            lagged_ic = compute_lagged_factor_ic(
                year_df, year_labels, lookback=20, realize_lag=realize_lag,
            )
            forward_ic = self._compute_forward_ic(year_df, year_labels)

            with open(lagged_path, "wb") as f:
                pickle.dump(lagged_ic, f)
            with open(forward_path, "wb") as f:
                pickle.dump(forward_ic, f)

            lagged_parts.append(lagged_ic)
            forward_parts.append(forward_ic)
            print(f"{len(year_df)} rows")
            del year_df, year_labels, lagged_ic, forward_ic
            gc.collect()

        if lagged_parts:
            self._lagged_ic_df = pd.concat(lagged_parts)
            self._lagged_ic_df = self._lagged_ic_df.loc[
                ~self._lagged_ic_df.index.duplicated(keep="last")
            ]
        if forward_parts:
            self._forward_ic_df = pd.concat(forward_parts)
            self._forward_ic_df = self._forward_ic_df.loc[
                ~self._forward_ic_df.index.duplicated(keep="last")
            ]
        print(f"  IC precomputation complete: "
              f"{len(self._lagged_ic_df) if self._lagged_ic_df is not None else 0} lagged, "
              f"{len(self._forward_ic_df) if self._forward_ic_df is not None else 0} forward rows")

    # ── Training ─────────────────────────────────────────────────────────

    def build_training_data(
        self,
        alpha158_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build (X, Y) where each row is one trading day.

        X: regime features + lagged factor IC  (~60 dims)
        Y: forward IC per factor category      (~10 dims)

        When alpha158_df is None, processes year-by-year to limit memory.
        train_start/train_end scope the data range (defaults to cfg values).
        """
        import gc

        cfg = self.cfg
        ts_start = train_start or cfg.train_start
        ts_end = train_end or cfg.train_end

        if regime_df is None:
            regime_df = load_market_regime_features(cfg, ts_start, ts_end)

        realize_lag = cfg.forward_horizon_days

        # Check if precomputed IC covers the requested range
        ts_start_ts = pd.Timestamp(ts_start)
        ts_end_ts = pd.Timestamp(ts_end)
        precomputed_ok = (
            self._lagged_ic_df is not None
            and self._forward_ic_df is not None
            and (self._lagged_ic_df.index >= ts_start_ts).any()
            and (self._lagged_ic_df.index <= ts_end_ts).any()
        )

        if precomputed_ok:
            # Fast path: slice from precomputed IC DataFrames
            mask_l = (
                (self._lagged_ic_df.index >= ts_start_ts)
                & (self._lagged_ic_df.index <= ts_end_ts)
            )
            lagged_ic = self._lagged_ic_df.loc[mask_l]
            mask_f = (
                (self._forward_ic_df.index >= ts_start_ts)
                & (self._forward_ic_df.index <= ts_end_ts)
            )
            fwd_ic = self._forward_ic_df.loc[mask_f]
        elif alpha158_df is not None:
            # Direct path: data already in memory — slice to window
            self._alpha158_df = alpha158_df
            labels = load_alpha158_labels(cfg, ts_start, ts_end)
            dt_idx = alpha158_df.index.get_level_values("datetime")
            mask = (dt_idx >= ts_start_ts) & (dt_idx <= ts_end_ts)
            windowed = alpha158_df.loc[mask]
            lagged_ic = compute_lagged_factor_ic(
                windowed, labels, lookback=20, realize_lag=realize_lag
            )
            fwd_ic = self._compute_forward_ic(windowed, labels)
        else:
            # Year-by-year path: only iterate years in [train_start, train_end]
            start_year = ts_start_ts.year
            end_year = ts_end_ts.year
            print(f"  Computing IC values year-by-year ({start_year}–{end_year})…")
            lagged_ic_parts = []
            fwd_ic_parts = []
            for year in range(start_year, end_year + 1):
                print(f"    {year}…", end=" ", flush=True)
                year_df = load_alpha158_year(cfg, year)
                if year_df.empty:
                    print("(skip)")
                    continue
                year_labels = load_alpha158_labels(
                    cfg, f"{year}-01-01", f"{year}-12-31"
                )
                lagged_ic_parts.append(
                    compute_lagged_factor_ic(
                        year_df, year_labels, lookback=20,
                        realize_lag=realize_lag,
                    )
                )
                fwd_ic_parts.append(self._compute_forward_ic(year_df, year_labels))
                print(f"{len(year_df)} rows")
                del year_df, year_labels
                gc.collect()

            lagged_ic = pd.concat(lagged_ic_parts)
            fwd_ic = pd.concat(fwd_ic_parts)
            del lagged_ic_parts, fwd_ic_parts
            gc.collect()

        # Store lagged IC for inference lookup (skip if already precomputed)
        if not precomputed_ok:
            if self._lagged_ic_df is None:
                self._lagged_ic_df = lagged_ic
            else:
                self._lagged_ic_df = pd.concat(
                    [self._lagged_ic_df, lagged_ic]
                ).loc[~pd.concat([self._lagged_ic_df, lagged_ic]).index.duplicated(keep="last")]

        # Merge X
        X = regime_df.join(lagged_ic, how="inner")
        Y = fwd_ic.reindex(X.index)
        mask = Y.notna().all(axis=1) & X.notna().all(axis=1)
        X, Y = X.loc[mask], Y.loc[mask]
        self.feature_names = list(X.columns)
        return X, Y

    def _compute_forward_ic(
        self, alpha158_df: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        """Compute daily forward IC per factor category (the label for training)."""
        ic_df = compute_daily_category_ic(alpha158_df, returns, min_valid=10)
        ic_df = ic_df.rename(columns={c: f"fwd_ic_{c}" for c in ic_df.columns})
        return ic_df.fillna(0.0)

    def _compute_sample_weights(self, index: pd.DatetimeIndex) -> np.ndarray | None:
        """Exponential recency weights: weight = 2^(-days_ago / halflife)."""
        halflife = self.cfg.sample_weight_halflife_days
        if halflife <= 0:
            return None
        days_ago = (index.max() - index).days.astype(float)
        return np.exp(-np.log(2) * days_ago / halflife)

    def train(
        self,
        X: pd.DataFrame | None = None,
        Y: pd.DataFrame | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
    ):
        """Fit the multi-output XGBoost model on training period."""
        if X is None or Y is None:
            X, Y = self.build_training_data(train_start=train_start, train_end=train_end)

        ts_start = pd.Timestamp(train_start or self.cfg.train_start)
        ts_end = pd.Timestamp(train_end or self.cfg.train_end)
        mask = (X.index >= ts_start) & (X.index <= ts_end)
        X_train, Y_train = X.loc[mask], Y.loc[mask]

        # Leakage trim: drop last N rows whose forward-return labels
        # peek past train_end into the test period
        trim = self.cfg.forward_horizon_days
        if len(X_train) > trim:
            X_train = X_train.iloc[:-trim]
            Y_train = Y_train.iloc[:-trim]
            print(f"  Leakage trim: dropped last {trim} days")

        print(f"Layer 1 training: {len(X_train)} days, {len(self.feature_names)} features, "
              f"{len(self.target_names)} targets")

        sw = self._compute_sample_weights(X_train.index)
        base = XGBRegressor(**self.cfg.layer1_xgb_params)
        self.model = MultiOutputRegressor(base)
        self.model.fit(X_train.values, Y_train.values, sample_weight=sw)
        print("Layer 1 model trained.")

    def finetune(self, X_new: pd.DataFrame, Y_new: pd.DataFrame):
        """Warm-start: add trees to existing model using new data.

        Uses fewer trees and lower learning rate for conservative adaptation.
        """
        if self.model is None:
            raise RuntimeError("No base model to fine-tune. Call .train() first.")

        # Leakage trim
        trim = self.cfg.forward_horizon_days
        if len(X_new) > trim:
            X_new = X_new.iloc[:-trim]
            Y_new = Y_new.iloc[:-trim]
            print(f"  Leakage trim: dropped last {trim} days")

        print(f"Layer 1 fine-tune: {len(X_new)} days, "
              f"+{self.cfg.finetune_n_estimators} trees @ lr={self.cfg.finetune_learning_rate}")

        sw = self._compute_sample_weights(X_new.index)
        for i, est in enumerate(self.model.estimators_):
            est.set_params(
                n_estimators=self.cfg.finetune_n_estimators,
                learning_rate=self.cfg.finetune_learning_rate,
            )
            est.fit(
                X_new.values, Y_new.iloc[:, i].values,
                sample_weight=sw,
                xgb_model=est.get_booster(),
            )
        print("Layer 1 fine-tune complete.")

    def validate(self, X: pd.DataFrame, Y: pd.DataFrame, val_start: str, val_end: str) -> dict:
        """Evaluate predicted IC vs actual forward IC on validation set."""
        mask = (X.index >= pd.Timestamp(val_start)) & (X.index <= pd.Timestamp(val_end))
        X_val, Y_val = X.loc[mask], Y.loc[mask]
        Y_pred = self.model.predict(X_val.values)
        Y_pred_df = pd.DataFrame(Y_pred, index=Y_val.index, columns=self.target_names)

        corrs = {}
        for col in self.target_names:
            valid = pd.DataFrame({"pred": Y_pred_df[col], "actual": Y_val[col]}).dropna()
            if len(valid) >= 10:
                corrs[col] = spearmanr(valid["pred"], valid["actual"]).correlation
            else:
                corrs[col] = float("nan")
        print(f"Layer 1 validation IC (predicted vs actual forward IC):")
        for k, v in corrs.items():
            print(f"  {k}: {v:.4f}")
        return corrs

    # ── Inference ────────────────────────────────────────────────────────

    def ensure_lagged_ic(self, years: range):
        """Extend lagged IC cache to cover the given years (for inference).

        Call after training to make lagged IC available for test dates.
        """
        import gc

        realize_lag = self.cfg.forward_horizon_days
        for year in years:
            # Skip years already covered
            if self._lagged_ic_df is not None:
                existing = self._lagged_ic_df.index
                year_start = pd.Timestamp(f"{year}-01-01")
                year_end = pd.Timestamp(f"{year}-12-31")
                if ((existing >= year_start) & (existing <= year_end)).any():
                    continue

            year_df = load_alpha158_year(self.cfg, year)
            if year_df.empty:
                continue
            year_labels = load_alpha158_labels(
                self.cfg, f"{year}-01-01", f"{year}-12-31"
            )
            ic_part = compute_lagged_factor_ic(
                year_df, year_labels, lookback=20, realize_lag=realize_lag,
            )
            if self._lagged_ic_df is None:
                self._lagged_ic_df = ic_part
            else:
                combined = pd.concat([self._lagged_ic_df, ic_part])
                self._lagged_ic_df = combined.loc[~combined.index.duplicated(keep="last")]
            del year_df, year_labels, ic_part
            gc.collect()

    def score_stocks(
        self,
        date: pd.Timestamp,
        alpha158_df: pd.DataFrame | None = None,
        regime_row: pd.Series | None = None,
    ) -> pd.Series:
        """Score all stocks on a given date, return top-N symbols.

        1. Predict factor category weights from regime features.
        2. Compute weighted composite score per stock.
        3. Return per-stock score (higher = better).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        if alpha158_df is None:
            alpha158_df = self._alpha158_df
        if alpha158_df is None:
            # Load only the needed year to save memory
            alpha158_df = load_alpha158_year(self.cfg, date.year)
            self._alpha158_df = alpha158_df
            self._alpha158_year = date.year
        elif hasattr(self, "_alpha158_year") and self._alpha158_year != date.year:
            # Year changed, reload
            alpha158_df = load_alpha158_year(self.cfg, date.year)
            self._alpha158_df = alpha158_df
            self._alpha158_year = date.year

        # Get regime features for this date
        if regime_row is None:
            regime_df = load_market_regime_features(self.cfg, str(date.date()), str(date.date()))
            regime_row = regime_df.iloc[-1] if len(regime_df) > 0 else pd.Series(dtype=float)

        # Merge lagged IC features into regime_row (fixes train-test mismatch)
        if self._lagged_ic_df is not None and date in self._lagged_ic_df.index:
            ic_row = self._lagged_ic_df.loc[date]
            regime_row = pd.concat([regime_row, ic_row])

        # Align features
        x = regime_row.reindex(self.feature_names, fill_value=0).values.reshape(1, -1)
        predicted_weights = self.model.predict(x)[0]  # shape (n_categories,)

        # Sign-preserving normalisation: negative IC → negative weight
        weights = np.clip(predicted_weights, -1, 1)
        abs_sum = np.abs(weights).sum() + 1e-9
        weights = weights / abs_sum
        weight_map = dict(zip(FACTOR_CATEGORIES, weights))

        # Get per-stock Alpha158 for this date
        try:
            day_factors = alpha158_df.xs(date, level="datetime")
        except KeyError:
            return pd.Series(dtype=float)

        feature_groups = group_features_by_category(list(day_factors.columns))
        composite = pd.Series(0.0, index=day_factors.index)
        for cat, w in weight_map.items():
            feats = feature_groups.get(cat, [])
            if feats:
                composite += w * day_factors[feats].mean(axis=1)

        return composite.sort_values(ascending=False)

    def select_top(self, date: pd.Timestamp, **kwargs) -> list[str]:
        """Return top-N stock symbols for a given date."""
        scores = self.score_stocks(date, **kwargs)
        return list(scores.head(self.cfg.layer1_top_n).index)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer1_factor_timing.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        print(f"Layer 1 model saved → {path}")

    def load(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer1_factor_timing.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        print(f"Layer 1 model loaded ← {path}")

"""
Layer 2: Technical Ranking Model  (200 → ~30 stocks)

Uses a MultiOutputRegressor wrapping two XGBRegressors to predict
upside and downside returns separately over the next 5 days.
Combined score = predicted upside + predicted downside.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from screener.config import ScreenerConfig
from screener.news_scorer import NewsScorer


class TechnicalRanker:
    """MultiOutput XGBRegressor that ranks stocks by predicted upside + downside."""

    TECH_FEATURES = [
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_5",
        "ma5_slope", "ma20_slope", "ma60_slope",
        "bb_position",
        "volume_trend",
        "mom_5", "mom_10", "mom_20",
        "atr_14",
        "obv_slope",
    ]
    NEWS_FEATURES = ["news_sentiment", "news_significance", "policy_flag"]
    ALL_FEATURES = TECH_FEATURES + NEWS_FEATURES

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.model: MultiOutputRegressor | None = None
        self.news_scorer: NewsScorer | None = None
        self._feature_cache: dict[str, pd.DataFrame] = {}

    # ── Feature Computation ──────────────────────────────────────────────

    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.Series:
        """Compute technical features from an OHLCV DataFrame for one stock.

        Expects columns: open, high, low, close, volume (or vol).
        Returns a Series with feature values for the *last* row.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else df["vol"]

        feats = {}

        # MACD(12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd_line.iloc[-1]
        feats["macd_signal"] = signal_line.iloc[-1]
        feats["macd_hist"] = (macd_line - signal_line).iloc[-1]

        # RSI(14), RSI(5)
        for period in [14, 5]:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            feats[f"rsi_{period}"] = (100 - 100 / (1 + rs)).iloc[-1]

        # MA slopes (normalised by price)
        for w in [5, 20, 60]:
            ma = close.rolling(w).mean()
            slope = (ma.iloc[-1] - ma.iloc[-min(w, 5)]) / (close.iloc[-1] + 1e-9)
            feats[f"ma{w}_slope"] = slope

        # Bollinger Band position: (close - lower) / (upper - lower)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        feats["bb_position"] = (close.iloc[-1] - lower.iloc[-1]) / (bb_range + 1e-9)

        # Volume trend (5-day MA / 20-day MA)
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        feats["volume_trend"] = (vol_ma5.iloc[-1] / (vol_ma20.iloc[-1] + 1e-9))

        # Momentum (return over N days)
        for d in [5, 10, 20]:
            feats[f"mom_{d}"] = (close.iloc[-1] / close.iloc[-d] - 1) if len(close) > d else 0.0

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean().iloc[-1] / (close.iloc[-1] + 1e-9)

        # OBV slope (normalised)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / (abs(obv.iloc[-5]) + 1e-9) if len(obv) > 5 else 0.0
        feats["obv_slope"] = obv_slope

        return pd.Series(feats)

    @staticmethod
    def _compute_technical_features_full(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized technical features across the full time series of one stock.

        Same math as compute_technical_features() but returns a DataFrame
        (dates x 15 features) instead of a single-row Series.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else df["vol"]

        feats = {}

        # MACD(12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd_line
        feats["macd_signal"] = signal_line
        feats["macd_hist"] = macd_line - signal_line

        # RSI(14), RSI(5)
        delta = close.diff()
        for period in [14, 5]:
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            feats[f"rsi_{period}"] = 100 - 100 / (1 + rs)

        # MA slopes (normalised by price)
        for w in [5, 20, 60]:
            ma = close.rolling(w).mean()
            shifted = ma.shift(min(w, 5))  # ma value 5 bars ago (or w bars ago if w<5)
            feats[f"ma{w}_slope"] = (ma - shifted) / (close + 1e-9)

        # Bollinger Band position
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower_bb = ma20 - 2 * std20
        bb_range = upper - lower_bb
        feats["bb_position"] = (close - lower_bb) / (bb_range + 1e-9)

        # Volume trend (5-day MA / 20-day MA)
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        feats["volume_trend"] = vol_ma5 / (vol_ma20 + 1e-9)

        # Momentum (return over N days)
        for d in [5, 10, 20]:
            feats[f"mom_{d}"] = close / close.shift(d) - 1

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean() / (close + 1e-9)

        # OBV slope (normalised)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_shifted = obv.shift(5)
        feats["obv_slope"] = (obv - obv_shifted) / (obv_shifted.abs() + 1e-9)

        return pd.DataFrame(feats, index=df.index).astype(np.float32)

    def precompute_features(self, ohlcv_dict: dict[str, pd.DataFrame]):
        """Precompute technical features for all stocks across full history.

        Stores results in self._feature_cache for O(1) lookup by (symbol, date).
        Uses disk cache to avoid recomputation across runtime restarts.
        """
        cache_path = os.path.join(self.cfg.cache_dir, "layer2_feature_cache.pkl")

        expected = sum(1 for df in ohlcv_dict.values() if len(df) >= 60)
        if os.path.exists(cache_path):
            print(f"Loading Layer 2 feature cache from {cache_path}…", flush=True)
            with open(cache_path, "rb") as f:
                self._feature_cache = pickle.load(f)
            if len(self._feature_cache) >= expected * 0.95:
                print(f"  Loaded {len(self._feature_cache)} stocks from cache.")
                return
            print(f"  Stale cache ({len(self._feature_cache)} stocks, expected ~{expected}). Recomputing…")

        print(f"Precomputing Layer 2 features for {len(ohlcv_dict)} stocks…", flush=True)
        cache = {}
        skipped = 0
        for sym, df in ohlcv_dict.items():
            if len(df) < 60:
                skipped += 1
                continue
            try:
                cache[sym] = self._compute_technical_features_full(df)
            except Exception:
                skipped += 1
                continue
        self._feature_cache = cache
        print(f"  Cached {len(cache)} stocks ({skipped} skipped).")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  Saved to {cache_path}")

    def compute_features_for_stocks(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        date: pd.Timestamp | None = None,
        symbols: list[str] | None = None,
        include_news: bool = True,
    ) -> pd.DataFrame:
        """Compute technical + news features for a list of stocks.

        Args:
            ohlcv_dict: symbol → OHLCV DataFrame.
            date: Date for which to compute features (uses latest data up to this date).
            symbols: Subset of symbols to process.  Defaults to all in ohlcv_dict.
            include_news: Whether to include news sentiment features.

        Returns:
            DataFrame indexed by symbol, one row per stock, columns = ALL_FEATURES.
        """
        symbols = symbols or list(ohlcv_dict.keys())
        records = []
        use_cache = bool(self._feature_cache) and date is not None
        for sym in symbols:
            if use_cache and sym in self._feature_cache:
                # O(1) lookup from precomputed cache
                cached = self._feature_cache[sym]
                if date in cached.index:
                    feats = cached.loc[date]
                    feats.name = sym
                    records.append(feats)
                    continue
            # Fallback: compute on the fly (backwards-compatible)
            df = ohlcv_dict.get(sym)
            if df is None or len(df) < 60:
                continue
            if date is not None:
                df = df.loc[:date]
                if len(df) < 60:
                    continue
            try:
                feats = self.compute_technical_features(df)
                feats.name = sym
                records.append(feats)
            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=self.ALL_FEATURES)

        tech_df = pd.DataFrame(records)

        # Add news features
        if include_news:
            news_df = self._get_news_features(list(tech_df.index))
            tech_df = tech_df.join(news_df, how="left")

        # Fill missing with 0
        for col in self.ALL_FEATURES:
            if col not in tech_df.columns:
                tech_df[col] = 0.0
        tech_df = tech_df[self.ALL_FEATURES].fillna(0)
        return tech_df

    def _get_news_features(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch news sentiment features for symbols."""
        if self.news_scorer is None:
            self.news_scorer = NewsScorer(self.cfg)
        try:
            return self.news_scorer.score_batch(symbols)
        except Exception:
            # Graceful degradation — return zeros if news fetching fails
            return pd.DataFrame(0.0, index=symbols, columns=self.NEWS_FEATURES)

    # ── Training ─────────────────────────────────────────────────────────

    def build_training_data(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        dates: list[pd.Timestamp],
        upside_returns: pd.DataFrame,
        downside_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Build (X, y, sample_weight) arrays for MultiOutput training.

        Args:
            ohlcv_dict: symbol → full OHLCV history.
            dates: List of training dates.
            upside_returns: DataFrame (datetime×symbol) of max_high/close - 1.
            downside_returns: DataFrame (datetime×symbol) of min_low/close - 1.

        Returns:
            X: (total_stocks_across_days, n_features)
            y: (total_stocks_across_days, 2) — [upside, downside]
            w: sample weights (or None if disabled)
        """
        X_list, y_up_list, y_dn_list, ts_list = [], [], [], []

        for dt in dates:
            feat_df = self.compute_features_for_stocks(
                ohlcv_dict, date=dt, include_news=False  # news only at inference
            )
            if feat_df.empty:
                continue

            # Get forward returns for this date
            if dt not in upside_returns.index or dt not in downside_returns.index:
                continue
            fwd_up = upside_returns.loc[dt]
            fwd_dn = downside_returns.loc[dt]
            common = feat_df.index.intersection(
                fwd_up.dropna().index.intersection(fwd_dn.dropna().index)
            )
            if len(common) < 10:
                continue

            feat_df = feat_df.loc[common]

            X_list.append(feat_df.values)
            y_up_list.append(fwd_up.loc[common].values)
            y_dn_list.append(fwd_dn.loc[common].values)
            ts_list.append(np.full(len(common), dt.timestamp()))

        X = np.vstack(X_list).astype(np.float32)
        y = np.column_stack([
            np.concatenate(y_up_list),
            np.concatenate(y_dn_list),
        ]).astype(np.float32)

        # Exponential recency weights
        halflife = self.cfg.sample_weight_halflife_days
        if halflife > 0:
            all_ts = np.concatenate(ts_list)
            days_ago = (all_ts.max() - all_ts) / 86400.0
            w = np.exp(-np.log(2) * days_ago / halflife).astype(np.float32)
        else:
            w = None
        return X, y, w

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Fit MultiOutputRegressor (2 XGBRegressors) on pre-built training data."""
        params = dict(self.cfg.layer2_xgb_params)
        base = XGBRegressor(**params)
        self.model = MultiOutputRegressor(base)
        self.model.fit(X, y, sample_weight=sample_weight)
        print(f"Layer 2 model trained: {X.shape[0]} samples, {X.shape[1]} features, 2 outputs")

    def finetune(self, X_new: np.ndarray, y_new: np.ndarray, sample_weight: np.ndarray | None = None):
        """Warm-start: add trees to each sub-estimator using new quarter's data."""
        if self.model is None:
            raise RuntimeError("No base model to fine-tune. Call .train() first.")

        print(f"Layer 2 fine-tune: {X_new.shape[0]} samples, "
              f"+{self.cfg.finetune_n_estimators} trees @ lr={self.cfg.finetune_learning_rate}")
        for i, est in enumerate(self.model.estimators_):
            est.set_params(
                n_estimators=self.cfg.finetune_n_estimators,
                learning_rate=self.cfg.finetune_learning_rate,
            )
            est.fit(X_new, y_new[:, i], sample_weight=sample_weight,
                    xgb_model=est.get_booster())
        print("Layer 2 fine-tune complete.")

    # ── Inference ────────────────────────────────────────────────────────

    def rank_stocks(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
        include_news: bool = True,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Rank symbols by predicted combined return.

        Returns:
            (combined, upside, downside) — all Series indexed by symbol,
            sorted by combined descending (best first).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        feat_df = self.compute_features_for_stocks(
            ohlcv_dict, date=date, symbols=symbols, include_news=include_news
        )
        if feat_df.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty

        X = feat_df.values.astype(np.float32)
        preds = self.model.predict(X)  # (n, 2): [upside, downside]
        upside = pd.Series(preds[:, 0], index=feat_df.index, name="l2_upside")
        downside = pd.Series(preds[:, 1], index=feat_df.index, name="l2_downside")
        combined = pd.Series(preds[:, 0] + preds[:, 1], index=feat_df.index, name="rank_score")
        order = combined.sort_values(ascending=False).index
        return combined.loc[order], upside.loc[order], downside.loc[order]

    def select_top(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
        **kwargs,
    ) -> list[str]:
        """Return top-N symbols."""
        combined, _, _ = self.rank_stocks(ohlcv_dict, symbols, date, **kwargs)
        return list(combined.head(self.cfg.layer2_top_n).index)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer2_technical_ranker.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)
        print(f"Layer 2 model saved → {path}")

    def load(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer2_technical_ranker.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        print(f"Layer 2 model loaded ← {path}")

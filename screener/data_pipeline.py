"""
Data pipeline: Alpha158 factor computation, regime features, raw OHLCV.

All data sourced from local pickle files (baostock format, no Qlib).
OHLCV pickle: Dict[str, DataFrame] keyed by baostock symbol (e.g. 'sh.600000').
Benchmark pickle: single DataFrame indexed by date.
"""

import os
import pickle

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.utils import (
    calendar_features_series,
    compute_daily_category_ic,
    group_features_by_category,
    FACTOR_CATEGORIES,
    robust_zscore,
)


# ── Module-level data cache ──────────────────────────────────────────────────

_ohlcv_cache: dict[str, pd.DataFrame] | None = None
_benchmark_cache: pd.DataFrame | None = None


def init_data(cfg: ScreenerConfig | None = None):
    """Load pickle data into module cache (idempotent)."""
    global _ohlcv_cache, _benchmark_cache
    cfg = cfg or ScreenerConfig()

    if _ohlcv_cache is None:
        print(f"Loading OHLCV pickle: {cfg.ohlcv_pickle_path}")
        raw = pd.read_pickle(cfg.ohlcv_pickle_path)
        # Convert to float32 to halve memory (~670 MB → ~335 MB)
        _ohlcv_cache = {
            sym: df.astype(np.float32) for sym, df in raw.items()
        }
        del raw
        print(f"  {len(_ohlcv_cache)} stocks loaded (float32).")

    if _benchmark_cache is None:
        print(f"Loading benchmark pickle: {cfg.benchmark_pickle_path}")
        _benchmark_cache = pd.read_pickle(cfg.benchmark_pickle_path)
        print(f"  {len(_benchmark_cache)} trading days loaded.")


def _get_ohlcv_cache(cfg: ScreenerConfig) -> dict[str, pd.DataFrame]:
    global _ohlcv_cache
    if _ohlcv_cache is None:
        init_data(cfg)
    return _ohlcv_cache


def _get_benchmark_cache(cfg: ScreenerConfig) -> pd.DataFrame:
    global _benchmark_cache
    if _benchmark_cache is None:
        init_data(cfg)
    return _benchmark_cache


# ── Calendar helpers ─────────────────────────────────────────────────────────

def get_calendar(cfg: ScreenerConfig, start: str, end: str) -> pd.DatetimeIndex:
    """Return trading-day calendar between start and end (inclusive)."""
    bench = _get_benchmark_cache(cfg)
    mask = (bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))
    return pd.DatetimeIndex(bench.index[mask])


def get_full_calendar(cfg: ScreenerConfig) -> pd.DatetimeIndex:
    """Return full trading-day calendar from benchmark data."""
    bench = _get_benchmark_cache(cfg)
    return pd.DatetimeIndex(bench.index)


# ── Alpha158-equivalent features (pure pandas) ──────────────────────────────

def _compute_alpha158_single(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~109 Alpha158-equivalent features for one stock's OHLCV.

    Args:
        df: DataFrame with columns [open, high, low, close, volume, amount],
            indexed by date.

    Returns:
        DataFrame with 109 feature columns, same date index.
    """
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    c = df["close"]
    v = df["volume"]

    features = {}

    # KBAR features (9)
    features["KMID$0"] = (c - o) / o
    features["KLEN$0"] = (h - lo) / o
    features["KMID2$0"] = (c - o) / (h - lo + 1e-12)
    features["KUP$0"] = (h - np.maximum(o, c)) / o
    features["KUP2$0"] = (h - np.maximum(o, c)) / (h - lo + 1e-12)
    features["KLOW$0"] = (np.minimum(o, c) - lo) / o
    features["KLOW2$0"] = (np.minimum(o, c) - lo) / (h - lo + 1e-12)
    features["KSFT$0"] = (2 * c - h - lo) / o
    features["KSFT2$0"] = (2 * c - h - lo) / (h - lo + 1e-12)

    # Pre-compute daily return and log volume ratio for rolling features
    prev_c = c.shift(1)
    daily_ret = c / prev_c
    abs_ret = (c - prev_c).abs()
    log_v = np.log(v + 1)
    log_v_ratio = np.log(v / v.shift(1).replace(0, np.nan) + 1)
    up = (c > prev_c).astype(float)
    down = (c < prev_c).astype(float)
    gain = np.maximum(c - prev_c, 0)
    loss = np.maximum(prev_c - c, 0)

    for N in [5, 10, 20, 30, 60]:
        # momentum / trend
        features[f"ROC${N}"] = c.shift(N) / c
        features[f"MA${N}"] = c.rolling(N).mean() / c

        # volatility
        features[f"STD${N}"] = c.rolling(N).std() / c

        # price extremes
        features[f"MAX${N}"] = h.rolling(N).max() / c
        features[f"MIN${N}"] = lo.rolling(N).min() / c

        # mean reversion (RSV)
        roll_max = h.rolling(N).max()
        roll_min = lo.rolling(N).min()
        features[f"RSV${N}"] = (c - roll_min) / (roll_max - roll_min + 1e-12)

        # correlation
        features[f"CORR${N}"] = c.rolling(N).corr(log_v)
        features[f"CORD${N}"] = (c / prev_c).rolling(N).corr(log_v_ratio)

        # volume-price counts
        features[f"CNTP${N}"] = up.rolling(N).mean()
        features[f"CNTN${N}"] = down.rolling(N).mean()
        features[f"CNTD${N}"] = features[f"CNTP${N}"] - features[f"CNTN${N}"]

        # sum positive / negative
        sum_abs = abs_ret.rolling(N).sum()
        features[f"SUMP${N}"] = gain.rolling(N).sum() / (sum_abs + 1e-12)
        features[f"SUMN${N}"] = loss.rolling(N).sum() / (sum_abs + 1e-12)
        features[f"SUMD${N}"] = features[f"SUMP${N}"] - features[f"SUMN${N}"]

        # volume features
        features[f"VMA${N}"] = v.rolling(N).mean() / (v + 1e-12)
        features[f"VSTD${N}"] = v.rolling(N).std() / (v + 1e-12)

        # WVMA: weighted volume-price volatility
        wv = (c / prev_c - 1).abs() * v
        features[f"WVMA${N}"] = wv.rolling(N).std() / (wv.rolling(N).mean() + 1e-12)

        # volume split by direction
        vol_up = (up * v).rolling(N).sum()
        vol_down = (down * v).rolling(N).sum()
        vol_sum = v.rolling(N).sum()
        features[f"VSUMP${N}"] = vol_up / (vol_sum + 1e-12)
        features[f"VSUMN${N}"] = vol_down / (vol_sum + 1e-12)
        features[f"VSUMD${N}"] = features[f"VSUMP${N}"] - features[f"VSUMN${N}"]

    return pd.DataFrame(features, index=df.index).astype(np.float32)


def _cs_robust_zscore(group, clip=3.0):
    median = group.median()
    mad = (group - median).abs().median()
    return ((group - median) / (1.4826 * mad + 1e-9)).clip(-clip, clip)


_SUBBATCH = 500


def _compute_year_chunk(
    ohlcv: dict[str, pd.DataFrame],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> pd.DataFrame:
    """Compute Alpha158 features for one year, cross-sectionally normalised.

    Processes stocks in sub-batches to limit peak memory.
    """
    import gc

    lookback_start = year_start - pd.Timedelta(days=120)

    # Build sub-batches: compute features for _SUBBATCH stocks at a time,
    # immediately concat and free the per-stock frames.
    merged = []
    batch = []
    for sym, sdf in ohlcv.items():
        sdf_slice = sdf.loc[sdf.index >= lookback_start]
        if len(sdf_slice) < 65:
            continue
        feat = _compute_alpha158_single(sdf_slice)
        feat = feat.loc[(feat.index >= year_start) & (feat.index <= year_end)]
        if feat.empty:
            continue
        feat["instrument"] = sym
        batch.append(feat)

        if len(batch) >= _SUBBATCH:
            sub = pd.concat(batch)
            sub = sub.set_index("instrument", append=True)
            merged.append(sub)
            batch = []

    if batch:
        sub = pd.concat(batch)
        sub = sub.set_index("instrument", append=True)
        merged.append(sub)
        batch = []

    if not merged:
        return pd.DataFrame()

    df = pd.concat(merged)
    del merged
    gc.collect()

    df.index.names = ["datetime", "instrument"]

    # Cross-sectional RobustZScore normalisation per day
    # Manual loop avoids huge intermediate copies from groupby().transform()
    import warnings
    dates = df.index.get_level_values("datetime").unique()
    for dt in dates:
        mask = df.index.get_level_values("datetime") == dt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df.loc[mask] = _cs_robust_zscore(df.loc[mask])
    df = df.fillna(0)
    return df


def _alpha158_year_path(cfg: ScreenerConfig, year: int) -> str:
    """Path to a single year's Alpha158 cache file."""
    return os.path.join(cfg.cache_dir, f"_alpha158_chunk_{year}.pkl")


def _ensure_alpha158_years(cfg: ScreenerConfig, start_year: int, end_year: int):
    """Compute and cache Alpha158 for any missing years in [start_year, end_year]."""
    import gc

    ohlcv = _get_ohlcv_cache(cfg)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        path = _alpha158_year_path(cfg, year)
        if os.path.exists(path):
            continue

        y_start = pd.Timestamp(f"{year}-01-01")
        y_end = pd.Timestamp(f"{year}-12-31")

        print(f"  {year}: computing…", end=" ", flush=True)
        chunk = _compute_year_chunk(ohlcv, y_start, y_end)

        if chunk.empty:
            print("(no data)")
            continue

        n_stocks = chunk.index.get_level_values("instrument").nunique()
        n_dates = chunk.index.get_level_values("datetime").nunique()
        print(f"{n_stocks} stocks × {n_dates} days")

        chunk.to_pickle(path)
        del chunk
        gc.collect()


def load_alpha158_factors(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
    *,
    cache: bool = True,
) -> pd.DataFrame:
    """Load Alpha158 factors for the requested date range.

    Uses yearly cache files (~300-500 MB each) and only loads the years
    that overlap with [start, end].  If loading the full 2015-2026 range
    would exceed memory, use load_alpha158_year() to iterate year-by-year.
    """
    import gc

    start = start or cfg.train_start
    end = end or cfg.backtest_end
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_year = start_ts.year
    end_year = end_ts.year

    # Ensure all needed year files exist
    _ensure_alpha158_years(cfg, start_year, end_year)

    # Load only the needed years
    chunks = []
    for year in range(start_year, end_year + 1):
        path = _alpha158_year_path(cfg, year)
        if os.path.exists(path):
            chunks.append(pd.read_pickle(path))

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks)
    del chunks
    gc.collect()

    # Trim to exact date range
    mask = (df.index.get_level_values("datetime") >= start_ts) & (
        df.index.get_level_values("datetime") <= end_ts
    )
    return df.loc[mask]


def load_alpha158_year(cfg: ScreenerConfig, year: int) -> pd.DataFrame:
    """Load a single year's Alpha158 data. Memory-efficient for iteration."""
    _ensure_alpha158_years(cfg, year, year)
    path = _alpha158_year_path(cfg, year)
    if os.path.exists(path):
        return pd.read_pickle(path)
    return pd.DataFrame()


def alpha158_year_range(cfg: ScreenerConfig) -> range:
    """Return the year range covered by the config."""
    return range(
        pd.Timestamp(cfg.train_start).year,
        pd.Timestamp(cfg.backtest_end).year + 1,
    )


def load_alpha158_labels(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load forward return labels (CSRankNorm'd)."""
    start = start or cfg.train_start
    end = end or cfg.backtest_end
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    fwd = cfg.layer1_forward_days

    ohlcv = _get_ohlcv_cache(cfg)

    pieces = []
    for sym, sdf in ohlcv.items():
        close = sdf["close"]
        high = sdf["high"]
        low = sdf["low"]
        # Combined upside + downside: penalises stocks with worse downside than upside
        fwd_max_high = high[::-1].rolling(fwd, min_periods=fwd).max()[::-1].shift(-1)
        fwd_min_low = low[::-1].rolling(fwd, min_periods=fwd).min()[::-1].shift(-1)
        fwd_ret = (fwd_max_high / close - 1) + (fwd_min_low / close - 1)
        fwd_ret = fwd_ret.loc[(fwd_ret.index >= start_ts) & (fwd_ret.index <= end_ts)]
        fwd_ret = fwd_ret.dropna()
        if fwd_ret.empty:
            continue
        fwd_ret_df = fwd_ret.to_frame("label")
        fwd_ret_df["instrument"] = sym
        pieces.append(fwd_ret_df)

    df = pd.concat(pieces)
    df = df.set_index("instrument", append=True)
    df.index.names = ["datetime", "instrument"]
    labels = df["label"]

    # Cross-sectional rank normalisation per day (centre around 0)
    labels = labels.groupby(level="datetime").transform(
        lambda g: g.rank(pct=True) - 0.5
    )
    return labels


# ── Market Regime Features ───────────────────────────────────────────────────

def load_market_regime_features(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Compute market-level regime features per trading day.

    Features (~40-60 dims):
      - Calendar (5): month, quarter, day_of_week, day_of_month, days_to_quarter_end
      - Market (3): 20d index return, 20d volatility, market breadth
      - Sector (≤28): 20d return per code-prefix grouping
      - Lagged factor IC (≤10): rolling IC per factor category over past 20d
    """
    start = start or cfg.train_start
    end = end or cfg.backtest_end

    cal_idx = get_calendar(cfg, start, end)
    cal_df = calendar_features_series(cal_idx)

    # Market-level features from benchmark
    bench = _get_benchmark_cache(cfg)
    idx_close = bench["close"]

    market = pd.DataFrame(index=cal_idx)
    market["idx_ret_20d"] = idx_close.pct_change(20).reindex(cal_idx)
    market["idx_vol_20d"] = idx_close.pct_change().rolling(20).std().reindex(cal_idx)

    # Market breadth: % of stocks above their 20-day MA
    breadth = _compute_market_breadth(cfg, start, end, cal_idx)
    market["market_breadth"] = breadth

    # Sector returns
    sector_df = _compute_sector_returns(cfg, start, end, cal_idx)

    # Merge everything
    regime = cal_df.join(market, how="left").join(sector_df, how="left")
    regime = regime.ffill().fillna(0)
    return regime


def _compute_market_breadth(
    cfg: ScreenerConfig, start: str, end: str, cal_idx: pd.DatetimeIndex
) -> pd.Series:
    """Fraction of universe stocks whose close > MA20."""
    ohlcv = _get_ohlcv_cache(cfg)
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=40)  # lookback for MA20
    end_ts = pd.Timestamp(end)

    close_dict = {}
    for sym, sdf in ohlcv.items():
        s = sdf["close"].loc[(sdf.index >= start_ts) & (sdf.index <= end_ts)]
        if len(s) >= 20:
            close_dict[sym] = s

    close_df = pd.DataFrame(close_dict)
    ma20 = close_df.rolling(20).mean()
    above = (close_df > ma20).mean(axis=1)
    return above.reindex(cal_idx)


def _compute_sector_returns(
    cfg: ScreenerConfig, start: str, end: str, cal_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Compute 20-day returns grouped by code prefix as a sector proxy."""
    ohlcv = _get_ohlcv_cache(cfg)
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=40)
    end_ts = pd.Timestamp(end)

    close_dict = {}
    for sym, sdf in ohlcv.items():
        s = sdf["close"].loc[(sdf.index >= start_ts) & (sdf.index <= end_ts)]
        if len(s) >= 20:
            close_dict[sym] = s

    close_df = pd.DataFrame(close_dict)

    # Group stocks by 3-digit code prefix (e.g. sh.600xxx → '600')
    sector_groups: dict[str, list[str]] = {}
    for sym in close_df.columns:
        # baostock format: 'sh.600000' → code = '600000' → prefix = '600'
        code = sym.split(".")[1] if "." in sym else sym
        prefix = code[:3]
        sector_groups.setdefault(f"sector_{prefix}", []).append(sym)

    # Keep only groups with >= 5 stocks
    sector_ret = pd.DataFrame(index=close_df.index)
    for name, syms in sector_groups.items():
        if len(syms) >= 5:
            grp = close_df[syms]
            sector_ret[name] = grp.pct_change(20).mean(axis=1)

    return sector_ret.reindex(cal_idx)


def compute_lagged_factor_ic(
    alpha158_df: pd.DataFrame,
    returns: pd.Series,
    lookback: int = 20,
    realize_lag: int = 0,
) -> pd.DataFrame:
    """Compute rolling IC (Spearman rank correlation) per factor category.

    Args:
        alpha158_df: Alpha158 features, MultiIndex (datetime, instrument).
        returns: Forward returns, same index as alpha158_df.
        lookback: Rolling window in trading days.
        realize_lag: Extra shift (in rows) so that only fully realized forward
            returns contribute.  Set to forward_horizon_days (e.g. 5) to
            guarantee no look-ahead: IC at date T uses returns[T+1:T+5],
            so shifting by 5 means lagged_ic at T only includes ICs up to
            T-5, whose returns are fully known by T.

    Returns:
        DataFrame indexed by datetime with one column per factor category.
    """
    ic_df = compute_daily_category_ic(alpha158_df, returns, min_valid=10)
    ic_df = ic_df.rename(columns={c: f"ic_{c}" for c in ic_df.columns})
    if realize_lag > 0:
        ic_df = ic_df.shift(realize_lag)
    ic_df = ic_df.fillna(0.0).rolling(lookback, min_periods=5).mean().fillna(0)
    return ic_df


# ── Raw OHLCV (for Kronos input) ────────────────────────────────────────────

def load_raw_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    cfg: ScreenerConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Load raw OHLCV data for specific symbols (Kronos input).

    Returns:
        Dict mapping symbol → DataFrame with columns
        [open, high, low, close, vol, amt].
    """
    cfg = cfg or ScreenerConfig()
    ohlcv = _get_ohlcv_cache(cfg)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    result = {}
    for sym in symbols:
        sdf = ohlcv.get(sym)
        if sdf is None:
            continue
        sdf = sdf.loc[(sdf.index >= start_ts) & (sdf.index <= end_ts)].copy()
        # Rename columns to match Kronos expected format
        sdf = sdf.rename(columns={"volume": "vol", "amount": "amt"})
        if len(sdf) >= cfg.kronos_lookback + cfg.kronos_pred_len:
            result[sym] = sdf
    return result

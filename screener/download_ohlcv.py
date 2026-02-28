"""
Download all A-share daily OHLCV via baostock.

Saves incremental progress as a pickle dict so a crash doesn't lose work.
Works locally (baostock uses its own TCP server, no eastmoney dependency).

Data layout (under data/):
    ohlcv_all_a.pkl      — Dict[str, pd.DataFrame]  keys = "sh.600000" etc.
    benchmark_000905.pkl  — pd.DataFrame  (CSI500 index)

Each DataFrame has:
    index:   DatetimeIndex named "date"
    columns: open, high, low, close, volume, amount  (float64)

Usage:
    python -m screener.download_ohlcv                 # full download
    python -m screener.download_ohlcv --test 5        # first 5 only
"""

import os
import time
import pickle
import argparse
from typing import Dict, List

import baostock as bs
import pandas as pd

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
OHLCV_PATH = os.path.join(DATA_DIR, "ohlcv_all_a.pkl")
BENCH_PATH = os.path.join(DATA_DIR, "benchmark_000905.pkl")
INDUSTRY_PATH = os.path.join(DATA_DIR, "industry_mapping.pkl")
FIELDS = "date,open,high,low,close,volume,amount"
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount"]


# ── Pickle helpers ──────────────────────────────────────────────────────────
def _save_pickle(data, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


# ── Core functions ──────────────────────────────────────────────────────────
def get_all_a_shares(day: str = "2026-02-25") -> List[str]:
    """Return baostock codes (e.g. 'sh.600000') for all listed A-shares."""
    rs = bs.query_all_stock(day=day)
    df = rs.get_data()
    mask = df["code"].str.match(r"^(sh\.6|sz\.0|sz\.3)")
    codes = df[mask]["code"].tolist()
    print(f"Found {len(codes)} A-share stocks")
    return codes


def _to_df(rs) -> pd.DataFrame:
    """Convert a baostock ResultData to a clean DataFrame."""
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def download_universe_ohlcv(
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2026-02-26",
    save_path: str = OHLCV_PATH,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV for all symbols. Resumes from existing pickle."""
    data = _load_pickle(save_path)
    if data:
        print(f"Resuming: {len(data)} stocks already in {save_path}")

    total = len(symbols)
    for i, code in enumerate(symbols, 1):
        if code in data:
            continue
        try:
            rs = bs.query_history_k_data_plus(
                code=code, fields=FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2",
            )
            df = _to_df(rs)
            if df.empty:
                print(f"[{i}/{total}] {code} — empty")
                continue
            data[code] = df
            rng = f"{df.index.min().date()} to {df.index.max().date()}"
            print(f"[{i}/{total}] {code} — {len(df)} rows ({rng})")
        except Exception as e:
            print(f"[{i}/{total}] {code} — FAILED: {e}")

        if len(data) % 100 == 0 and len(data) > 0:
            _save_pickle(data, save_path)
            print(f"  ** checkpoint ({len(data)} stocks) **")

    _save_pickle(data, save_path)
    print(f"Final save: {len(data)} stocks → {save_path}")
    return data


def download_industry_mapping(
    date: str = "2026-02-25",
    save_path: str = INDUSTRY_PATH,
) -> Dict[str, str]:
    """Download CSRC industry classification for all A-shares.

    Returns Dict[symbol, industry_code] e.g. {"sh.600000": "J66"}.
    The industry code is the 1-letter + 2-digit prefix from the CSRC
    classification (e.g. "J66货币金融服务" → "J66").
    """
    import re

    rs = bs.query_stock_industry(code="", date=date)
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        print("WARNING: query_stock_industry returned no data")
        return {}

    df = pd.DataFrame(rows, columns=rs.fields)
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        code = row.get("code", "")
        industry = row.get("industry", "")
        if not code or not industry:
            continue
        # Extract industry code prefix: letter(s) + digits, e.g. "J66" from "J66货币金融服务"
        m = re.match(r"([A-Z]\d{2})", industry)
        if m:
            mapping[code] = m.group(1)

    _save_pickle(mapping, save_path)
    print(f"Industry mapping: {len(mapping)} stocks → {save_path}")
    return mapping


def download_benchmark(
    index_code: str = "sh.000905",
    start_date: str = "2015-01-01",
    end_date: str = "2026-02-26",
    save_path: str = BENCH_PATH,
) -> pd.DataFrame:
    """Download daily OHLCV for a benchmark index."""
    rs = bs.query_history_k_data_plus(
        code=index_code, fields=FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d",
    )
    df = _to_df(rs)
    _save_pickle(df, save_path)
    print(f"Benchmark {index_code}: {len(df)} rows → {save_path}")
    return df


# ── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download A-share OHLCV")
    parser.add_argument("--test", type=int, default=0,
                        help="Download only first N stocks (0 = all)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-02-26")
    args = parser.parse_args()

    lg = bs.login()
    print(f"baostock login: {lg.error_msg}")

    symbols = get_all_a_shares()
    if args.test > 0:
        symbols = symbols[:args.test]
        print(f"TEST MODE: downloading {args.test} stocks only")

    download_universe_ohlcv(symbols, args.start, args.end)
    download_benchmark(start_date=args.start, end_date=args.end)
    download_industry_mapping()

    bs.logout()
    print("Done.")

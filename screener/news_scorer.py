"""
News / Policy Sentiment Scorer

Uses AKShare for per-stock news from East Money (东方财富) and a pretrained
Chinese financial NLP model for sentiment scoring.  Captures 0-2 day moves
driven by news/policy that pure technical indicators miss.

Output per stock-day:
  - news_sentiment:   mean sentiment score (-1 to +1)
  - news_significance: headline_count × avg confidence
  - policy_flag:      1 if any headline contains policy keywords, else 0
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig


class NewsScorer:
    """Batch sentiment scorer using FinBERT-Chinese on Colab GPU."""

    def __init__(self, cfg: ScreenerConfig | None = None, device: str = "cuda:0"):
        self.cfg = cfg or ScreenerConfig()
        self.device = device
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    # ── Lazy model loading ───────────────────────────────────────────────

    def _load_model(self):
        """Load the FinBERT model and tokenizer on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        print(f"Loading NLP model: {self.cfg.news_model_name} → {self.device}")
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self.cfg.news_model_name,
            device=self.device,
            truncation=True,
            max_length=512,
            batch_size=self.cfg.news_batch_size,
        )

    # ── News fetching ────────────────────────────────────────────────────

    @staticmethod
    def fetch_news(symbol: str, max_headlines: int = 20) -> list[str]:
        """Fetch recent news headlines for a stock via AKShare.

        Args:
            symbol: A-share stock code, e.g. "600519" or "SH600519".
            max_headlines: Maximum number of headlines to return.

        Returns:
            List of headline strings (may be empty on failure).
        """
        try:
            import akshare as ak
        except ImportError:
            warnings.warn("akshare not installed — news features disabled")
            return []

        # Normalise symbol to bare 6-digit code
        # baostock format: 'sh.600000' → '600000'
        # Qlib/bare format: 'SH600000' → '600000'
        if "." in symbol:
            code = symbol.split(".")[-1]
        else:
            code = symbol.upper().lstrip("SH").lstrip("SZ")

        try:
            df = ak.stock_news_em(symbol=code)
        except Exception:
            return []

        if df is None or df.empty:
            return []

        # AKShare returns columns like: 新闻标题, 新闻内容, 发布时间, ...
        title_col = "新闻标题" if "新闻标题" in df.columns else df.columns[0]
        headlines = df[title_col].dropna().tolist()
        return headlines[:max_headlines]

    # ── Sentiment scoring ────────────────────────────────────────────────

    def _score_headlines(self, headlines: list[str]) -> list[dict]:
        """Run sentiment inference on a batch of headlines.

        Returns list of dicts with keys: label, score.
        """
        if not headlines:
            return []
        self._load_model()
        results = self._pipeline(headlines)
        return results

    @staticmethod
    def _label_to_numeric(label: str) -> float:
        """Convert FinBERT label to numeric score in [-1, 1]."""
        label_lower = label.lower()
        if "positive" in label_lower:
            return 1.0
        if "negative" in label_lower:
            return -1.0
        return 0.0  # neutral

    def _has_policy_keywords(self, headlines: list[str]) -> bool:
        for headline in headlines:
            for kw in self.cfg.policy_keywords:
                if kw in headline:
                    return True
        return False

    # ── Public API ───────────────────────────────────────────────────────

    def score_stock(self, symbol: str) -> dict:
        """Score a single stock's news sentiment.

        Returns:
            Dict with keys: news_sentiment, news_significance, policy_flag.
        """
        headlines = self.fetch_news(symbol, self.cfg.news_max_headlines)
        if not headlines:
            return {"news_sentiment": 0.0, "news_significance": 0.0, "policy_flag": 0}

        results = self._score_headlines(headlines)
        scores = [self._label_to_numeric(r["label"]) for r in results]
        confidences = [r["score"] for r in results]

        sentiment = float(np.mean(scores))
        significance = len(headlines) * float(np.mean(confidences))
        policy = int(self._has_policy_keywords(headlines))

        return {
            "news_sentiment": sentiment,
            "news_significance": significance,
            "policy_flag": policy,
        }

    def score_batch(self, symbols: list[str]) -> pd.DataFrame:
        """Score multiple stocks in batch.

        Returns:
            DataFrame indexed by symbol with columns:
            news_sentiment, news_significance, policy_flag.
        """
        # Collect all headlines with their symbol mapping
        all_headlines: list[str] = []
        symbol_map: list[tuple[str, int, int]] = []  # (symbol, start_idx, count)

        for sym in symbols:
            headlines = self.fetch_news(sym, self.cfg.news_max_headlines)
            start = len(all_headlines)
            all_headlines.extend(headlines)
            symbol_map.append((sym, start, len(headlines)))

        # Batch inference on all headlines at once
        if all_headlines:
            all_results = self._score_headlines(all_headlines)
        else:
            all_results = []

        # Aggregate per symbol
        records = []
        for sym, start, count in symbol_map:
            if count == 0:
                records.append({
                    "symbol": sym,
                    "news_sentiment": 0.0,
                    "news_significance": 0.0,
                    "policy_flag": 0,
                })
                continue

            sym_results = all_results[start:start + count]
            sym_headlines = all_headlines[start:start + count]
            scores = [self._label_to_numeric(r["label"]) for r in sym_results]
            confidences = [r["score"] for r in sym_results]

            records.append({
                "symbol": sym,
                "news_sentiment": float(np.mean(scores)),
                "news_significance": count * float(np.mean(confidences)),
                "policy_flag": int(self._has_policy_keywords(sym_headlines)),
            })

        df = pd.DataFrame(records).set_index("symbol")
        return df

    def cleanup(self):
        """Free GPU memory."""
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

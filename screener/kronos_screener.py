"""
Layer 3: Kronos Prediction  (30 → ~5 stocks)

Wraps the existing ``KronosPredictor`` to run predictions on the 30 survivors
from Layer 2.  Ranks by predicted_return × confidence, keeps top 5.
"""

import numpy as np
import pandas as pd
import torch

from screener.config import ScreenerConfig


class KronosScreener:
    """Layer 3: predict 10-day candles for ~30 stocks, rank, keep top 5."""

    def __init__(self, cfg: ScreenerConfig | None = None, device: str = "cuda:0"):
        self.cfg = cfg or ScreenerConfig()
        self.device = device
        self.predictor = None

    # ── Model Loading ────────────────────────────────────────────────────

    def load_model(self):
        """Load Kronos tokenizer and predictor onto device."""
        from screener.kronos import KronosTokenizer, Kronos, KronosPredictor

        print(f"Loading Kronos models → {self.device}")
        tokenizer = KronosTokenizer.from_pretrained(
            self.cfg.kronos_tokenizer_path
        ).to(self.device).eval()
        model = Kronos.from_pretrained(
            self.cfg.kronos_predictor_path
        ).to(self.device).eval()

        self.predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            max_context=self.cfg.kronos_max_context,
            clip=self.cfg.kronos_clip,
        )
        print("Kronos models loaded.")

    def unload_model(self):
        """Move models off GPU and free memory."""
        if self.predictor is not None:
            self.predictor.tokenizer.cpu()
            self.predictor.model.cpu()
            self.predictor = None
        torch.cuda.empty_cache()

    # ── Single-Stock Prediction ──────────────────────────────────────────

    def predict_stock(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.DatetimeIndex,
        y_timestamp: pd.DatetimeIndex,
    ) -> dict:
        """Run Kronos prediction for a single stock with confidence scoring.

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, vol/volume, amt).
            x_timestamp: Timestamps for the lookback context.
            y_timestamp: Timestamps for the prediction horizon.

        Returns:
            Dict with keys:
              - pred_df: DataFrame of predicted candles
              - predicted_return: (pred_close[-1] / current_close) - 1
              - confidence: 1 / (1 + std_across_samples), higher = more agreement
              - sample_preds: full per-sample predictions (n_samples, pred_len, features)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call .load_model() first.")

        cfg = self.cfg

        # Prepare input data (same as KronosPredictor.predict)
        price_cols = ["open", "high", "low", "close"]
        vol_col = "vol" if "vol" in df.columns else "volume"
        amt_col = "amt" if "amt" in df.columns else "amount"

        df = df.copy()
        if amt_col not in df.columns:
            df["amt"] = df[price_cols].mean(axis=1) * df[vol_col]
            amt_col = "amt"

        feature_cols = price_cols + [vol_col, amt_col]
        x = df[feature_cols].values.astype(np.float32)

        # Normalisation (instance-level, same as KronosPredictor)
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_normed = (x - x_mean) / (x_std + 1e-5)
        x_normed = np.clip(x_normed, -cfg.kronos_clip, cfg.kronos_clip)

        # Timestamps
        from screener.kronos import calc_time_stamps
        x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

        # Expand batch dim
        x_tensor = torch.from_numpy(x_normed[np.newaxis]).to(self.device)
        x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis]).to(self.device)
        y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis]).to(self.device)

        # Run inference with return_samples=True
        from screener.kronos import auto_regressive_inference
        sample_preds = auto_regressive_inference(
            self.predictor.tokenizer,
            self.predictor.model,
            x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context=cfg.kronos_max_context,
            pred_len=cfg.kronos_pred_len,
            clip=cfg.kronos_clip,
            T=cfg.kronos_T,
            top_k=cfg.kronos_top_k,
            top_p=cfg.kronos_top_p,
            sample_count=cfg.kronos_sample_count,
            return_samples=True,
        )
        # sample_preds shape: (1, sample_count, seq_len, features)
        sample_preds = sample_preds[0]  # (sample_count, seq_len, features)

        # Take only the predicted portion (last pred_len steps)
        sample_preds = sample_preds[:, -cfg.kronos_pred_len:, :]

        # Denormalise
        sample_preds = sample_preds * (x_std + 1e-5) + x_mean

        # Averaged prediction
        avg_pred = np.mean(sample_preds, axis=0)  # (pred_len, features)
        pred_df = pd.DataFrame(
            avg_pred, columns=feature_cols, index=y_timestamp[:cfg.kronos_pred_len]
        )

        # Metrics
        current_close = df["close"].iloc[-1]
        pred_closes = sample_preds[:, -1, 3]  # close is index 3
        mean_pred_close = np.mean(pred_closes)
        std_pred_close = np.std(pred_closes)

        predicted_return = mean_pred_close / current_close - 1
        confidence = 1.0 / (1.0 + std_pred_close / (current_close + 1e-9))

        return {
            "pred_df": pred_df,
            "predicted_return": float(predicted_return),
            "confidence": float(confidence),
            "sample_preds": sample_preds,
        }

    # ── Batch Screening ──────────────────────────────────────────────────

    def screen_stocks(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run Kronos on symbols, return ranked results.

        Args:
            ohlcv_dict: symbol → full OHLCV DataFrame.
            symbols: Stocks to evaluate (typically ~30 from Layer 2).
            date: Current evaluation date.

        Returns:
            DataFrame with columns: predicted_return, confidence, composite_score,
            indexed by symbol, sorted by composite_score descending.
        """
        if self.predictor is None:
            self.load_model()

        cfg = self.cfg
        from screener.data_pipeline import get_full_calendar
        cal = get_full_calendar(cfg)

        results = []
        for sym in symbols:
            df = ohlcv_dict.get(sym)
            if df is None:
                continue

            # Slice up to date
            df = df.loc[:date]
            if len(df) < cfg.kronos_lookback:
                continue

            # Use last lookback_window rows
            context = df.iloc[-cfg.kronos_lookback:]
            x_timestamp = pd.DatetimeIndex(context.index)

            # Build y_timestamp: next pred_len trading days after date
            date_idx = cal.searchsorted(date)
            y_end_idx = min(date_idx + cfg.kronos_pred_len, len(cal) - 1)
            y_timestamp = cal[date_idx + 1:y_end_idx + 1]
            if len(y_timestamp) < cfg.kronos_pred_len:
                # Pad with business days if calendar is too short
                last = y_timestamp[-1] if len(y_timestamp) > 0 else date
                while len(y_timestamp) < cfg.kronos_pred_len:
                    last = last + pd.tseries.offsets.BDay(1)
                    y_timestamp = y_timestamp.append(pd.DatetimeIndex([last]))

            try:
                result = self.predict_stock(context, x_timestamp, y_timestamp)
                results.append({
                    "symbol": sym,
                    "predicted_return": result["predicted_return"],
                    "confidence": result["confidence"],
                    "pred_df": result["pred_df"],
                    "sample_preds": result["sample_preds"],
                })
            except Exception as e:
                print(f"  Kronos failed for {sym}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        scores_df = pd.DataFrame([
            {
                "symbol": r["symbol"],
                "predicted_return": r["predicted_return"],
                "confidence": r["confidence"],
            }
            for r in results
        ]).set_index("symbol")

        # Composite score: predicted_return weighted by confidence
        scores_df["composite_score"] = (
            scores_df["predicted_return"] * scores_df["confidence"]
        )
        scores_df = scores_df.sort_values("composite_score", ascending=False)

        # Store full predictions for paper trader exit rules
        self._last_predictions = {r["symbol"]: r for r in results}

        return scores_df

    def select_top(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
    ) -> list[str]:
        """Return top-N symbols after Kronos scoring."""
        scores = self.screen_stocks(ohlcv_dict, symbols, date)
        return list(scores.head(self.cfg.layer3_top_n).index)

    def get_prediction(self, symbol: str) -> dict | None:
        """Get the last prediction result for a symbol (for paper trader)."""
        return getattr(self, "_last_predictions", {}).get(symbol)

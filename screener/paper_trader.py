"""
Layer 4: Paper Trading Engine

Full-capital-per-trade paper trader with fixed TP/SL exit rules,
A-share lot rounding, commission fees, and 涨停/跌停 (limit) handling.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.utils import get_board_type, get_limit_threshold


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    hold_days: int = 0


@dataclass
class Trade:
    symbol: str
    action: str         # "buy" or "sell"
    date: pd.Timestamp
    price: float
    shares: int
    commission: float
    pnl: float = 0.0   # for sell trades
    reason: str = ""    # exit reason


class PaperTrader:
    """Paper trading engine with fixed TP/SL exit rules."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.cash = self.cfg.initial_capital
        self.position: Position | None = None
        self.trade_log: list[Trade] = []
        self.daily_nav: list[tuple[pd.Timestamp, float]] = []  # (date, NAV)

    # ── Commission ───────────────────────────────────────────────────────

    def _buy_cost(self, price: float, shares: int) -> float:
        return price * shares * self.cfg.buy_commission

    def _sell_cost(self, price: float, shares: int) -> float:
        comm = price * shares * self.cfg.sell_commission
        tax = price * shares * self.cfg.stamp_tax
        return comm + tax

    # ── Limit-Price Checks ───────────────────────────────────────────────

    def _is_limit_up_open(
        self, symbol: str, open_price: float, prev_close: float
    ) -> bool:
        """Check if stock opens at 涨停 (can't buy)."""
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (open_price / prev_close - 1) >= threshold - 0.001  # small tolerance

    def _is_limit_down_open(
        self, symbol: str, open_price: float, prev_close: float
    ) -> bool:
        """Check if stock opens at 跌停 (can't sell)."""
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (open_price / prev_close - 1) <= -threshold + 0.001

    def _is_yizi_ban(self, row: pd.Series) -> bool:
        """Check for 一字板: open == high == low == close."""
        return (
            abs(row["open"] - row["high"]) < 0.001
            and abs(row["high"] - row["low"]) < 0.001
            and abs(row["low"] - row["close"]) < 0.001
        )

    # ── Buy / Sell ───────────────────────────────────────────────────────

    def buy(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        prev_close: float | None = None,
    ) -> bool:
        """Attempt to buy a stock with full capital.

        Returns True if trade executed, False if blocked (涨停/insufficient cash).
        """
        if self.position is not None:
            return False  # already holding

        # Limit-up check
        if prev_close is not None and self._is_limit_up_open(symbol, price, prev_close):
            return False

        # Calculate shares (round down to lots of 100)
        max_cost = self.cash
        raw_shares = int(max_cost / price)
        shares = (raw_shares // self.cfg.lot_size) * self.cfg.lot_size
        if shares <= 0:
            return False

        cost = price * shares
        commission = self._buy_cost(price, shares)
        total_cost = cost + commission

        if total_cost > self.cash:
            shares -= self.cfg.lot_size
            if shares <= 0:
                return False
            cost = price * shares
            commission = self._buy_cost(price, shares)
            total_cost = cost + commission

        self.cash -= total_cost
        self.position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=date,
            hold_days=0,
        )
        self.trade_log.append(Trade(
            symbol=symbol, action="buy", date=date, price=price,
            shares=shares, commission=commission, reason="entry",
        ))
        return True

    def sell(
        self,
        date: pd.Timestamp,
        price: float,
        reason: str = "",
    ) -> bool:
        """Sell current position at given price.

        Returns True if executed, False if no position.
        """
        if self.position is None:
            return False

        shares = self.position.shares
        revenue = price * shares
        commission = self._sell_cost(price, shares)
        net_revenue = revenue - commission

        pnl = net_revenue - (self.position.entry_price * shares + self._buy_cost(
            self.position.entry_price, shares
        ))

        self.cash += net_revenue
        self.trade_log.append(Trade(
            symbol=self.position.symbol, action="sell", date=date,
            price=price, shares=shares, commission=commission,
            pnl=pnl, reason=reason,
        ))
        self.position = None
        return True

    # ── Exit Rule Evaluation ─────────────────────────────────────────────

    def _check_exit_rules(
        self, today_row: pd.Series,
    ) -> tuple[str | None, float | None]:
        """Check TP/SL exit rules against today's OHLCV bar.

        Returns (exit_reason, fill_price) or (None, None) if no exit.
        SL checked first (conservative — assume worst case when both hit).
        """
        pos = self.position
        if pos is None or pos.hold_days < 1:  # T+1: can't sell on buy day
            return None, None

        entry = pos.entry_price
        tp_price = entry * (1 + self.cfg.tp_pct)
        sl_price = entry * (1 - self.cfg.sl_pct)
        today_high = today_row.get("high", today_row.get("$high", float("inf")))
        today_low = today_row.get("low", today_row.get("$low", 0))
        today_open = today_row.get("open", today_row.get("$open", 0))
        today_close = today_row.get("close", today_row.get("$close", 0))

        # SL first (conservative)
        if today_low <= sl_price:
            return "stop_loss", min(sl_price, today_open)  # gap-down → fill at open

        # TP
        if today_high >= tp_price:
            return "take_profit", max(tp_price, today_open)  # gap-up → fill at open

        # Time limit
        if pos.hold_days >= self.cfg.max_hold_days:
            return "time_limit", today_close

        return None, None

    # ── Daily Update ─────────────────────────────────────────────────────

    def daily_update(
        self,
        date: pd.Timestamp,
        ranked_symbols: list[str],
        ohlcv_today: dict[str, pd.Series],
        ohlcv_prev: dict[str, pd.Series] | None = None,
    ):
        """Process one trading day.

        1. If holding: check TP/SL/time exit against today's bar. Sell if triggered.
        2. If no position: buy the top-ranked stock at today's open.

        Args:
            date: Today's date.
            ranked_symbols: Symbols ranked by the pipeline (best first).
            ohlcv_today: symbol → Series with open/high/low/close/volume for today.
            ohlcv_prev: symbol → Series for previous day (for limit checks).
        """
        prev_close_map = {}
        if ohlcv_prev:
            for sym, row in ohlcv_prev.items():
                prev_close_map[sym] = row.get("close", 0)

        # ── Step 1: Check exits for held position ────────────────────────
        if self.position is not None:
            sym = self.position.symbol
            today_row = ohlcv_today.get(sym)

            # Suspended stocks (volume=0) are untradeable
            if today_row is not None and today_row.get("volume", today_row.get("vol", 0)) <= 0:
                today_row = None

            if today_row is not None:
                exit_reason, fill_price = self._check_exit_rules(today_row)
                if exit_reason and fill_price is not None:
                    prev_cl = prev_close_map.get(sym, 0)
                    # Check 一字板 / 跌停 before selling
                    if self._is_yizi_ban(today_row):
                        pass  # can't sell, carry to next day
                    elif prev_cl > 0 and self._is_limit_down_open(sym, today_row.get("open", 0), prev_cl):
                        pass  # limit down at open, can't sell
                    else:
                        self.sell(date, fill_price, exit_reason)

                # Increment hold_days after exit check
                if self.position is not None:
                    self.position.hold_days += 1
            else:
                # No data / suspended stock today — increment hold_days
                if self.position is not None:
                    self.position.hold_days += 1

        # ── Step 2: Buy if no position ───────────────────────────────────
        if self.position is None and ranked_symbols:
            for sym in ranked_symbols:
                today_row = ohlcv_today.get(sym)
                if today_row is None:
                    continue

                # Skip suspended stocks (volume=0)
                if today_row.get("volume", today_row.get("vol", 0)) <= 0:
                    continue

                open_price = today_row.get("open", today_row.get("$open", 0))
                if open_price <= 0:
                    continue

                prev_cl = prev_close_map.get(sym, 0)

                # Check 一字板 at 涨停
                if self._is_yizi_ban(today_row) and self._is_limit_up_open(
                    sym, open_price, prev_cl
                ):
                    continue

                bought = self.buy(date, sym, open_price, prev_cl)
                if bought:
                    break

        # ── Record NAV ───────────────────────────────────────────────────
        nav = self.cash
        if self.position is not None:
            sym = self.position.symbol
            today_row = ohlcv_today.get(sym)
            if today_row is not None:
                close = today_row.get("close", today_row.get("$close", 0))
                nav += close * self.position.shares
        self.daily_nav.append((date, nav))

    # ── Metrics ──────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Compute performance metrics from trade log and NAV history."""
        trades = self.trade_log
        sells = [t for t in trades if t.action == "sell"]

        if not sells:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
                "total_pnl": 0.0,
                "total_commission": sum(t.commission for t in trades),
            }

        wins = sum(1 for t in sells if t.pnl > 0)
        total_pnl = sum(t.pnl for t in sells)
        total_comm = sum(t.commission for t in trades)

        # NAV-based metrics
        nav_series = pd.Series(
            {d: v for d, v in self.daily_nav},
            name="nav",
        ).sort_index()

        if len(nav_series) < 2:
            return {
                "total_return": total_pnl / self.cfg.initial_capital,
                "win_rate": wins / len(sells),
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trade_count": len(sells),
                "total_pnl": total_pnl,
                "total_commission": total_comm,
            }

        daily_returns = nav_series.pct_change().dropna()
        total_return = nav_series.iloc[-1] / self.cfg.initial_capital - 1

        # Annualised Sharpe (252 trading days)
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = drawdown.min()

        return {
            "total_return": float(total_return),
            "win_rate": wins / len(sells),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "trade_count": len(sells),
            "total_pnl": float(total_pnl),
            "total_commission": float(total_comm),
        }

    def get_nav_series(self) -> pd.Series:
        """Return daily NAV as a Series."""
        return pd.Series(
            {d: v for d, v in self.daily_nav}, name="nav"
        ).sort_index()

    def reset(self):
        """Reset the trader to initial state."""
        self.cash = self.cfg.initial_capital
        self.position = None
        self.trade_log = []
        self.daily_nav = []

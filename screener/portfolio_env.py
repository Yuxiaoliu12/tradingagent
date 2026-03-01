"""
Layer 4: RL Portfolio Environment (Gymnasium)

PPO environment for 3-stock portfolio management with discrete position
sizing and per-stock execution timing.  Uses L2 screening signals as
observations and enforces A-share trading constraints (T+1, limit-up/down,
suspension, 一字板).  Supports action masking via ``action_masks()`` for
use with sb3-contrib MaskablePPO.
"""

import random
from itertools import product

import gymnasium
import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.utils import get_limit_threshold

# ── Observation layout ────────────────────────────────────────────────────────
# Per-slot tech features extracted from L2 feature cache
_SLOT_TECH_FEATURES = [
    "mom_5", "mom_20", "rsi_14", "bb_position", "volume_trend", "atr_14",
]
_N_SLOT_FEATURES = 12   # l2_upside + l2_downside + 6 tech + 4 position state
_N_GLOBAL_FEATURES = 4
# Total: 3 × 12 + 4 = 40


def _obs_dim(n_slots: int) -> int:
    return _N_SLOT_FEATURES * n_slots + _N_GLOBAL_FEATURES


# ── Action table ──────────────────────────────────────────────────────────────

def build_action_table(n_slots: int = 3, weight_steps: int = 3) -> list[tuple]:
    """Enumerate all valid (w0, w1, w2, t0, t1, t2) action tuples.

    For n_slots=3, weight_steps=3 this produces 63 actions:
      1 all-cash + 18 one-stock + 36 two-stock + 8 three-stock.
    """
    actions = []
    for weights in product(range(weight_steps + 1), repeat=n_slots):
        if sum(weights) > weight_steps:
            continue
        frac_weights = tuple(w / weight_steps for w in weights)
        non_zero = [i for i in range(n_slots) if weights[i] > 0]
        if not non_zero:
            actions.append(frac_weights + (None,) * n_slots)
        else:
            for timing_combo in product(["open", "close"], repeat=len(non_zero)):
                timings = [None] * n_slots
                for j, slot_idx in enumerate(non_zero):
                    timings[slot_idx] = timing_combo[j]
                actions.append(frac_weights + tuple(timings))
    return actions


# ── Environment ───────────────────────────────────────────────────────────────

class PortfolioEnv(gymnasium.Env):
    """PPO environment for 3-stock portfolio management with action masking.

    Action:  index into precomputed table of (weight, timing) combos.
    Obs:     per-slot features (12 each) + global features (4) = 40 dims.
    Reward:  rolling 5-day log return × 10.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ScreenerConfig,
        daily_signals: list[dict],
        ohlcv_dict: dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        training_mode: bool = True,
        candidate_mode: str = "top",
    ):
        """
        Args:
            cfg: ScreenerConfig with RL hyperparameters.
            daily_signals: list of dicts, one per trading day, with keys:
                date, l2_scores (Series), l2_ranking (list[str]),
                l2_features (DataFrame symbol × tech features).
            ohlcv_dict: {symbol: DataFrame with open/high/low/close/volume}.
            benchmark_df: DataFrame with benchmark OHLCV (needs 'close').
            training_mode: if True, randomly sample candidates from L2 top-N.
            candidate_mode: how to pick candidates during inference:
                "top" — best from L2 ranking (default);
                "random_l2" — random from L2 top 30;
                "bottom_l2" — worst of L2 top 30;
                "random_l1" — random from full L1 pool.
        """
        super().__init__()
        self.cfg = cfg
        self._daily_signals = daily_signals
        self._ohlcv_dict = ohlcv_dict
        self._training_mode = training_mode
        self._candidate_mode = candidate_mode
        self._n_slots = cfg.rl_n_slots

        # Action table
        self._action_table = build_action_table(self._n_slots, cfg.rl_weight_steps)

        # Spaces
        dim = _obs_dim(self._n_slots)
        self.action_space = gymnasium.spaces.Discrete(len(self._action_table))
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32,
        )

        # Benchmark close series for market_ret_20d
        if "close" in benchmark_df.columns:
            self._bench_close = benchmark_df["close"].sort_index()
        else:
            self._bench_close = benchmark_df.iloc[:, 0].sort_index()

        # State (initialised in reset)
        self._day_idx: int = 0
        self._cash: float = 0.0
        self._nav: float = 0.0
        self._nav_peak: float = 0.0
        self._nav_history: list[float] = []
        self._holdings: dict[str, dict] = {}   # sym → {shares, entry_price, hold_days}
        self._weights: dict[str, float] = {}   # sym → drifted weight
        self._active_slots: list[str | None] = []
        self.trade_log: list[dict] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._day_idx = 0
        self._cash = self.cfg.initial_capital
        self._nav = self.cfg.initial_capital
        self._nav_peak = self.cfg.initial_capital
        self._nav_history = [self.cfg.initial_capital]
        self._holdings = {}
        self._weights = {}
        self.trade_log = []

        self._active_slots = self._get_active_slots()
        obs = self._build_obs()
        return obs, {}

    def step(self, action_idx: int):
        action = self._action_table[action_idx]
        target_weights = list(action[: self._n_slots])
        timings = list(action[self._n_slots :])

        # Slots come from the PREVIOUS observation — the stocks the agent saw
        slots = list(self._active_slots)

        # Advance to next trading day
        self._day_idx += 1
        dim = _obs_dim(self._n_slots)
        if self._day_idx >= len(self._daily_signals):
            reward = self._compute_reward()
            return np.zeros(dim, dtype=np.float32), reward, True, False, {}

        date = self._daily_signals[self._day_idx]["date"]

        # Increment hold_days (enables T+1 rule)
        for h in self._holdings.values():
            h["hold_days"] += 1

        info: dict = {"date": date, "blocked_trades": [], "substituted_trades": []}
        ref_nav = self._nav  # reference NAV for allocation targets

        # ── Process SELLS first (free capital) ────────────────────────
        for i, sym in enumerate(slots):
            if sym is None or sym not in self._holdings:
                continue
            tw = target_weights[i]
            h = self._holdings[sym]

            timing = timings[i] if timings[i] is not None else "close"
            sell_price = self._get_price(sym, date, timing)
            if sell_price is None or sell_price <= 0:
                continue

            current_value = h["shares"] * sell_price
            target_value = tw * ref_nav

            if target_value >= current_value:
                continue  # no sell needed

            # Compute shares to sell
            if tw == 0:
                shares_to_sell = h["shares"]
            else:
                target_shares = int(target_value / sell_price)
                target_shares = (target_shares // self.cfg.lot_size) * self.cfg.lot_size
                shares_to_sell = h["shares"] - target_shares
                shares_to_sell = (shares_to_sell // self.cfg.lot_size) * self.cfg.lot_size
                if shares_to_sell <= 0:
                    continue

            # Constraint check
            if not self._can_sell(sym, date, timing):
                info["blocked_trades"].append(
                    {"symbol": sym, "action": "sell", "reason": "constraint"}
                )
                continue

            # Execute sell
            proceeds = shares_to_sell * sell_price
            commission = proceeds * (self.cfg.sell_commission + self.cfg.stamp_tax)
            self._cash += proceeds - commission

            pnl = (sell_price - h["entry_price"]) * shares_to_sell - commission
            self.trade_log.append({
                "date": date, "symbol": sym, "action": "sell",
                "price": sell_price, "shares": shares_to_sell,
                "commission": commission, "pnl": pnl,
                "hold_days": h["hold_days"], "timing": timing,
                "prev_close": self._get_prev_close(sym, date),
            })

            h["shares"] -= shares_to_sell
            if h["shares"] <= 0:
                del self._holdings[sym]

        # ── Process BUYS (with L2 fallback) ─────────────────────────
        bought_syms: set[str] = set()
        for i, sym in enumerate(slots):
            if sym is None:
                continue
            tw = target_weights[i]
            if tw <= 0:
                continue

            timing = timings[i] if timings[i] is not None else "open"
            target_value = tw * ref_nav

            self._try_buy_with_fallback(
                sym, date, timing, target_value, ref_nav, bought_syms, info,
            )

        # ── Mark-to-market at close ───────────────────────────────────
        holdings_value = 0.0
        close_prices: dict[str, float] = {}
        for sym, h in list(self._holdings.items()):
            cp = self._get_price(sym, date, "close")
            if cp is not None and cp > 0:
                close_prices[sym] = cp
                holdings_value += h["shares"] * cp
            else:
                close_prices[sym] = h["entry_price"]
                holdings_value += h["shares"] * h["entry_price"]

        self._nav = self._cash + holdings_value
        self._nav_history.append(self._nav)
        self._nav_peak = max(self._nav_peak, self._nav)

        # Drift weights
        self._weights = {}
        if self._nav > 0:
            for sym, h in self._holdings.items():
                self._weights[sym] = h["shares"] * close_prices.get(
                    sym, h["entry_price"]
                ) / self._nav

        # Reward
        reward = self._compute_reward()

        # Prepare next observation
        self._active_slots = self._get_active_slots()
        obs = self._build_obs()

        terminated = self._day_idx >= len(self._daily_signals) - 1
        return obs, reward, terminated, False, info

    # ── Active Slot Management ────────────────────────────────────────

    def _get_active_slots(self) -> list[str | None]:
        """Fill slots: held stocks first (by hold_days desc), then L2 candidates."""
        if self._day_idx >= len(self._daily_signals):
            return [None] * self._n_slots

        signals = self._daily_signals[self._day_idx]
        l2_ranking = signals.get("l2_ranking", [])

        slots: list[str | None] = []

        # Held stocks sorted by hold_days descending
        held_sorted = sorted(
            self._holdings.keys(),
            key=lambda s: self._holdings[s]["hold_days"],
            reverse=True,
        )
        for sym in held_sorted:
            if len(slots) < self._n_slots:
                slots.append(sym)

        # Fill remaining with L2 candidates (exclude held)
        remaining = self._n_slots - len(slots)
        if remaining > 0:
            held_set = set(self._holdings.keys())
            candidates = [s for s in l2_ranking if s not in held_set]
            top_n = self.cfg.layer2_top_n
            top_candidates = candidates[:top_n]

            if self._training_mode:
                # Training: random from top-N (unchanged)
                if len(top_candidates) > remaining:
                    selected = random.sample(top_candidates, remaining)
                else:
                    selected = top_candidates[:remaining]
            elif self._candidate_mode == "random_l2":
                pool = candidates[:30]
                k = min(len(pool), remaining)
                selected = random.sample(pool, k) if k > 0 else []
            elif self._candidate_mode == "bottom_l2":
                pool = candidates[:30]
                selected = list(reversed(pool))[:remaining]
            elif self._candidate_mode == "random_l1":
                k = min(len(candidates), remaining)
                selected = random.sample(candidates, k) if k > 0 else []
            else:  # "top" (default)
                selected = top_candidates[:remaining]
            slots.extend(selected)

        # Pad with None
        while len(slots) < self._n_slots:
            slots.append(None)

        return slots

    # ── Action Masking ─────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Boolean mask over all actions. Called BEFORE step().

        Only masks based on information knowable at day T (the observation
        day).  Market constraints on day T+1 (suspension, limit-up/down,
        一字板) are NOT checked here — step() silently skips blocked
        trades, and the agent learns from the consequence.

        Masked conditions (day-T knowable):
        - Empty slot: can't assign positive weight to a slot with no stock.
        - T+1 rule: can't sell a stock with hold_days == 0 (bought today;
          after the increment in step() it will be 1, but that's the
          minimum for selling — hold_days == 0 at mask time means the
          stock was just bought this step and can't be sold next step).
        """
        n_actions = len(self._action_table)
        mask = np.ones(n_actions, dtype=bool)
        slots = list(self._active_slots)

        for act_i, action in enumerate(self._action_table):
            weights = action[: self._n_slots]

            legal = True
            for slot_i in range(self._n_slots):
                sym = slots[slot_i]
                tw = weights[slot_i]
                is_held = sym is not None and sym in self._holdings

                if sym is None and tw > 0:
                    legal = False
                    break

                if is_held and tw == 0:
                    # T+1: hold_days hasn't been incremented yet.
                    # hold_days == 0 → bought today → after +1 it's 1,
                    # which is the minimum for _can_sell (hold_days >= 1).
                    # So hold_days == 0 at mask time is OK to sell.
                    # Only block if hold_days < 0 (shouldn't happen).
                    h = self._holdings[sym]
                    if h["hold_days"] < 0:
                        legal = False
                        break

            if not legal:
                mask[act_i] = False

        if not mask.any():
            mask[:] = True

        return mask

    # ── Observation Builder ───────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Build observation vector (40-dim for 3 slots)."""
        dim = _obs_dim(self._n_slots)
        if self._day_idx >= len(self._daily_signals):
            return np.zeros(dim, dtype=np.float32)

        signals = self._daily_signals[self._day_idx]
        date = signals["date"]
        l2_upside = signals.get("l2_upside", pd.Series(dtype=float))
        l2_downside = signals.get("l2_downside", pd.Series(dtype=float))
        l2_features = signals.get("l2_features", pd.DataFrame())

        # Z-score upside and downside independently within the day
        def _zscore(s: pd.Series) -> pd.Series:
            if len(s) > 1:
                mu, sigma = s.mean(), s.std()
                if sigma > 1e-9:
                    return ((s - mu) / sigma).clip(-3, 3)
            return s * 0

        upside_z = _zscore(l2_upside)
        downside_z = _zscore(l2_downside)

        obs = np.zeros(dim, dtype=np.float32)

        for i, sym in enumerate(self._active_slots):
            if sym is None:
                continue
            off = i * _N_SLOT_FEATURES

            # 1-2. L2 upside + downside (standardised)
            if sym in upside_z.index:
                obs[off] = float(upside_z[sym])
            if sym in downside_z.index:
                obs[off + 1] = float(downside_z[sym])

            # 3-8. Tech features (normalised to ~[-1, 1])
            if sym in l2_features.index:
                row = l2_features.loc[sym]
                close_price = self._get_price(sym, date, "close")
                for j, feat in enumerate(_SLOT_TECH_FEATURES):
                    val = row.get(feat, 0.0)
                    if val != val:  # NaN check
                        val = 0.0
                    else:
                        val = float(val)
                        if feat == "rsi_14":
                            val = val / 100.0                        # [0, 1]
                        elif feat in ("mom_5", "mom_20"):
                            val = np.clip(val, -0.3, 0.3) / 0.3     # [-1, 1]
                        elif feat == "atr_14":
                            if close_price and close_price > 0:
                                val = val / close_price              # relative ATR
                            val = np.clip(val, 0.0, 0.1) * 10.0     # [0, 1]
                        elif feat == "volume_trend":
                            val = np.clip(val, -2.0, 2.0) / 2.0     # [-1, 1]
                    obs[off + 2 + j] = val

            # 9-12. Position features
            h = self._holdings.get(sym)
            if h:
                obs[off + 8] = 1.0  # is_held
                obs[off + 9] = self._weights.get(sym, 0.0)  # current_weight
                obs[off + 10] = min(h["hold_days"] / 20.0, 1.0)  # hold_days_norm
                cp = self._get_price(sym, date, "close")
                if cp and h["entry_price"] > 0:
                    pnl = cp / h["entry_price"] - 1.0
                    obs[off + 11] = np.clip(pnl, -0.5, 0.5) / 0.5  # [-1, 1]

        # Global features
        g = self._n_slots * _N_SLOT_FEATURES
        total_held_w = sum(self._weights.values())
        obs[g] = 1.0 - total_held_w                      # cash_weight
        obs[g + 1] = (                                    # portfolio_drawdown
            (self._nav_peak - self._nav) / self._nav_peak
            if self._nav_peak > 0 else 0.0
        )
        obs[g + 2] = self._get_market_ret_20d(date)       # market_ret_20d
        obs[g + 3] = len(self._holdings) / self._n_slots  # n_held_norm

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Rolling 5-day log return × 10."""
        t = len(self._nav_history) - 1
        w = self.cfg.rl_reward_window
        t_ref = max(0, t - w)
        nav_ref = self._nav_history[t_ref]
        if nav_ref <= 0 or self._nav <= 0:
            return 0.0
        return 10.0 * float(np.log(self._nav / nav_ref))

    # ── Price Helpers ─────────────────────────────────────────────────

    def _get_price(
        self, symbol: str, date: pd.Timestamp, timing: str
    ) -> float | None:
        ohlcv = self._ohlcv_dict.get(symbol)
        if ohlcv is None or date not in ohlcv.index:
            return None
        row = ohlcv.loc[date]
        col = "open" if timing == "open" else "close"
        price = row[col]
        return float(price) if not (price != price) else None  # NaN check

    def _get_prev_close(self, symbol: str, date: pd.Timestamp) -> float | None:
        ohlcv = self._ohlcv_dict.get(symbol)
        if ohlcv is None:
            return None
        prev_dates = ohlcv.index[ohlcv.index < date]
        if len(prev_dates) == 0:
            return None
        return float(ohlcv.loc[prev_dates[-1], "close"])

    def _get_market_ret_20d(self, date: pd.Timestamp) -> float:
        if date not in self._bench_close.index:
            return 0.0
        loc = self._bench_close.index.get_loc(date)
        ref_loc = max(0, loc - 20)
        ref_val = self._bench_close.iloc[ref_loc]
        if ref_val <= 0:
            return 0.0
        return float(self._bench_close.iloc[loc] / ref_val - 1.0)

    # ── A-Share Constraint Checks ─────────────────────────────────────

    @staticmethod
    def _is_yizi_ban(row: pd.Series) -> bool:
        """一字板: open == high == low == close."""
        return (
            abs(row["open"] - row["high"]) < 0.001
            and abs(row["high"] - row["low"]) < 0.001
            and abs(row["low"] - row["close"]) < 0.001
        )

    @staticmethod
    def _is_limit_up(symbol: str, price: float, prev_close: float) -> bool:
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (price / prev_close - 1) >= threshold - 0.001

    @staticmethod
    def _is_limit_down(symbol: str, price: float, prev_close: float) -> bool:
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (price / prev_close - 1) <= -threshold + 0.001

    def _can_buy(self, symbol: str, date: pd.Timestamp, timing: str) -> bool:
        """Check if buying is legal (suspension, 涨停, 一字板)."""
        ohlcv = self._ohlcv_dict.get(symbol)
        if ohlcv is None or date not in ohlcv.index:
            return False
        row = ohlcv.loc[date]

        # Suspension (NaN or 0)
        vol = row.get("volume", row.get("vol", 0))
        if not (vol > 0):
            return False

        prev_close = self._get_prev_close(symbol, date)
        if prev_close is None:
            return True  # first trading day — allow

        # 一字板 at limit up
        if self._is_yizi_ban(row) and self._is_limit_up(symbol, row["open"], prev_close):
            return False

        # Limit up at execution price
        price = row["open"] if timing == "open" else row["close"]
        if self._is_limit_up(symbol, price, prev_close):
            return False

        return True

    def _try_buy_with_fallback(
        self,
        slot_sym: str | None,
        date: pd.Timestamp,
        timing: str,
        target_value: float,
        ref_nav: float,
        bought_syms: set[str],
        info: dict,
    ) -> str | None:
        """Try to buy slot_sym; on failure walk down L2 ranking for a substitute.

        Args:
            slot_sym: Primary symbol assigned to this slot (may be None).
            date: Current trading day.
            timing: "open" or "close".
            target_value: Target position value in cash.
            ref_nav: Reference NAV for allocation.
            bought_syms: Symbols already bought this step (mutated on success).
            info: Step info dict (mutated — appends to blocked/substituted).

        Returns:
            Symbol actually bought, or None if nothing was buyable.
        """
        if slot_sym is None:
            return None

        l2_ranking = self._daily_signals[self._day_idx].get("l2_ranking", [])
        held_set = set(self._holdings.keys())

        # Build candidate list: primary first, then L2 ranking fallbacks
        candidates = [slot_sym]
        for sym in l2_ranking:
            if sym != slot_sym and sym not in held_set and sym not in bought_syms:
                candidates.append(sym)

        primary_blocked = False
        for sym in candidates:
            if sym in bought_syms:
                continue

            buy_price = self._get_price(sym, date, timing)
            if buy_price is None or buy_price <= 0:
                if sym == slot_sym:
                    primary_blocked = True
                    info["blocked_trades"].append(
                        {"symbol": sym, "action": "buy", "reason": "no_price"}
                    )
                continue

            h = self._holdings.get(sym)
            current_shares = h["shares"] if h else 0
            current_value = current_shares * buy_price

            if target_value <= current_value:
                # Already at or above target — counts as "bought" for slot
                bought_syms.add(sym)
                return sym

            if not self._can_buy(sym, date, timing):
                if sym == slot_sym:
                    primary_blocked = True
                    info["blocked_trades"].append(
                        {"symbol": sym, "action": "buy", "reason": "constraint"}
                    )
                continue

            # Compute shares
            delta_value = target_value - current_value
            shares_to_buy = int(delta_value / buy_price)
            shares_to_buy = (shares_to_buy // self.cfg.lot_size) * self.cfg.lot_size
            if shares_to_buy <= 0:
                continue

            # Afford check
            cost = shares_to_buy * buy_price
            commission = cost * self.cfg.buy_commission
            total_cost = cost + commission
            if total_cost > self._cash:
                affordable = int(
                    self._cash / (buy_price * (1 + self.cfg.buy_commission))
                )
                shares_to_buy = (affordable // self.cfg.lot_size) * self.cfg.lot_size
                if shares_to_buy <= 0:
                    # Cash exhausted — no point trying cheaper stocks
                    return None
                cost = shares_to_buy * buy_price
                commission = cost * self.cfg.buy_commission
                total_cost = cost + commission

            # Execute buy
            self._cash -= total_cost

            self.trade_log.append({
                "date": date, "symbol": sym, "action": "buy",
                "price": buy_price, "shares": shares_to_buy,
                "commission": commission, "pnl": 0.0,
                "hold_days": 0, "timing": timing,
                "prev_close": self._get_prev_close(sym, date),
            })

            if h:
                old_cost = h["shares"] * h["entry_price"]
                new_cost = shares_to_buy * buy_price
                total_shares = h["shares"] + shares_to_buy
                h["shares"] = total_shares
                h["entry_price"] = (old_cost + new_cost) / total_shares
            else:
                self._holdings[sym] = {
                    "shares": shares_to_buy,
                    "entry_price": buy_price,
                    "hold_days": 0,
                }

            bought_syms.add(sym)

            # Record substitution if we fell back to a different stock
            if sym != slot_sym and primary_blocked:
                info["substituted_trades"].append(
                    {"slot_symbol": slot_sym, "bought_symbol": sym}
                )

            return sym

        return None

    def _can_sell(self, symbol: str, date: pd.Timestamp, timing: str) -> bool:
        """Check if selling is legal (T+1, suspension, 跌停, 一字板)."""
        h = self._holdings.get(symbol)
        if h and h["hold_days"] < 1:
            return False  # T+1 rule

        ohlcv = self._ohlcv_dict.get(symbol)
        if ohlcv is None or date not in ohlcv.index:
            return False
        row = ohlcv.loc[date]

        # Suspension (NaN or 0)
        vol = row.get("volume", row.get("vol", 0))
        if not (vol > 0):
            return False

        # 一字板 — can't sell regardless of direction
        if self._is_yizi_ban(row):
            return False

        prev_close = self._get_prev_close(symbol, date)
        if prev_close is None:
            return True

        # Limit down at execution price
        price = row["open"] if timing == "open" else row["close"]
        if self._is_limit_down(symbol, price, prev_close):
            return False

        return True

"""
Layer 4: RL Portfolio Trader (MaskablePPO wrapper)

Thin wrapper around sb3-contrib MaskablePPO for training, saving, and
loading portfolio agents.  Depends on ``sb3-contrib`` and ``gymnasium``.
"""

from __future__ import annotations

import time

from screener.config import ScreenerConfig


def _mask_fn(env):
    """Extract action masks from the underlying PortfolioEnv."""
    return env.action_masks()


class RLTrader:
    """Train and run a MaskablePPO agent for portfolio management."""

    def __init__(self, cfg: ScreenerConfig):
        self.cfg = cfg

    def train(self, env):
        """Train a MaskablePPO model on the given PortfolioEnv.

        Wraps the env with ActionMasker so illegal actions are masked
        during rollout collection.  Returns the trained model.
        """
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback

        total_steps = self.cfg.rl_total_timesteps

        class _ProgressCallback(BaseCallback):
            """Print a concise one-liner every 10k steps."""

            def __init__(self):
                super().__init__()
                self._t0 = time.time()
                self._next_log = 10_000

            def _on_step(self) -> bool:
                if self.num_timesteps >= self._next_log:
                    elapsed = time.time() - self._t0
                    fps = self.num_timesteps / max(elapsed, 1)
                    loss = self.model.logger.name_to_value.get(
                        "train/policy_gradient_loss", float("nan")
                    )
                    print(
                        f"    PPO {self.num_timesteps:>6}/{total_steps} "
                        f"({elapsed:.0f}s, {fps:.0f}fps) "
                        f"pg_loss={loss:.4f}"
                    )
                    self._next_log += 10_000
                return True

        masked_env = ActionMasker(env, _mask_fn)

        t0 = time.time()
        model = MaskablePPO(
            "MlpPolicy",
            masked_env,
            learning_rate=self.cfg.rl_learning_rate,
            n_steps=self.cfg.rl_n_steps,
            batch_size=self.cfg.rl_batch_size,
            n_epochs=self.cfg.rl_n_epochs,
            gamma=self.cfg.rl_gamma,
            clip_range=self.cfg.rl_clip_range,
            ent_coef=self.cfg.rl_ent_coef,
            policy_kwargs=dict(net_arch=self.cfg.rl_net_arch),
            verbose=0,
        )
        model.learn(
            total_timesteps=total_steps,
            callback=_ProgressCallback(),
        )
        elapsed = time.time() - t0
        print(f"    PPO training complete: {total_steps} steps in {elapsed:.0f}s")
        return model

    @staticmethod
    def save(model, path: str):
        """Save a trained MaskablePPO model to disk."""
        model.save(path)

    @staticmethod
    def load(path: str):
        """Load a MaskablePPO model from disk."""
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path)

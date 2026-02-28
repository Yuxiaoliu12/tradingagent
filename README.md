# tradingagent

A-share stock screening and RL trading agent.

**Layers:**
1. Factor Timing (XGBoost) — macro regime → factor weight prediction
2. Technical Ranker (XGBRanker) — cross-sectional stock ranking
3. Kronos Prediction (optional) — probabilistic price forecasting
4. RL Portfolio Agent (MaskablePPO) — discrete portfolio management with A-share constraints

## Install

```bash
pip install -e .            # core only
pip install -e ".[rl]"      # + RL agent
pip install -e ".[all]"     # everything
```

On Colab:

```bash
pip install -r requirements.txt
```

## Colab Quickstart

See `docs/COLAB_GUIDE.md` and `notebooks/train_colab.ipynb`.

1. Upload `data/ohlcv_all_a.pkl` and `data/benchmark_000905.pkl` to Google Drive
2. Open `notebooks/train_colab.ipynb` in Colab with T4 GPU runtime
3. Run all cells

## Provenance

Extracted from [facecat-kronos](https://github.com/Yuxiaoliu12/facecat-kronos). The `screener/kronos/` subdirectory is vendored from the original `facecat/model/` module.

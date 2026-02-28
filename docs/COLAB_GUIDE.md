# Colab Setup Guide — Multi-Layer Stock Screener

## Prerequisites

- Google Colab Pro recommended (T4 or A100 GPU for Kronos inference)
- Google Drive with ~5 GB free space (OHLCV data + model cache + results)

---

## 1. Initial Setup (first time only)

### 1a. Get the repo

The notebooks clone from GitHub automatically. No need to upload the repo to Drive.

```python
!git clone https://github.com/Yuxiaoliu12/tradingagent.git /content/tradingagent
!pip install -r /content/tradingagent/requirements.txt -q
```

### 1c. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1d. Download OHLCV data via baostock

Use `notebooks/download_colab.ipynb` to download all A-share OHLCV data.
The data persists on Drive across sessions.

---

## 2. Google Drive Storage Layout

After running everything once, your Drive will look like:

```
MyDrive/kronos/
├── data/
│   ├── ohlcv_all_a.pkl       # ~2 GB, all A-share OHLCV (baostock)
│   └── benchmark_000905.pkl   # CSI500 benchmark index
└── output/screener/
    ├── cache/                 # Reusable feature caches
    └── runs/YYYYMMDD_HHMMSS/  # Per-session results
        ├── backtest_results.pkl
        ├── config.json
        └── models/
```

---

## 3. Running the Notebook Cell-by-Cell

Open `notebooks/train_colab.ipynb` in Colab (or copy the cells manually).
Make sure the runtime is set to **GPU** (Runtime → Change runtime type → T4 GPU).

### Cell 0 — Install & Mount

Installs packages and mounts Drive. Run once per session.

### Cell 2 — Imports & Config

Adds the repo to `sys.path` and loads config. **You must edit the path**
if your repo is somewhere other than `/content/tradingagent`:

```python
sys.path.insert(0, '/content/tradingagent')  # ← adjust this
```

If you have fine-tuned Kronos weights, set the paths here:

```python
cfg.kronos_tokenizer_path = '/content/drive/MyDrive/kronos/models/kronos_tokenizer'
cfg.kronos_predictor_path = '/content/drive/MyDrive/kronos/models/kronos_predictor'
```

### Cell 3 — Compute Alpha158 Factors

Computes 158 cross-sectional factors for all stocks in the universe. First run
takes 3-5 minutes; results are cached to Drive as `alpha158_cache.pkl`.
Subsequent runs load from cache in seconds.

Expected output:

```
Alpha158 shape: (~2500000, 158)   # rows = dates × stocks
Labels shape:   (~2500000,)
Regime shape:   (~2500, 40-60)    # rows = trading days
```

### Cell 4 — Train Layer 1 (Factor Timing)

Trains the XGBoost multi-output regressor. Takes ~30 seconds on CPU.
Model is saved to Drive. The validation output shows predicted-vs-actual IC
correlations per factor category — positive values are good.

### Cell 5 — Train Layer 2 (Technical Ranker)

Loads OHLCV, computes technical features, trains XGBRanker. Takes 1-3 minutes.
The validation output shows mean Spearman rank correlation — values above 0.05
indicate the ranker is learning something useful.

Note: this cell loads OHLCV for 500 symbols as a speed tradeoff. To use the
full universe, change `all_symbols[:500]` to `all_symbols`.

### Cell 6 — Load Kronos to GPU

Loads the Kronos tokenizer + predictor onto GPU. If you don't have fine-tuned
weights yet, this cell will print an error — that's fine, the rest of the
pipeline still works without Layer 3.

### Cell 7 — Single-Day Inference Demo

Runs the full screening pipeline for one date. Useful for sanity checking:

- Layer 1 should output ~200 stocks
- Layer 2 should output ~30 stocks
- Layer 3 (if loaded) should output ~5 stocks with predicted returns

### Cell 8 — Paper Trading Dashboard

Shows current paper trading metrics. Only useful after running the backtest
(Cell 9) or a live trading loop.

### Cell 9 — Full Walk-Forward Backtest

Runs the complete backtest with quarterly retraining. This is the slowest cell:

| Setting | Time (estimate) |
|---------|----------------|
| `run_kronos=False` | 10-30 min (CPU only, Layers 1+2) |
| `run_kronos=True` on T4 | 2-4 hours |
| `run_kronos=True` on A100 | 30-60 min |

Start with `run_kronos=False` to verify the pipeline works, then enable Kronos.

### Cell 10 — Visualisation

Generates two charts saved to Drive:

- **NAV curve + drawdown** — overall portfolio performance
- **Layer attribution** — bar chart comparing average 5-day forward return at
  each layer's cutoff vs the universe baseline

---

## 4. Resuming After a Runtime Reset

Colab runtimes reset after ~12 hours (or on disconnect). To resume:

1. Run Cell 0 (install + mount) — packages need reinstalling each session
2. Run Cell 2 (imports + config)
3. **Skip training cells** if models exist on Drive — load them instead:

```python
layer1 = FactorTimingModel(cfg)
layer1.load()  # loads from Drive

layer2 = TechnicalRanker(cfg)
layer2.load()  # loads from Drive
```

4. Continue from wherever you left off

---

## 5. Config Overrides

You can override any config value after creating the `ScreenerConfig` object:

```python
cfg = ScreenerConfig()

# Use a smaller universe for faster iteration
cfg.universe = 'csi300'
cfg.benchmark = 'SH000300'

# Reduce Kronos samples for speed (less accurate confidence)
cfg.kronos_sample_count = 3

# Tighter backtest window for quick testing
cfg.backtest_start = '2024-01-01'
cfg.backtest_end = '2024-06-30'

# Layer 1 passes more stocks through (wider funnel)
cfg.layer1_top_n = 300
```

---

## 6. Troubleshooting

### "CUDA out of memory" during Kronos inference

Kronos runs 30 stocks sequentially (not batched), so memory usage is modest.
If you still hit OOM:

- Reduce `cfg.kronos_sample_count` from 5 to 3
- Reduce `cfg.layer2_top_n` from 30 to 15 (fewer stocks sent to Kronos)
- Use `layer3.unload_model()` after Layer 3 to free GPU memory

### "Module not found: screener"

Make sure `sys.path` points to the repo root (one level above `screener/`):

```python
import sys
sys.path.insert(0, '/content/tradingagent')  # contains screener/ directory
```

### Alpha158 computation is very slow

The first run computes factors for ~1000 stocks × ~10 years. This is normal
(3-5 min). The result is cached to Drive — subsequent runs load in seconds.

If it takes longer than 10 minutes, reduce the time range:

```python
cfg.train_start = '2018-01-01'  # shorter training window
```

### News features fail (AKShare errors)

AKShare scrapes East Money and can break when the website changes. The
`NewsScorer` gracefully degrades — if news fetching fails, all news features
default to 0 and the ranker works on technical indicators alone.

To disable news entirely:

```python
layer2_picks = layer2.select_top(ohlcv, layer1_picks, date, include_news=False)
```

### XGBoost training crashes with "too many features"

Alpha158 produces 158 features × ~1000 stocks per day. If memory is tight:

- Reduce the training date range
- Subsample dates more aggressively (change `cal[::5]` to `cal[::10]`)

---

## 7. Running Live Daily Screening (Paper Trading)

Once models are trained, you can run the screener daily with a simple loop:

```python
import pandas as pd

today = pd.Timestamp.now().normalize()

# Layer 1
picks_200 = layer1.select_top(today, alpha158_df=alpha158)

# Layer 2
picks_30 = layer2.select_top(ohlcv, picks_200, today, include_news=True)

# Layer 3 (Kronos)
layer3.load_model()
scores = layer3.screen_stocks(ohlcv, picks_30, today)
top_5 = list(scores.head(5).index)
layer3.unload_model()

print(f'Top picks for {today.date()}: {top_5}')
print(scores.head(5))
```

For actual paper trading with position tracking, use the `PaperTrader` class
with `daily_update()` — see Cell 9 in the notebook for the pattern.

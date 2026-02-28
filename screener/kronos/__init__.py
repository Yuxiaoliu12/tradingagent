"""Vendored Kronos model — requires torch, einops, huggingface_hub."""

try:
    from screener.kronos.kronos import (
        KronosTokenizer,
        Kronos,
        KronosPredictor,
        calc_time_stamps,
        auto_regressive_inference,
    )
except Exception:
    pass

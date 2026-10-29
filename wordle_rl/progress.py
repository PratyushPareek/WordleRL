"""Plain-text progress lines for log files: step/total, elapsed, ETA, IST timestamps."""

import time
from datetime import datetime, timedelta, timezone

from transformers import TrainerCallback

IST = timezone(timedelta(hours=5, minutes=30))
KEYS = ("reward", "rewards/solved/mean", "rewards/invalid/mean", "num_tools_called", "completions/mean_length")


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def fmt_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start = time.time()
        self.eval_start = None
        self.eval_batches = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # One line per evaluation batch, so a long baseline evaluation shows progress and an ETA.
        if self.eval_start is None:
            self.eval_start, self.eval_batches = time.time(), 0
        self.eval_batches += 1
        total = len(eval_dataloader) if eval_dataloader is not None else 0
        elapsed = time.time() - self.eval_start
        eta = elapsed / self.eval_batches * (total - self.eval_batches) if total else float("nan")
        print(f"[{now_ist()}] eval batch {self.eval_batches}/{total or '?'} elapsed {fmt_seconds(elapsed)} "
              f"ETA {fmt_seconds(eta) if total else '?'}", flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        print(f"[{now_ist()}] training started, {state.max_steps} steps", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step, total = state.global_step, state.max_steps
        elapsed = time.time() - self.start
        eta = elapsed / step * (total - step) if step else float("nan")
        shown = " ".join(f"{k}={logs[k]:.3f}" for k in KEYS if isinstance(logs.get(k), (int, float)))
        evals = " ".join(f"{k}={v:.3f}" for k, v in logs.items() if k.startswith("eval_") and isinstance(v, float))
        print(
            f"[{now_ist()}] step {step}/{total} elapsed {fmt_seconds(elapsed)} "
            f"ETA {fmt_seconds(eta) if step else '?'} {shown} {evals}".rstrip(),
            flush=True,
        )

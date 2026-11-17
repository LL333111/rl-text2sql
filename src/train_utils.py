"""Pieces shared by the SFT and GRPO trainers: config loading, the GPU-time
budget, checkpoint timing and the "did the LoRA weights change" check."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load_config(path: str | Path) -> dict:
    """Load a YAML config; `inherit: other.yaml` merges the parent underneath."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        cfg = base
    return cfg


def load_budget() -> dict:
    """Rule 3: both arms read the GPU budget from configs/budget.yaml."""
    p = CONFIGS / "budget.yaml"
    if not p.exists():
        raise FileNotFoundError(
            "configs/budget.yaml does not exist yet. It is written after the Stage 1 "
            "timing test; use --max-steps for smoke and timing runs.")
    return load_config(p)


def make_lora_config():
    from peft import LoraConfig
    return LoraConfig(**load_config(CONFIGS / "lora.yaml")["lora"])


def lora_b_norm(model) -> float:
    """L2 norm over all lora_B matrices. They are initialized to zero, so a
    non-zero value after training shows the adapter was actually updated."""
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            total += float(p.detach().float().pow(2).sum())
    return total ** 0.5


def make_budget_callback(out_dir: Path, budget_hours: float | None):
    """Stop training once `budget_hours` of training wall-clock time is used,
    and record elapsed time at every log and checkpoint so that val score can
    later be plotted against GPU time."""
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.log_path = out_dir / "time_log.jsonl"

        def _elapsed(self):
            return time.time() - self.t0

        def _write(self, kind, state, extra=None):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": kind, "step": state.global_step,
                                    "elapsed_s": round(self._elapsed(), 1), **(extra or {})}) + "\n")

        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            if budget_hours is not None and self._elapsed() >= budget_hours * 3600:
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kw):
            self._write("log", state, {k: v for k, v in (logs or {}).items()
                                       if isinstance(v, (int, float))})

        def on_save(self, args, state, control, **kw):
            self._write("save", state)

        def on_train_end(self, args, state, control, **kw):
            self._write("end", state)

    return BudgetCallback()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "vllm"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out

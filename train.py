"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import contextlib
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from mnemebrain_hooks import RunConfig, RunResults, create_hooks
from models import create_model
from models.nanochat import GPT, build_model_config
from platform_config import PLATFORM
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

MODEL_NAME = os.environ.get("AUTORESEARCH_MODEL", "nanochat")

# ---------------------------------------------------------------------------
# Platform config loading
# ---------------------------------------------------------------------------

def load_platform_config():
    """Load platform-specific training config from TOML file."""
    config_path = Path(__file__).parent / "configs" / f"{PLATFORM.kind}.toml"
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise SystemExit(
            f"[autoresearch] Config file not found: {config_path}\n"
            f"Run 'git checkout configs/' to restore it."
        ) from e

_cfg = load_platform_config()

# ---------------------------------------------------------------------------
# Hyperparameters (loaded from configs/{platform}.toml, agent can override)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = _cfg["model"]["aspect_ratio"]
HEAD_DIM = _cfg["model"]["head_dim"]
WINDOW_PATTERN = _cfg["model"]["window_pattern"]

# Optimization
TOTAL_BATCH_SIZE = _cfg["optimization"]["total_batch_size"]
EMBEDDING_LR = _cfg["optimization"]["embedding_lr"] * 0.2  # 0.2x: fine-tuning between 0.25x (2.766) and 0.125x (2.784)
UNEMBEDDING_LR = _cfg["optimization"]["unembedding_lr"] * 0.2
MATRIX_LR = _cfg["optimization"]["matrix_lr"] * 0.2
SCALAR_LR = _cfg["optimization"]["scalar_lr"] * 0.2
WEIGHT_DECAY = _cfg["optimization"]["weight_decay"]
ADAM_BETAS = _cfg["optimization"]["adam_betas"]
WARMUP_RATIO = _cfg["optimization"]["warmup_ratio"]
WARMDOWN_RATIO = _cfg["optimization"]["warmdown_ratio"]
FINAL_LR_FRAC = _cfg["optimization"]["final_lr_frac"]

# Model size (hardware-dependent, not in TOML)
DEPTH = int(os.environ.get("AUTORESEARCH_DEPTH", PLATFORM.recommended_depth))
DEVICE_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_DEVICE_BATCH", _cfg["hardware"]["device_batch_size"]))

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("AUTORESEARCH_SEED", 42))

t_start = time.time()
torch.manual_seed(SEED)
if PLATFORM.kind == "cuda":
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
device = PLATFORM.device
autocast_dtype = torch.bfloat16 if PLATFORM.supports_bf16 else (torch.float16 if PLATFORM.supports_fp16 else torch.float32)
if PLATFORM.kind == "cpu":
    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
elif PLATFORM.kind == "mps":
    autocast_ctx = contextlib.nullcontext()
else:
    autocast_ctx = torch.amp.autocast(device_type=PLATFORM.kind, dtype=autocast_dtype)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

if MODEL_NAME == "nanochat":
    config = build_model_config(
        vocab_size=vocab_size,
        depth=DEPTH,
        aspect_ratio=ASPECT_RATIO,
        head_dim=HEAD_DIM,
        window_pattern=WINDOW_PATTERN,
        sequence_len=MAX_SEQ_LEN,
    )
    print(f"Model config: {asdict(config)}")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
else:
    model, config = create_model(MODEL_NAME, vocab_size=vocab_size, n_layer=DEPTH, sequence_len=MAX_SEQ_LEN)
    print(f"Model ({MODEL_NAME}) config: {config}")
    model.to(device)
    model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

if MODEL_NAME == "nanochat":
    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
else:
    optimizer = model.setup_optimizer()
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

if PLATFORM.supports_compile:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", device=device)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# --- MnemeBrain hooks: pre-run ---
hooks = create_hooks(seed=SEED)
run_config = RunConfig(
    matrix_lr=MATRIX_LR, embedding_lr=EMBEDDING_LR,
    unembedding_lr=UNEMBEDDING_LR, scalar_lr=SCALAR_LR,
    wd=WEIGHT_DECAY, depth=DEPTH,
    total_batch_size=TOTAL_BATCH_SIZE, device_batch_size=DEVICE_BATCH_SIZE,
    warmup_ratio=WARMUP_RATIO, warmdown_ratio=WARMDOWN_RATIO, seed=SEED,
)
run_context = hooks.pre_run(run_config)
hooks.start_log_capture(run_config)

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
grad_norm_max = 0.0
loss_history: list[float] = []  # smoothed loss at each step, for trend computation

while True:
    if PLATFORM.kind == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get('kind') == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    # Track gradient norm for telemetry (sample every 50 steps to avoid overhead)
    if step % 50 == 0:
        _gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        if _gnorm > grad_norm_max:
            grad_norm_max = _gnorm

    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("\nFAIL — loss diverged")
        hooks.stop_log_capture()
        hooks.post_run(
            config=run_config,
            results=RunResults(
                val_bpb=999999.0, steps=step, peak_vram_mb=0.0,
                final_loss=train_loss_f if not math.isnan(train_loss_f) else 999999.0,
                mfu=0.0, diverged=True, loss_trend="diverging", grad_norm_max=grad_norm_max,
            ),
        )
        exit(1)

    if PLATFORM.kind == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    loss_history.append(debiased_smooth_loss)
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if PLATFORM.kind == "cuda" else 0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# --- MnemeBrain hooks: post-run ---
hooks.stop_log_capture()
# Derive loss trend from final 20% of training
_diverged = math.isnan(debiased_smooth_loss) or debiased_smooth_loss > 100
if _diverged:
    _loss_trend = "diverging"
elif len(loss_history) > 20:
    # Compare mean of final 20% to mean of preceding 20%
    n = len(loss_history)
    tail_start = int(n * 0.8)
    mid_start = int(n * 0.6)
    tail_mean = sum(loss_history[tail_start:]) / (n - tail_start)
    mid_mean = sum(loss_history[mid_start:tail_start]) / max(tail_start - mid_start, 1)
    delta = tail_mean - mid_mean
    if delta < -0.01:
        _loss_trend = "improving"
    elif delta > 0.01:
        _loss_trend = "diverging"
    else:
        _loss_trend = "flat"
else:
    _loss_trend = "flat"

hooks.post_run(
    config=run_config,
    results=RunResults(
        val_bpb=val_bpb, steps=step, peak_vram_mb=peak_vram_mb,
        final_loss=debiased_smooth_loss, mfu=steady_state_mfu,
        diverged=_diverged, loss_trend=_loss_trend, grad_norm_max=grad_norm_max,
    ),
)

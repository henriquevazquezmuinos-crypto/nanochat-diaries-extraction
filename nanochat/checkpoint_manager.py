"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def make_checkpoint_schedule(
    num_iterations,
    num_checkpoints=0,
    spacing="log",
    min_frac=0.01,
    steps=None,
    fracs=None,
    every=-1,
    include_init=False,
    include_last=True,
    multiple_of=1,
):
    """
    Decide at which training steps to save checkpoints. Returns a sorted list of unique ints.
    Sources are combined (union), so any mix of them works:

    - num_checkpoints + spacing: N checkpoints spread over [min_frac*T, T] where T = num_iterations
        spacing="log"    : geometric spacing (dense early when the model changes fastest). Default.
        spacing="linear" : evenly spaced.
        spacing="sqrt"   : in between linear and log.
    - steps: explicit absolute step numbers, e.g. [100, 1000, 5000]
    - fracs: explicit fractions of training in (0, 1], e.g. [0.1, 0.5, 0.9]
    - every: a checkpoint every N steps (-1 = disable)
    - include_init: also save step 0 (random init baseline)
    - include_last: always include the final step (num_iterations)
    - multiple_of: snap steps to a multiple of this (e.g. eval_every), never below 1 or above T

    Steps outside [0, T] are dropped. Step 0 only appears if include_init=True.
    """
    T = num_iterations
    assert T > 0, "num_iterations must be positive"
    assert spacing in ("log", "linear", "sqrt"), f"unknown spacing {spacing!r}"
    assert 0 < min_frac <= 1, "min_frac must be in (0, 1]"
    out = set()

    n = num_checkpoints
    if n > 0:
        lo = min_frac * T
        if n == 1:
            points = [float(T)]
        elif spacing == "linear":
            # evenly spaced and ending at T; min_frac is ignored here (start at T/n)
            points = [T * (i + 1) / n for i in range(n)]
        elif spacing == "log":
            ratio = (T / lo) ** (1.0 / (n - 1))
            points = [lo * ratio**i for i in range(n)]
        else: # sqrt
            a, b = lo**0.5, T**0.5
            points = [(a + (b - a) * i / (n - 1)) ** 2 for i in range(n)]
        out.update(round(p) for p in points)

    if steps:
        out.update(int(s) for s in steps)
    if fracs:
        assert all(0 < f <= 1 for f in fracs), "fracs must be in (0, 1]"
        out.update(round(f * T) for f in fracs)
    if every and every > 0:
        out.update(range(every, T + 1, every))

    if multiple_of > 1:
        out = {max(multiple_of, round(s / multiple_of) * multiple_of) for s in out if s > 0}
        out = {min(s, T) for s in out}

    if include_last:
        out.add(T)
    if include_init:
        out.add(0)
    return sorted(s for s in out if (s >= 1 or (s == 0 and include_init)) and s <= T)


def make_checkpoint_fractions(num_checkpoints=0, spacing="linear", min_frac=0.01, fracs=None, include_last=True):
    """
    Like make_checkpoint_schedule, but for stages whose total length isn't known in advance
    (e.g. SFT with dataset-driven stopping). Returns sorted fractions of training progress in (0, 1].
    The caller saves when its progress counter first crosses each fraction.
    """
    N = 1_000_000 # fine-grained virtual horizon, fractions are step/N
    steps = make_checkpoint_schedule(N, num_checkpoints, spacing=spacing, min_frac=min_frac,
                                     fracs=fracs, include_last=include_last)
    return [s / N for s in steps]


def split_checkpoint_budget(total, weights=None):
    """
    Split a total checkpoint budget across training stages so that the sum is exactly `total`
    and every stage gets at least 1 (its final model). Returns a dict stage -> count.
    Default weights favor pretraining (where the curve is richest), then SFT, then RL.
    E.g. total=8 -> {base: 5, sft: 2, rl: 1}; total=6 -> {base: 4, sft: 1, rl: 1}.
    """
    weights = weights or {"base": 0.6, "sft": 0.25, "rl": 0.15}
    assert total >= len(weights), f"need at least {len(weights)} checkpoints (one per stage)"
    wsum = sum(weights.values())
    ideal = {k: total * w / wsum for k, w in weights.items()}
    counts = {k: max(1, int(v)) for k, v in ideal.items()}
    while sum(counts.values()) < total: # give leftovers to the stage with the largest fractional remainder
        k = max(counts, key=lambda k: ideal[k] - counts[k])
        counts[k] += 1
    while sum(counts.values()) > total: # take back from the stage that is most over its ideal share
        k = max((k for k in counts if counts[k] > 1), key=lambda k: counts[k] - ideal[k])
        counts[k] -= 1
    return counts


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data

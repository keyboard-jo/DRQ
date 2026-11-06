"""Ablation-arm presets and run-path stamping for the RQ-VAE pipeline.

Deliberately pure Python: importing this module must not pull in torch, numpy or the
dataset stack, so the grid wiring is testable on a machine with no training environment.

Vocabulary (see GLOSSARY.md):
  mechanism  a toggleable change to the training recipe
  arm        one configuration of mechanisms, i.e. one preset
  A0         the ablation-zero arm: every claimed mechanism off

Arms are sequential, not factorial. Every arm differs from the one before it by exactly
one mechanism, so an effect is attributable without a 2^D grid.
"""

import math
import os

# The switches a preset is allowed to set. Anything else is a reference-recipe setting
# (data, optimiser, schedule, epochs) and must stay identical across arms.
SWITCH_KEYS = (
    "run_baseline",
    "bottleneck_norm",       # "on" | "off"
    "jitter",
    "entropy_loss_weight",
    "codebook_lr_mult",
)

# Sequential ladder. Each entry differs from its predecessor by one mechanism.
PRESETS = {
    # A0: every claimed mechanism off. This is the reference point for all comparisons.
    "a0": {
        "run_baseline": True,
        "bottleneck_norm": "off",
        "jitter": 0.0,
        "entropy_loss_weight": 0.0,
        "codebook_lr_mult": 1.0,
    },
    # + bottleneck normalisation
    "norm": {
        "run_baseline": False,
        "bottleneck_norm": "on",
        "jitter": 0.0,
        "entropy_loss_weight": 0.0,
        "codebook_lr_mult": 1.0,
    },
    # + codebook learning-rate multiplier
    "norm_cb": {
        "run_baseline": False,
        "bottleneck_norm": "on",
        "jitter": 0.0,
        "entropy_loss_weight": 0.0,
        "codebook_lr_mult": 2.0,
    },
    # + entropy-deficit regulariser
    "norm_cb_entropy": {
        "run_baseline": False,
        "bottleneck_norm": "on",
        "jitter": 0.0,
        "entropy_loss_weight": 0.002,
        "codebook_lr_mult": 2.0,
    },
    # + latent jitter (the full stack)
    "norm_cb_entropy_jitter": {
        "run_baseline": False,
        "bottleneck_norm": "on",
        "jitter": 0.001,
        "entropy_loss_weight": 0.002,
        "codebook_lr_mult": 2.0,
    },
}

# Arm order, for printing the ladder in help text and logs.
PRESET_ORDER = ("a0", "norm", "norm_cb", "norm_cb_entropy", "norm_cb_entropy_jitter")


def preset_help() -> str:
    """One-line ladder description for argparse help."""
    return " -> ".join(PRESET_ORDER)


def resolve_preset(name):
    """Return a copy of an arm's switch dict, or {} when no preset was requested."""
    if not name:
        return {}
    try:
        return dict(PRESETS[name])
    except KeyError:
        raise SystemExit(
            f"Unknown --preset '{name}'. Choose one of: {', '.join(PRESET_ORDER)}"
        )


def merge_explicit(preset_switches: dict, explicit: dict) -> dict:
    """Overlay flags the user passed explicitly on top of a preset.

    `explicit` must contain only keys the user actually set on the command line (i.e. those
    whose value differs from the argparse default). An explicit flag always wins, so a group
    member can start from an arm and nudge one hyperparameter without leaving the ladder.
    """
    resolved = dict(preset_switches)
    for key, value in explicit.items():
        if key in SWITCH_KEYS:
            resolved[key] = value
    return resolved


def describe_arm(switches: dict) -> str:
    """Name the preset that matches these switches exactly, else "custom".

    Lets the log say which arm was trained even when the switches were typed by hand.
    """
    for name in PRESET_ORDER:
        preset = PRESETS[name]
        if all(switches.get(key, preset[key]) == preset[key] for key in SWITCH_KEYS):
            return name
    return "custom"


def _with_slug(path: str, slug: str) -> str:
    """`dir/name.ext` -> `dir/slug/name.ext`."""
    directory, base = os.path.split(path)
    return os.path.join(directory, slug, base)


def stamped_paths(
    run_name,
    *,
    log_dir,
    checkpoint_dir,
    output_path,
    model_path,
    default_log_dir,
    default_checkpoint_dir,
    default_output_path,
    default_model_path,
):
    """Insert ``run_name`` as a directory segment into every path still at its default.

    A path the caller set explicitly is returned untouched, so nothing can be relocated
    behind the caller's back. When ``run_name`` is falsy every path is returned as given.
    """
    paths = {
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_path": output_path,
        "model_path": model_path,
    }
    if not run_name:
        return paths

    defaults = {
        "log_dir": default_log_dir,
        "checkpoint_dir": default_checkpoint_dir,
        "output_path": default_output_path,
        "model_path": default_model_path,
    }
    for key, value in paths.items():
        # An unset directory option arrives as None; treat it as "still default".
        if value is None or value == defaults[key]:
            paths[key] = _with_slug(value if value is not None else defaults[key], run_name)
    return paths


def usage_entropy(counts) -> float:
    """Shannon entropy (nats) of hard-assignment counts, ignoring empty cells."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def utilisation_deficit(entropy: float, codebook_size: int) -> float:
    """Fraction of a codebook's entropy range left unused: 0 uniform, 1 fully concentrated."""
    max_entropy = math.log(codebook_size)
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def effective_cells(n_items: int, n_unique: int) -> float:
    """Invert the birthday formula for the number of equally likely cells.

    If items were spread uniformly over ``M`` cells, the expected number of distinct
    occupied cells would be ``M * (1 - exp(-N/M))``. Solving that for ``M`` gives a
    measure of code capacity that does not move when ``N`` changes, unlike the raw
    uniqueness ratio.

    Solving goes through ``x = N/M``: the ratio ``U/N = (1 - exp(-x)) / x`` decreases
    monotonically in ``x``, so one bisection recovers ``x`` and ``M = N / x``.

    Caveat: the inversion is coarse when ``U/N`` approaches 1 (a nearly-unique code), because
    dU/dM vanishes there. Above roughly ``U/N = 0.999`` the returned value should be read as
    "at least this many" rather than a measurement.
    """
    n_items = int(n_items)
    n_unique = int(n_unique)
    if n_items <= 0 or n_unique <= 0:
        return 0.0
    if n_unique >= n_items:
        # Everything distinct: the data cannot see a ceiling, only a lower bound.
        return float(n_unique)

    ratio = n_unique / n_items

    def ratio_for_x(x: float) -> float:
        return (1.0 - math.exp(-x)) / x

    lo, hi = 1e-12, 1.0
    while ratio_for_x(hi) > ratio and hi < 1e12:
        hi *= 2.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if ratio_for_x(mid) > ratio:
            lo = mid
        else:
            hi = mid
    x = math.sqrt(lo * hi)
    return n_items / x


def tuple_redundancy(marginal_entropies, joint_entropy: float) -> tuple:
    """Total correlation across residual stages, in nats and bits.

    Returns ``(redundancy_nats, redundancy_bits)``. A positive value means the stages
    share information: the product of the stage marginals overstates how many distinct
    code tuples the model can actually produce.
    """
    redundancy_nats = float(sum(marginal_entropies)) - float(joint_entropy)
    return redundancy_nats, redundancy_nats / math.log(2.0)


# A centroid is under-used when its EMA usage is below this fraction of the mean usage per
# code. 0.025 reproduces the old absolute threshold of 0.1 in the configuration every study
# arm used (batch 4096 over 1024 codes, so mean usage 4.0).
DEAD_CODE_THRESHOLD_FRAC = 0.025


def dead_usage_cutoff(total_usage: float, num_codes: int,
                      threshold_frac: float = DEAD_CODE_THRESHOLD_FRAC) -> float:
    """Counts below this cutoff mark a centroid as under-used.

    Deliberately relative to the mean usage per code instead of an absolute count. An
    absolute cutoff silently rescales with ``--batch_size`` and ``--codebook_size``: the
    same 0.1 means 2.5% of mean usage at 4096/1024 (1024 codes, mean 4.0) but 0.6% at
    4096/256 (mean 16.0), so the same recipe revives different numbers of centroids for
    two geometries that differ only in codebook width.
    """
    mean_usage = float(total_usage) / max(1, int(num_codes))
    return float(threshold_frac) * mean_usage

"""Deterministic, exhaustive sample-ID splits with explicit blocked holdouts."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def validate_split(split, sample_ids):
    """Require three nonempty, disjoint lists partitioning all supplied IDs."""
    expected = list(sample_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected sample IDs must be nonempty and have no duplicates")
    all_ids = []
    for key in ("train", "val", "test"):
        values = split.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"Split {key} must be a nonempty ID list")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs within {key}")
        all_ids.extend(values)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split IDs overlap between partitions")
    if set(all_ids) != set(expected):
        missing = set(expected) - set(all_ids)
        unknown = set(all_ids) - set(expected)
        raise ValueError(f"Split is not exhaustive: missing={sorted(missing)}, unknown={sorted(unknown)}")


def make_split(records, scheme="random", seed=8012, test_fraction=0.1,
               val_fraction=0.1, pressure_block=(27.5, 28.5),
               preload_block=(3300, 3500)):
    """Split stable, sorted IDs rather than relying on a directory row order.

    Random test size rounds up and validation size rounds down, both as fractions
    of the full dataset. Blocked schemes hold every inclusive block member out;
    validation has the same full-dataset fraction and is drawn only from the
    remainder. Every partition must contain at least one sample.
    """
    if scheme not in {"random", "pressure_block", "preload_block"}:
        raise ValueError(f"Unknown split scheme: {scheme}")
    if (not np.isfinite(test_fraction) or not np.isfinite(val_fraction)
            or not 0 < test_fraction < 1 or not 0 < val_fraction < 1
            or test_fraction + val_fraction >= 1):
        raise ValueError("test_fraction and val_fraction must be positive and sum to less than one")
    blocks = {"pressure_block": pressure_block, "preload_block": preload_block}
    for name, bounds in blocks.items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid {name} interval")
    values = list(records)
    ids = [record["id"] for record in values]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Sample IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in records")
    if len(ids) < 3:
        raise ValueError("At least three samples are required for nonempty train/val/test splits")
    by_id = {record["id"]: record for record in values}
    canonical_ids = sorted(ids)
    rng = np.random.default_rng(seed)
    n_validation = math.floor(len(ids) * val_fraction)
    if scheme == "random":
        shuffled = rng.permutation(canonical_ids).tolist()
        n_test = math.ceil(len(ids) * test_fraction)
        test_ids, remainder = shuffled[:n_test], shuffled[n_test:]
    else:
        key = "pressure_MPa" if scheme == "pressure_block" else "preload_N"
        lower, upper = blocks[scheme]
        condition = np.asarray([float(by_id[sample_id][key]) for sample_id in canonical_ids])
        if not np.isfinite(condition).all():
            raise ValueError(f"Non-finite {key} in split records")
        test_ids = [sample_id for sample_id, number in zip(canonical_ids, condition) if lower <= number <= upper]
        heldout = set(test_ids)
        remainder = rng.permutation([sample_id for sample_id in canonical_ids if sample_id not in heldout]).tolist()
    split = {"train": remainder[n_validation:], "val": remainder[:n_validation], "test": test_ids,
             "scheme": scheme, "seed": int(seed), "test_fraction": float(test_fraction),
             "val_fraction": float(val_fraction), "pressure_block": list(pressure_block),
             "preload_block": list(preload_block)}
    validate_split(split, canonical_ids)
    split["split_fingerprint"] = hashlib.sha256(json.dumps(split, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest()
    return split

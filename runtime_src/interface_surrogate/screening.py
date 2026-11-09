"""Predeclared numerical baseline search, using validation only."""
from copy import deepcopy
import csv
import os
from pathlib import Path

import numpy as np

from .baselines import fit_baseline
from .data import verify_prepared_dataset
from .evaluation import evaluate_run
from .experiments import file_hash, read_json, source_hash, write_json
from .settings import resolve_config
from .splits import validate_split


def baseline_candidates(seed=8012, epochs=100):
    """A small declared grid; neural comparisons and seed repetitions are later."""
    models = [("bilinear", {"kind": "bilinear"})]
    models += [(f"rbf_n{n}", {"kind": "rbf", "rbf_neighbors": n, "rbf_smoothing": 0.})
               for n in (24, 48, 96)]
    models += [(f"pod_gpr_k{k}_l{scale}", {"kind": "pod_gpr", "pod_components": k,
                "gpr_length_scale": scale, "gpr_noise": 1e-6})
               for k in (2, 4, 8) for scale in (.3, 1., 3.)]
    models += [(f"pod_mlp_k{k}", {"kind": "pod_mlp", "pod_components": k, "hidden": [128, 128]})
               for k in (2, 4, 8)]
    return [resolve_config({"name": f"screen_{name}", "seed": seed, "model": model,
                            "training": {"epochs": epochs, "batch_size": 16,
                                         "gradient_accumulation": 1, "amp": False}})
            for name, model in models]


def screen_baselines(processed_dir, split, output_dir, candidates):
    """Fit candidates on train, choose within each family by validation node RMSE.

    A failed candidate stops the screen, rather than silently excluding cases.
    The selected validation result is developmental, not unbiased final evidence.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Baseline screening output already exists; never overwrite")
    metadata = verify_prepared_dataset(processed_dir)
    validate_split(split, [r["id"] for r in metadata["samples"]])
    configs = [resolve_config(deepcopy(c)) for c in candidates]
    if (not configs or len({c['name'] for c in configs}) != len(configs)
            or len({c['seed'] for c in configs}) != 1):
        raise ValueError("Screening candidates must have unique names and one shared seed")
    if any(c['model']['kind'] not in {'bilinear', 'rbf', 'pod_gpr', 'pod_mlp'} for c in configs):
        raise ValueError("This screen is restricted to the four declared baseline families")
    protocol = {"stage": "val", "test_evaluated": False,
                "selection_metric": "node_rmse_um_mean", "seed": configs[0]['seed'],
                "dataset_fingerprint": metadata['dataset_fingerprint'],
                "prepared_metadata_sha256": metadata['prepared_metadata_sha256'],
                "source_hash": source_hash(), "split": split, "candidates": configs}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol, exclusive=True)
    rows = []
    for index, config in enumerate(configs, 1):
        print(f"baseline screen {index}/{len(configs)}: {config['name']}", flush=True)
        run = fit_baseline(processed_dir, split, config, output / "runs")
        summary = evaluate_run(run, stage="val", device="cpu")
        if summary['n_samples'] != len(split['val']):
            raise ValueError("Validation evaluation did not cover every declared sample")
        manifest = read_json(run / "manifest.json")
        row = {"name": config['name'], "kind": config['model']['kind'],
               "run_relative": os.path.relpath(run, output),
               "node_rmse_um_mean": summary['node_rmse_um_mean'],
               "node_mae_um_mean": summary['node_mae_um_mean'],
               "rmse_um_mean": summary['rmse_um_mean'], "r2_mean": summary.get('r2_mean'),
               "bias_um_mean": summary['bias_um_mean'],
               "node_max_abs_um_mean": summary['node_max_abs_um_mean'],
               "pointwise_pass_rate": summary['pointwise_pass_rate'],
               "representation_node_rmse_um_mean": summary['representation_node_rmse_um_mean'],
               "nearest_fallback_count": summary['nearest_fallback_count'],
               "checkpoint_sha256": file_hash(run / manifest['checkpoint_file'])}
        if not np.isfinite(row['node_rmse_um_mean']):
            raise ValueError("Validation selection metric is not finite")
        rows.append(row)
        write_json(output / "progress.json", {"completed": len(rows), "total": len(configs),
                   "stage": "val", "test_evaluated": False, "candidates": rows})
    selected = []
    for kind in sorted({row['kind'] for row in rows}):
        best = min((row for row in rows if row['kind'] == kind), key=lambda row: row['node_rmse_um_mean'])
        selected.append(best)
    result = {**{key: value for key, value in protocol.items() if key != 'candidates'},
              "candidates": rows, "selected": selected,
              "interpretation": "Validation-selected exploratory baselines; no claim that deep learning is necessary or superior."}
    write_json(output / "selection.json", result, exclusive=True)
    with (output / "validation_comparison.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], 'selected'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": row in selected})
    return result

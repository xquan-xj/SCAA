# SCAA for assembly sealing interface morphology inversion

This repository accompanies the manuscript *Morphology Inversion for Assembly Sealing Interfaces Using Decoder-Guided Cross-Axis Attention*. It contains the PyTorch training and inference code, the prepared finite element dataset, the fixed case split, the recorded configurations, and training curves for the main field prediction networks.

## Contents

| Path | Contents |
| --- | --- |
| `runtime_src/` | Training, evaluation and screening code for the SCAA and reference networks. |
| `surface.py` | Weighted bicubic B-spline reconstruction of the interface morphology. |
| `configs/` | Original run configurations for Dual-32, U-Net32, U-Net64 and TransUNet at seeds 2085, 4125 and 8012. |
| `splits/` | Fixed 712/89/90 case split, in the original case order. |
| `run_records/` | Recorded training curves and the run index. |
| `data/` | 891 prepared finite element cases with reference morphologies on a 224 by 224 polar grid and at 570 interface nodes. `data/metadata.json` records the engineering inputs and units. |

All seeds use the same 712 training, 89 validation and 90 test cases. The configurations and curves in `configs/` and `run_records/` cover Dual-32, U-Net32, U-Net64 and TransUNet.

## Train Dual-32

The recorded runs used Python 3.12.7 and PyTorch 2.7.1 with CUDA 12.8. Install the appropriate PyTorch build and the remaining dependencies from `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path runtime_src).Path
foreach ($seed in 2085, 4125, 8012) {
    python -m interface_surrogate train --config "configs/full_scaa_dual32_s$seed.json" --data data --split splits/fixed_split.json --runs runs --device cuda
}
```

The training command writes a checkpoint and manifest in each run directory. Replace `RUN_DIR` with one of those directories to generate predictions on the validation cases.

```
python -m interface_surrogate evaluate --run RUN_DIR --stage val --data data --device cuda
```

Freeze the run before evaluating it on the test cases.

```
python -m interface_surrogate freeze --runs RUN_DIR --output test_lock.json
python -m interface_surrogate evaluate --run RUN_DIR --stage test --data data --device cuda --lock test_lock.json
```

The evaluation command exports predicted heights in micrometers. The `surface.py` module fits a weighted bicubic B-spline, with a nominal knot spacing of 0.5 mm in the main comparison.

## License

- Source code (`runtime_src/`, `surface.py`, `configs/`): MIT, see `LICENSE`.
- Prepared dataset, case split and run records (`data/`, `splits/`, `run_records/`): CC BY 4.0, see `LICENSE-DATA`.

## Citation

If you use this code or dataset, please cite the accompanying manuscript and this repository. Machine-readable metadata is provided in `CITATION.cff`.

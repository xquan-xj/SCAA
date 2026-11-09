"""Rebuild portable morphology labels from explicitly identified raw histories.

Geometry uses mm/rad and arrays use [theta, radius]. Morphology is stored in
micrometres relative to the declared base plane. No fitted statistics are saved
by preparation: callers must fit them on their training IDs after splitting.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import __version__ as scipy_version
from scipy.interpolate import RBFInterpolator
import torch
from torch.utils.data import Dataset

FTPS_FIELDS = ("preload_N", "temperature_C", "pressure_MPa", "nominal_contact_area_mm2")
FP_FIELDS = ("preload_N", "pressure_MPa")
PARAMETER_UNITS = {"preload_N": "N", "temperature_C": "degC", "pressure_MPa": "MPa",
                   "nominal_contact_area_mm2": "mm2"}


def parameter_names(metadata):
    """The explicit input order; old FP metadata remains readable as two inputs."""
    return list(metadata.get("parameter_names", FP_FIELDS))


def validate_parameter_domain(values, normalization):
    """Reject unseen values along dimensions constant throughout training.

    Constant temperature/geometry cannot support generalization on those axes.
    This is a recorded training-support check, not a learned physical constraint.
    """
    array = _finite(values, "Parameters")
    mean = np.asarray(normalization["parameter_mean"], dtype=float)
    if array.shape[-1:] != mean.shape:
        raise ValueError("Parameter shape differs from the declared input order")
    active = normalization.get("active_parameter_indices", list(range(len(mean))))
    constant = [i for i in range(len(mean)) if i not in active]
    if constant and not np.all(np.isclose(array[..., constant], mean[constant], rtol=1e-10, atol=1e-10)):
        raise ValueError("Prediction is outside the training domain of constant parameters (temperature/geometry)")
    return array


def _portable_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or path.drive:
        raise ValueError(f"Expected a relative path: {relative!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the dataset directory: {relative!r}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_sha256(metadata) -> str:
    content = {key: value for key, value in metadata.items() if key != "prepared_metadata_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finite(values, description: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{description} contains non-finite values")
    return array


def prepare_dataset(raw_dir, output_dir, grid_size=224,
                    label_policy="history_column11", max_history_column=11,
                    base_z_mm=-2.0, parameter_names=None):
    """Build labels from exactly the eleventh history column; never fall back.

    The redundant column argument is checked to prevent silent label changes.
    FTPS tables explicitly store F/T/P/nominal S. FP-only synthetic/legacy tables
    retain an explicit two-parameter identity, without invented missing values.
    All input rows are validated before any processed files are written.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output must be empty; refusing nonempty output: {output_dir}")
    if label_policy != "history_column11":
        raise ValueError("label_policy must be 'history_column11'")
    if max_history_column != 11:
        raise ValueError("history_column11 requires max_history_column=11")
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 3:
        raise ValueError("grid_size must be an integer at least 3")
    if not np.isfinite(base_z_mm):
        raise ValueError("base_z_mm must be finite")

    xy_path = raw_dir / "XY1.TXT"
    xy = _finite(np.loadtxt(xy_path, skiprows=1, ndmin=2), "XY coordinates")
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
        raise ValueError("XY1.TXT must have at least three rows and exactly two coordinate columns")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("XY1.TXT contains duplicate nodes")
    radii = np.linalg.norm(xy, axis=1)
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    if radii.min() <= 0 or np.ptp(radii) <= 0 or np.ptp(angles) <= 0:
        raise ValueError("Coordinates must span positive radii and distinct angles")
    r_mm = np.linspace(radii.min(), radii.max(), int(grid_size))
    theta_rad = np.linspace(angles.min(), angles.max(), int(grid_size))
    theta_plane, radius_plane = np.meshgrid(theta_rad, r_mm, indexing="ij")
    grid_xy = np.column_stack(((radius_plane * np.cos(theta_plane)).ravel(),
                               (radius_plane * np.sin(theta_plane)).ravel()))

    manifest_path = raw_dir / "conditions.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "preload_N", "pressure_MPa", "history_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"conditions.csv requires columns {sorted(required)}")
        available = set(reader.fieldnames)
        if parameter_names is None:
            if {"temperature_C", "nominal_contact_area_mm2"} & available:
                if not set(FTPS_FIELDS).issubset(available):
                    raise ValueError("FTPS input requires both temperature_C and nominal_contact_area_mm2")
                parameter_names = list(FTPS_FIELDS)
            else:
                parameter_names = list(FP_FIELDS)
        if (not isinstance(parameter_names, (list, tuple)) or tuple(parameter_names) not in (FTPS_FIELDS, FP_FIELDS)
                or not set(parameter_names).issubset(available)):
            raise ValueError("parameter_names must explicitly match the FP or FTPS input columns")
        parameter_names = list(parameter_names)
        rows = list(reader)
    if not rows:
        raise ValueError("conditions.csv must contain at least one sample")

    samples, targets, seen_ids, seen_files = [], [], set(), set()
    source_hashes = {"XY1.TXT": _sha256(xy_path), "conditions.csv": _sha256(manifest_path)}
    for row_number, row in enumerate(rows, 2):
        sample_id = (row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Empty sample id at manifest row {row_number}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        try:
            parameters = _finite([float(row[name]) for name in parameter_names],
                                 f"Parameters for {sample_id}")
            if "temperature_C" in parameter_names and float(row["temperature_C"]) < -273.15:
                raise ValueError("Temperature in degC is below absolute zero")
            if "nominal_contact_area_mm2" in parameter_names and float(row["nominal_contact_area_mm2"]) <= 0:
                raise ValueError("Nominal geometric contact area must be positive")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameters for {sample_id}") from exc
        history_file = (row.get("history_file") or "").strip()
        history_path = _portable_path(raw_dir, history_file)
        if history_path in seen_files:
            raise ValueError(f"Repeated history file for sample {sample_id}")
        seen_files.add(history_path)
        history = _finite(np.loadtxt(history_path, ndmin=2), f"History for {sample_id}")
        if history.shape[0] != len(xy) or history.shape[1] < 1:
            raise ValueError(f"History row count for {sample_id} must match the {len(xy)} XY nodes")
        if history.shape[1] < 11:
            raise ValueError(f"History for {sample_id} has fewer than 11 columns; no last-column fallback")
        selected = 10
        node_target = (history[:, selected] - float(base_z_mm)) * 1000.0
        targets.append(node_target)
        portable_history = history_path.relative_to(raw_dir.resolve()).as_posix()
        source_hashes[portable_history] = _sha256(history_path)
        samples.append({"id": sample_id, **dict(zip(parameter_names, map(float, parameters))),
                        "npz": f"samples/sample_{len(samples):06d}.npz",
                        "selected_column_1based": selected + 1,
                        "history_columns": history.shape[1],
                        "history_file": portable_history})

    options = {"schema_version": 2, "grid_size": int(grid_size),
               "label_policy": label_policy, "max_history_column": 11, "history_column_1based": 11,
               "parameter_names": parameter_names,
               "base_z_mm": float(base_z_mm), "kernel": "thin_plate_spline",
               "degree": 1, "smoothing": 0.0, "axes": ["theta", "radius"],
               "morphology_unit": "um", "coordinates_unit": "mm",
               "numpy_version": np.__version__, "scipy_version": scipy_version}
    fingerprint_material = {"sources": source_hashes, "build_options": options}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_material, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    # One factorization with multiple right-hand sides shares exactly one
    # geometry/interpolation definition across all samples.
    node_targets = np.column_stack(targets)
    interpolator = RBFInterpolator(xy, node_targets, kernel="thin_plate_spline", degree=1, smoothing=0.0)
    grid_targets = _finite(interpolator(grid_xy), "Interpolated morphology")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir()
    np.save(output_dir / "nodes_xy_mm.npy", xy, allow_pickle=False)
    for index, record in enumerate(samples):
        np.savez_compressed(output_dir / record["npz"],
                            target_um=grid_targets[:, index].reshape(int(grid_size), int(grid_size)),
                            node_target_um=node_targets[:, index],
                            parameters=np.array([record[name] for name in parameter_names]))
    prepared_paths = ["nodes_xy_mm.npy", *(record["npz"] for record in samples)]
    prepared_hashes = {relative: _sha256(output_dir / relative) for relative in prepared_paths}
    metadata = {"dataset_fingerprint": fingerprint, "parameter_names": parameter_names,
                "parameter_units": {name: PARAMETER_UNITS[name] for name in parameter_names},
                "grid": {"r_mm": r_mm.tolist(),
                "theta_rad": theta_rad.tolist()}, "nodes_xy_mm": "nodes_xy_mm.npy",
                "samples": samples, "build_options": options, "source_sha256": source_hashes,
                "prepared_files_sha256": prepared_hashes}
    metadata["prepared_metadata_sha256"] = _metadata_sha256(metadata)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    return metadata


def load_metadata(processed_dir):
    """Load portable metadata and reject ambiguous sample identities/paths."""
    root = Path(processed_dir)
    with (root / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    records = metadata.get("samples", [])
    ids = [record["id"] for record in records]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Metadata must contain nonempty unique sample IDs")
    for record in records:
        _portable_path(root, record["npz"])
    _portable_path(root, metadata["nodes_xy_mm"])
    for key in ("r_mm", "theta_rad"):
        values = _finite(metadata["grid"][key], f"Grid {key}")
        if values.ndim != 1 or len(values) < 3 or np.any(np.diff(values) <= 0):
            raise ValueError(f"Grid {key} must have at least 3 strictly increasing coordinates")
    return metadata


def verify_prepared_dataset(processed_dir):
    """Check the metadata seal and every prepared array SHA256; return metadata.

    Training/evaluation entry points should call this once before reading data.
    The semantic fingerprint identifies raw content and build settings; these
    metadata seal and file hashes additionally detect later accidental edits,
    corruption, or replacement of geometry, conditions, settings, and arrays.
    """
    root = Path(processed_dir)
    metadata = load_metadata(root)
    if metadata.get("prepared_metadata_sha256") != _metadata_sha256(metadata):
        raise ValueError("Prepared metadata SHA256 integrity check failed")
    expected_paths = {metadata["nodes_xy_mm"], *(record["npz"] for record in metadata["samples"])}
    hashes = metadata.get("prepared_files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_paths:
        raise ValueError("Prepared file SHA256 hashes are missing or do not cover the dataset")
    for relative, recorded_digest in hashes.items():
        if (not isinstance(recorded_digest, str) or len(recorded_digest) != 64
                or any(char not in "0123456789abcdef" for char in recorded_digest)):
            raise ValueError(f"Invalid SHA256 hash for prepared file: {relative}")
        path = _portable_path(root, relative)
        if not path.is_file() or _sha256(path) != recorded_digest:
            raise ValueError(f"Prepared file SHA256 integrity check failed: {relative}")
    return metadata


def _select_records(metadata, sample_ids: Iterable[str]):
    ids = list(sample_ids)
    if not ids:
        raise ValueError("Sample IDs must contain at least one ID; empty selections are invalid")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in selection")
    mapping = {record["id"]: record for record in metadata["samples"]}
    unknown = set(ids) - mapping.keys()
    if unknown:
        raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
    return [mapping[sample_id] for sample_id in ids]


def _load_sample(root: Path, record, shape, parameter_count=2):
    with np.load(_portable_path(root, record["npz"]), allow_pickle=False) as sample:
        target = _finite(sample["target_um"], f"Target for {record['id']}")
        parameters = _finite(sample["parameters"], f"Parameters for {record['id']}")
    if target.shape != shape or parameters.shape != (parameter_count,):
        raise ValueError(f"Invalid target/parameter shape for {record['id']}")
    return parameters, target


def fit_normalization(processed_dir, sample_ids):
    """Fit population means/stds on exactly the selected training samples.

    Target statistics pool all training pixels. Constant dimensions use a scale
    of one physical unit so transforms remain finite and invertible.
    """
    root = Path(processed_dir)
    metadata = load_metadata(root)
    names = parameter_names(metadata)
    records = _select_records(metadata, sample_ids)
    shape = (len(metadata["grid"]["theta_rad"]), len(metadata["grid"]["r_mm"]))
    parameters, target_means, target_variances = [], [], []
    for record in records:
        parameter, target = _load_sample(root, record, shape, len(names))
        parameters.append(parameter)
        target_means.append(float(target.mean()))
        target_variances.append(float(target.var()))
    parameters = np.stack(parameters)
    parameter_mean = parameters.mean(axis=0)
    parameter_scale = parameters.std(axis=0)
    constant = np.ptp(parameters, axis=0) == 0
    parameter_scale[constant] = 1.0
    parameter_mean[constant] = parameters[0, constant]
    offset = float(np.mean(target_means))
    variance = float(np.mean(np.asarray(target_variances) + (np.asarray(target_means) - offset) ** 2))
    target_scale = float(np.sqrt(variance)) if variance > 0 else 1.0
    return {"parameter_names": names, "active_parameter_indices": np.flatnonzero(~constant).tolist(),
            "parameter_mean": parameter_mean.tolist(), "parameter_scale": parameter_scale.tolist(),
            "target_offset_um": offset, "target_scale_um": target_scale}


class MorphologyDataset(Dataset):
    """Read one normalized target and either field or parameter-only input.

    Field channels are declared standardized parameters followed by radius and
    theta in [-1,1]. Main FTPS datasets have six channels. Coordinates are fixed geometry,
    independent of the training distribution.
    """

    def __init__(self, processed_dir, sample_ids, normalization, input_kind="field"):
        self.root = Path(processed_dir)
        self.metadata = load_metadata(self.root)
        self.parameter_names = parameter_names(self.metadata)
        self.normalization = normalization
        self.records = _select_records(self.metadata, sample_ids)
        if input_kind not in {"field", "parameters"}:
            raise ValueError("input_kind must be 'field' or 'parameters'")
        self.input_kind = input_kind
        self.parameter_mean = _finite(normalization["parameter_mean"], "parameter_mean")
        self.parameter_scale = _finite(normalization["parameter_scale"], "parameter_scale")
        self.target_offset = float(normalization["target_offset_um"])
        self.target_scale = float(normalization["target_scale_um"])
        if (self.parameter_mean.shape != (len(self.parameter_names),)
                or self.parameter_scale.shape != (len(self.parameter_names),)
                or normalization.get("parameter_names", self.parameter_names) != self.parameter_names):
            raise ValueError("Parameter normalization must match the declared input names and dimensions")
        if (self.parameter_scale <= 0).any() or not np.isfinite(self.target_scale) or self.target_scale <= 0:
            raise ValueError("All normalization scales must be finite and positive")
        if not np.isfinite(self.target_offset):
            raise ValueError("target_offset_um must be finite")
        r = np.asarray(self.metadata["grid"]["r_mm"])
        theta = np.asarray(self.metadata["grid"]["theta_rad"])
        self.shape = (len(theta), len(r))
        scaled_r = 2 * (r - r[0]) / (r[-1] - r[0]) - 1
        scaled_theta = 2 * (theta - theta[0]) / (theta[-1] - theta[0]) - 1
        theta_plane, radius_plane = np.meshgrid(scaled_theta, scaled_r, indexing="ij")
        self.coordinates = np.stack((radius_plane, theta_plane))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        parameters, target = _load_sample(self.root, record, self.shape, len(self.parameter_names))
        validate_parameter_domain(parameters, self.normalization)
        parameters = (parameters - self.parameter_mean) / self.parameter_scale
        if self.input_kind == "field":
            planes = np.broadcast_to(parameters[:, None, None], (len(self.parameter_names), *self.shape))
            inputs = np.concatenate((planes, self.coordinates))
        else:
            inputs = parameters
        normalized_target = (target - self.target_offset) / self.target_scale
        return {"input": torch.tensor(inputs, dtype=torch.float32),
                "target": torch.tensor(normalized_target[None], dtype=torch.float32),
                "id": record["id"]}

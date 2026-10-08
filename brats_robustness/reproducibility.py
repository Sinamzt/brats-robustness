"""Fail closed on incompatible cached runs and record model/data provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusion_ids():
    """Optional verified dataset IDs in a text file, one ID per line.

    This cannot detect cross-release renaming. Only use an independently verified
    identity mapping; absence of this file means pretraining overlap is unverified.
    """
    path = os.environ.get("BRATS_PRETRAINING_IDS")
    if not path:
        return set()
    ids = {line.strip() for line in Path(path).read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")}
    if not ids:
        raise ValueError("BRATS_PRETRAINING_IDS is empty")
    return ids


def experiment_identity(cfg, repo_root, study):
    root = Path(repo_root)
    paths = sorted(Path(cfg.train_root).glob("**/*.nii")) + sorted(Path(cfg.train_root).glob("**/*.nii.gz"))
    if not paths:
        raise ValueError("No NIfTI files found in data root")
    # Hash bytes, not just paths or timestamps. This adds one dataset read per study.
    files = [(str(p.relative_to(cfg.train_root)), sha256_file(p)) for p in paths]
    sources = sorted((root / "brats_robustness").glob("*.py"))
    sources += [root / "notebooks" / "robustness_experiments.ipynb", root / "requirements.txt"]
    versions = {}
    for name in ["numpy", "scipy", "pandas", "torch", "monai", "nibabel", "SimpleITK"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    pretrain_path = os.environ.get("BRATS_PRETRAINING_IDS")
    source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in sources if p.suffix != '.ipynb'}
    notebook = root / 'notebooks' / 'robustness_experiments.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
    # Jupyter autosaves outputs while executing. Hash executable sources only.
    source_hashes[str(notebook.relative_to(root))] = hashlib.sha256(json.dumps(code).encode()).hexdigest()
    return {"pipeline_version": 2, "study": study, "configuration": asdict(cfg),
            "dataset_sha256": hashlib.sha256(json.dumps(files).encode()).hexdigest(),
            "source_sha256": source_hashes,
            "environment": {"python": platform.python_version(), **versions},
            "pretraining_ids_sha256": sha256_file(pretrain_path) if pretrain_path else None,
            "pretraining_overlap_status": "provided IDs excluded; cross-release identity mapping must be verified"
                if pretrain_path else "unverified; test set held out from local fine-tuning only"}


def guard_run(out_root, identity):
    root = Path(out_root)
    marker = root / "run_metadata.json"
    canonical = json.loads(json.dumps(identity, sort_keys=True, default=str))
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if recorded != canonical:
            raise RuntimeError(f"Incompatible cached experiment at {root}. Choose a new output root; do not reuse old predictions.")
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Untracked outputs at {root}. Choose a fresh output root.")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n")


def guard_bundle(out_root, bundle_path, version):
    path = Path(bundle_path)
    metadata = json.loads((path / "configs" / "metadata.json").read_text())
    if metadata.get("version") != version:
        raise RuntimeError(f"Bundle version {metadata.get('version')} != pinned {version}")
    record = {"version": version, "weights_sha256": sha256_file(path / "models" / "model.pt"),
              "inference_config_sha256": sha256_file(path / "configs" / "inference.json"),
              "metadata_sha256": sha256_file(path / "configs" / "metadata.json")}
    marker = Path(out_root) / "model_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != record:
        raise RuntimeError("Bundle weights/configuration changed: use a new experiment output root")
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def guard_prediction_dir(out_dir, net, condition):
    """Detect changed fine-tuned weights even if a checkpoint is manually replaced."""
    import numpy as np

    h = hashlib.sha256()
    for key, tensor in sorted(net.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str((array.shape, array.dtype)).encode())
        h.update(np.asarray(array).tobytes())
    root = Path(out_dir)
    record = {"model_state_sha256": h.hexdigest(), "condition": condition}
    marker = root / "prediction_provenance.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise RuntimeError(f"Model/condition changed for predictions in {root}; use a new output root")
    elif root.exists() and any(p.name.endswith('.nii.gz') for p in root.iterdir()):
        raise RuntimeError(f"Untracked predictions in {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

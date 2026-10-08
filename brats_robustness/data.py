"""BraTS manifest, split, input geometry checks and nonzero normalization."""
import numpy as np
import pandas as pd
from .reproducibility import exclusion_ids


def build_manifest(root):
    excluded = exclusion_ids()
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name in excluded:
            continue
        files = {}
        for suffix in ["flair", "t1", "t1ce", "t2", "seg"]:
            candidates = [folder / f"{folder.name}_{suffix}{ext}" for ext in [".nii", ".nii.gz"]]
            match = next((p for p in candidates if p.exists()), None)
            if match is None:
                raise ValueError(f"Incomplete patient {folder.name}: missing {suffix}")
            files[suffix] = str(match)
        rows.append({"pid": folder.name, **files})
    if not rows:
        raise ValueError("No complete patient folders; point BRATS_DATA_ROOT directly at their parent")
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)


def make_split(df, train_frac, val_frac, seed):
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Invalid split fractions")
    if not df.pid.is_unique:
        raise ValueError("Duplicate patients")
    pids = sorted(df.pid.tolist())
    np.random.default_rng(seed).shuffle(pids)
    n_train, n_val = round(len(pids) * train_frac), round(len(pids) * val_frac)
    if min(n_train, n_val, len(pids) - n_train - n_val) < 1:
        raise ValueError("Not enough patients for nonempty splits")
    assignment = {p: "train" if i < n_train else "val" if i < n_train + n_val else "test"
                  for i, p in enumerate(pids)}
    return df.assign(split=df.pid.map(assignment))


def load_or_create_split(manifest, path, train_frac, val_frac, seed):
    if path.exists():
        assignments = pd.read_csv(path, usecols=["pid", "split"])
        if not assignments.pid.is_unique or set(assignments.pid) != set(manifest.pid):
            raise ValueError("Cached split does not match current dataset IDs")
        if not assignments.split.isin(["train", "val", "test"]).all():
            raise ValueError("Invalid split labels")
        return assignments.merge(manifest, on="pid", validate="one_to_one")
    df = make_split(manifest, train_frac, val_frac, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["pid", "split"]].to_csv(path, index=False)
    return df


def load_4mod_volume(row, suffix_to_channel):
    import nibabel as nib
    channels = [None] * 4
    affine, shape = None, None
    for suffix in ["flair", "t1", "t1ce", "t2"]:
        image = nib.load(row[suffix])
        x = image.get_fdata(dtype=np.float32)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError(f"Invalid 3D MRI for {row['pid']} / {suffix}")
        if not np.allclose(image.header.get_zooms()[:3], [1, 1, 1]):
            raise ValueError("Pipeline expects BraTS isotropic 1mm volumes")
        if affine is None:
            affine, shape = image.affine, x.shape
        elif shape != x.shape or not np.allclose(affine, image.affine):
            raise ValueError("MRI modalities must be co-registered with identical affines")
        channels[suffix_to_channel[suffix]] = x
    seg = nib.load(row["seg"])
    if seg.shape != shape or not np.allclose(seg.affine, affine):
        raise ValueError("Segmentation geometry differs from the input MRI")
    return np.stack(channels), affine


def znorm_nonzero(vol):
    out = np.zeros_like(vol, dtype=np.float32)
    for c, x in enumerate(vol):
        mask = x != 0
        if mask.any():
            out[c][mask] = (x[mask] - x[mask].mean()) / (x[mask].std() + 1e-8)
    return out

"""Patient-level segmentation metrics; empty-mask HD95 uses volume diagonal.

The diagonal penalty is a project convention, not a BraTS challenge metric.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from monai.metrics import compute_hausdorff_distance
from .regions import gt_to_regions, pred_to_regions
log = logging.getLogger(__name__)

def _voxel_volume_ml(affine: np.ndarray) -> float:
    return abs(np.linalg.det(affine[:3, :3])) / 1000.0

def case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                 spacing: Tuple[float, float, float], voxel_ml: float) -> Dict[str, float]:
    """Per-case metrics for one (pred_mask, gt_mask) pair of binary 3-D arrays."""
    p = pred_mask.astype(bool); g = gt_mask.astype(bool)
    if not p.any() and not g.any():
        return {"DSC": 1.0, "HD95": 0.0, "Sens": 1.0, "FPvol": 0.0, "FNvol": 0.0}
    if not p.any() or not g.any():
        diag = float(np.linalg.norm(np.array(p.shape) * np.array(spacing)))
        return {"DSC": 0.0, "HD95": diag,
                "Sens": 0.0 if g.any() else 1.0,
                "FPvol": p.sum() * voxel_ml,
                "FNvol": g.sum() * voxel_ml}
    inter = np.logical_and(p, g).sum()
    dsc = 2.0 * inter / (p.sum() + g.sum())
    sens = inter / g.sum()
    fpvol = np.logical_and(p, ~g).sum() * voxel_ml
    fnvol = np.logical_and(~p, g).sum() * voxel_ml
    pt = torch.from_numpy(p[None, None]).float()
    gt = torch.from_numpy(g[None, None]).float()
    hd95 = compute_hausdorff_distance(pt, gt, percentile=95, include_background=False,
                                       spacing=spacing).item()
    return {"DSC": float(dsc), "HD95": float(hd95), "Sens": float(sens),
            "FPvol": float(fpvol), "FNvol": float(fnvol)}

def evaluate_condition(df_subset: pd.DataFrame, pred_dir: Path, condition_name: str) -> pd.DataFrame:
    """For every patient in df_subset, load its prediction from pred_dir, compute metrics
    against GT, return a long-format DataFrame with one row per (patient, region).
    """
    rows = []
    for _, row in df_subset.iterrows():
        pid = row["pid"]
        pred_path = pred_dir / f"{pid}_pred.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        pred_nii = nib.load(str(pred_path))
        pred = pred_nii.get_fdata().astype(np.int16)
        gt_nii = nib.load(row["seg"])
        gt = gt_nii.get_fdata().astype(np.int16)
        if pred.shape != gt.shape or not np.allclose(pred_nii.affine, gt_nii.affine):
            raise ValueError(f"Prediction/GT geometry mismatch: {pid}")
        spacing = tuple(map(float, gt_nii.header.get_zooms()[:3]))
        voxel_ml = _voxel_volume_ml(gt_nii.affine)
        p_regs = pred_to_regions(pred)
        g_regs = gt_to_regions(gt)
        for region in ["WT", "TC", "ET"]:
            m = case_metrics(p_regs[region], g_regs[region], spacing, voxel_ml)
            rows.append({"pid": pid, "condition": condition_name, "region": region, **m})
    return pd.DataFrame(rows)

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case → mean ± std and median[IQR] per (condition, region)."""
    metrics = ["DSC", "HD95", "Sens", "FPvol", "FNvol"]
    agg = df.groupby(["condition", "region"])[metrics].agg(
        ["mean", "std", "median",
         lambda x: np.percentile(x, 25),
         lambda x: np.percentile(x, 75)]
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().rename(
        columns={"DSC_<lambda_0>": "DSC_q25", "DSC_<lambda_1>": "DSC_q75",
                 "HD95_<lambda_0>": "HD95_q25", "HD95_<lambda_1>": "HD95_q75",
                 "Sens_<lambda_0>": "Sens_q25", "Sens_<lambda_1>": "Sens_q75",
                 "FPvol_<lambda_0>": "FPvol_q25", "FPvol_<lambda_1>": "FPvol_q75",
                 "FNvol_<lambda_0>": "FNvol_q25", "FNvol_<lambda_1>": "FNvol_q75"}
    )


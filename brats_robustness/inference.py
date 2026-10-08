"""Sliding-window inference for raw, aligned 4-modality MRI."""
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from .data import znorm_nonzero

@torch.no_grad()
def infer_volume(net: torch.nn.Module, vol: np.ndarray, cfg, device) -> np.ndarray:
    """Run sliding-window inference on one (4, H, W, D) raw volume.

    Returns sigmoid probabilities (3, H, W, D) in (TC, WT, ET) channel order.
    Sliding-window is necessary because the model was trained on 224^3 patches and
    inference is on full 240×240×155 volumes which exceed that.
    """
    norm = znorm_nonzero(vol)
    x = torch.from_numpy(norm).unsqueeze(0).to(device)  # (1, 4, H, W, D)
    logits = sliding_window_inference(
        inputs=x, roi_size=cfg.roi_size, sw_batch_size=cfg.sw_batch,
        predictor=net, overlap=cfg.sw_overlap, mode="gaussian",
    )
    return torch.sigmoid(logits)[0].cpu().numpy()

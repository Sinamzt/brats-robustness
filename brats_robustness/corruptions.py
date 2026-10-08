"""Synthetic corruptions and preprocessing, with an explicit clean support mask."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def corrupt_gaussian_noise(flair, sigma_rel, seed=None):
    if sigma_rel < 0:
        raise ValueError("Noise severity must be nonnegative")
    x = np.asarray(flair, dtype=np.float32)
    mask = x != 0
    out = x.copy()
    if mask.any() and sigma_rel:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma_rel * float(x[mask].std()), size=x.shape)
        out[mask] += noise[mask].astype(np.float32)
    return out


def gen_bias_field(shape, seed, strength, n_blobs=4):
    """Positive exp(strength * smooth_field), bounded by exp(+/-strength).

    Strength is log-amplitude, not the legacy additive-deviation parameter.
    The field has no sign changes and never requires clipping corrupted tissue.
    """
    if strength < 0 or not np.isfinite(strength) or strength > 20:
        raise ValueError("Log-amplitude strength must be finite and in [0, 20]")
    if len(shape) != 3 or min(shape) < 1 or n_blobs < 1:
        raise ValueError("Expected a nonempty 3D shape and at least one blob")
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(n_blobs):
        center = rng.uniform(-0.8, 0.8, 3)
        sign = rng.choice([-1.0, 1.0])
        width = rng.uniform(0.55, 1.10)
        amplitude = sign * rng.uniform(0.5, 1.0)
        distance = sum((g - c) ** 2 for g, c in zip(grid, center))
        field += (amplitude * np.exp(-distance / (2 * width**2))).astype(np.float32)
    peak = float(np.abs(field).max())
    smooth = field / peak if peak > 0 else field
    return np.exp(float(strength) * smooth).astype(np.float32)


def corrupt_bias_field(flair, strength, seed=None):
    x = np.asarray(flair, dtype=np.float32)
    if (x < 0).any():
        raise ValueError("Bias corruption expects nonnegative raw MRI intensities")
    return x * gen_bias_field(x.shape, seed, strength)


def _mask(volume, brain_mask):
    mask = np.asarray(brain_mask, dtype=bool)
    if mask.shape != np.shape(volume):
        raise ValueError("Clean brain mask must match the volume shape")
    return mask


def filter_identity(volume, brain_mask):
    out = np.asarray(volume, dtype=np.float32).copy()
    out[~_mask(out, brain_mask)] = 0
    return out


def filter_mean_3(volume, brain_mask):
    out = ndimage.uniform_filter(np.asarray(volume, dtype=np.float32), size=3,
                                 mode="constant", cval=0)
    out[~_mask(out, brain_mask)] = 0
    return out


def filter_gaussian(volume, sigma, brain_mask):
    if sigma <= 0:
        raise ValueError("Filter sigma must be positive")
    # Separable equivalent of the original finite normalized 3D Gaussian kernel.
    radius = max(round(3 * sigma), 1)
    out = ndimage.gaussian_filter(np.asarray(volume, dtype=np.float32), sigma,
                                  radius=radius, mode="constant", cval=0)
    out[~_mask(out, brain_mask)] = 0
    return out


def filter_n4(volume, brain_mask):
    import SimpleITK as sitk

    mask = _mask(volume, brain_mask)
    x = np.asarray(volume, dtype=np.float32)
    if not mask.any():
        return x.copy()
    if (x[mask] <= 0).any():
        raise ValueError("N4 requires strictly positive intensities inside the mask")
    # BraTS is pre-resampled to isotropic 1mm; the I/O helper checks this.
    image = sitk.GetImageFromArray(x)
    image.SetSpacing((1.0, 1.0, 1.0))
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 20, 20])
    out = sitk.GetArrayFromImage(corrector.Execute(image, mask_image)).astype(np.float32)
    out[~mask] = 0
    return out


def make_filter_bank(include_n4=False):
    bank = {"none": filter_identity, "mean_3": filter_mean_3,
            "gaussian_0p5": lambda v, m: filter_gaussian(v, 0.5, m),
            "gaussian_1p0": lambda v, m: filter_gaussian(v, 1.0, m),
            "gaussian_1p5": lambda v, m: filter_gaussian(v, 1.5, m)}
    if include_n4:
        bank["n4"] = filter_n4
    for name, function in bank.items():
        function.preprocessing_name = name
    return bank

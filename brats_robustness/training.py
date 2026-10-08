"""Loss for overlapping, independently predicted tumor regions."""


def make_region_loss():
    import torch
    from monai.losses import DiceLoss

    class RegionLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = DiceLoss(sigmoid=True, include_background=True)
            self.bce = torch.nn.BCEWithLogitsLoss()

        def forward(self, logits, targets):
            if logits.shape != targets.shape or logits.shape[1] != 3:
                raise ValueError("Expected matching Bx3xHxWxD region tensors")
            # Channel 0 is TC, not background. Keep all three channels.
            return self.dice(logits, targets) + self.bce(logits, targets)

    return RegionLoss()

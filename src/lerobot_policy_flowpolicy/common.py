import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_mse(pred: Tensor, target: Tensor, batch: dict, mask_padding: bool) -> Tensor:
    """MSE over (B, T, D), optionally ignoring padded actions. Same rule as LeRobot's Diffusion Policy."""
    loss = F.mse_loss(pred, target, reduction="none")
    if not mask_padding:
        return loss.mean()
    mask = (~batch["action_is_pad"]).unsqueeze(-1)
    return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)

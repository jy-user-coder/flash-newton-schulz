import torch
import torch.nn.functional as F


def get_aol_conv2d_rescale(weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    oc, ic, kH, kW = weight.shape
    x = weight.permute(1, 0, 2, 3).contiguous()
    # Filters: out_channels=ic, in_channels=oc, H=kH, W=kW
    filt = weight.permute(1, 0, 2, 3).contiguous()
    # Full conv padding to capture all overlaps
    v = F.conv2d(x, filt, stride=1, padding=(kH - 1, kW - 1))  # [ic, ic, 2kH-1, 2kW-1]
    lipschitz_bounds_squared = v.abs().sum(dim=(1, 2, 3))  # [ic]
    factors = (lipschitz_bounds_squared + epsilon).pow(-0.5)
    return factors


@torch.no_grad()
def aol_conv2d_rescale(weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    factors = get_aol_conv2d_rescale(weight, epsilon)  # [in_channels]
    return weight * factors[None, :, None, None]

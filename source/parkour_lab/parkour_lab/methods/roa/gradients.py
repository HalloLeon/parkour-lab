"""Stable L2 clipping for ROA's dense gradient groups."""

import torch


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm):
    """Keep the existing clipping rule without overflowing finite float32 norms."""
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64)
    # Finite float32 entries can overflow a float32 sum of squares. Accumulate
    # both tensor and group norms in float64, keeping the original gradient dtype.
    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(gradient, dtype=torch.float64)
                for gradient in gradients
            ]
        )
    )
    if not torch.isfinite(total_norm):
        raise RuntimeError("The total gradient norm is non-finite")
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient)
    return total_norm

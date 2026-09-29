"""PyTorch components for terrain-conditioned actor/critic methods."""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_TERRAIN_LATENT_DIM = 32


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    """Build an ELU MLP with a linear output layer."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class PrivilegedScanEncoder(nn.Module):
    """Compress simulator-only terrain scans into the shared terrain latent."""

    def __init__(
        self,
        scan_dim: int,
        latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()

        if scan_dim <= 0 or latent_dim <= 0:
            raise ValueError("Scan and terrain-latent dimensions must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("Scan-encoder hidden dimensions must be positive.")
        self.network = build_mlp(scan_dim, latent_dim, hidden_dims)

    def forward(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        """Return one fixed-width terrain latent per environment."""
        return self.network(terrain_scan)


class StockTerrainInput(nn.Module):
    """Terrain conditioning without changing the stock first-layer arithmetic.

    Inputs are the original 48 motor values followed by 132 heights and their
    132 validity bits. A zero-initialized projection adds the scan encoding to
    the first hidden preactivation. Initially the motor is exactly unchanged,
    even for nonzero terrain inputs; after learning this is one conditioned
    motor, not an action blend or a policy switch. This is a teacher input:
    the original simulator-velocity values are still privileged.
    """

    def __init__(self, reference: nn.Linear, *, critic_task_dim: int = 0) -> None:
        super().__init__()
        if not isinstance(reference, nn.Linear) or (
            reference.in_features,
            reference.out_features,
        ) != (48, 128):
            raise ValueError("Terrain conditioning requires a 48-to-128 layer.")
        if type(critic_task_dim) is not int or critic_task_dim not in (0, 4):
            raise ValueError(
                "Only the explicit four-value critic task schema is supported."
            )
        self.critic_task_dim = critic_task_dim
        self.reference = reference
        self.encoder = PrivilegedScanEncoder(264)
        self.projection = nn.Linear(DEFAULT_TERRAIN_LATENT_DIM, 128, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.encoder.to(reference.weight)
        self.projection.to(reference.weight)
        if critic_task_dim:
            # Critic-only task state must not change any existing initialization
            # or the subsequent rollout RNG stream (including on CUDA).
            device = reference.weight.device
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                self.task_projection = nn.Linear(
                    critic_task_dim,
                    128,
                    bias=False,
                    device=device,
                    dtype=reference.weight.dtype,
                )
                nn.init.zeros_(self.task_projection.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != 312 + self.critic_task_dim:
            raise ValueError(
                "Expected 48 stock + 264 terrain values and the declared critic task schema."
            )
        # Preserve the original contiguous 48-column GEMM, not a wider GEMM
        # whose reduction order can differ despite zero additional weights.
        state = observations[..., :48].contiguous()
        terrain = observations[..., 48:312]
        hidden = self.reference(state) + self.projection(self.encoder(terrain))
        if self.critic_task_dim:
            hidden = hidden + self.task_projection(observations[..., 312:])
        return hidden

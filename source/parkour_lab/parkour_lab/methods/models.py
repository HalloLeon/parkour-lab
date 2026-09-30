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


class TerrainConditionedInput(nn.Module):
    """Add a privileged 132-height/132-validity scan to a critic preactivation."""

    def __init__(self, reference: nn.Linear) -> None:
        super().__init__()
        if not isinstance(reference, nn.Linear):
            raise ValueError("Terrain conditioning requires a linear state input.")
        self.reference = reference
        self.encoder = PrivilegedScanEncoder(264)
        self.projection = nn.Linear(
            DEFAULT_TERRAIN_LATENT_DIM, reference.out_features, bias=False
        )
        nn.init.zeros_(self.projection.weight)
        self.encoder.to(reference.weight)
        self.projection.to(reference.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        width = self.reference.in_features
        if observations.shape[-1] != width + 264:
            raise ValueError("Expected critic state followed by 264 terrain values.")
        state = observations[..., :width].contiguous()
        terrain = observations[..., width:]
        return self.reference(state) + self.projection(self.encoder(terrain))

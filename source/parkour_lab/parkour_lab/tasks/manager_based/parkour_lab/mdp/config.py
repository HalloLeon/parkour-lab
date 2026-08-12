from isaaclab.utils import configclass

# ==================== OBSERVATION CONFIGURATIONS ====================


@configclass
class HeightScanObservationCfg:
    """
    Configuration for the Phase 1 teacher's privileged terrain-height scan.

    The future deployable student will not consume this simulator ray cast;
    its terrain representation is planned to come from onboard depth instead.
    """

    num_rays: int = 132
    """Fixed number of ray samples in each flattened height and validity term."""

    vertical_offset: float = 0.3
    """Reference-plane distance below the robot root, in metres."""

    clip: float = 1.0
    """Symmetric metric clipping bound in metres, also used as the fixed normalization divisor."""

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError("num_rays must be positive.")

        if self.clip <= 0.0:
            raise ValueError("clip must be positive.")


DEFAULT_HEIGHT_SCAN_OBSERVATION = HeightScanObservationCfg()

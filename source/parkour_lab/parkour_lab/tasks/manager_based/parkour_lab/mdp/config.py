from dataclasses import field

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


# ==================== REWARD CONFIGURATIONS ====================


@configclass
class FeetMotionCfg:
    """
    Configuration for contact-aware foot-speed penalties.

    Stance feet are expected to move slowly.
    Swing feet may move faster, but not violently.
    """

    max_stance_speed: float = 0.25
    max_swing_speed: float = 2.00
    contact_threshold: float = 1.0
    max_penalty_per_foot: float = 4.0

    def __post_init__(self) -> None:
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate contact-aware foot-motion limits."""

        if self.max_stance_speed < 0.0:
            raise ValueError("max_stance_speed must be non-negative.")

        if self.max_swing_speed <= self.max_stance_speed:
            raise ValueError("max_swing_speed must be greater than max_stance_speed.")

        if self.contact_threshold < 0.0:
            raise ValueError("contact_threshold must be non-negative.")

        if self.max_penalty_per_foot <= 0.0:
            raise ValueError("max_penalty_per_foot must be positive.")


DEFAULT_FOOT_MOTION_PENALTY = FeetMotionCfg()


@configclass
class FeetStumbleCfg:
    """Configuration for detecting foot impacts against near-vertical surfaces."""

    lateral_to_vertical_force_ratio: float = 1.0
    min_vertical_force: float = 0.5

    def __post_init__(self) -> None:
        if self.lateral_to_vertical_force_ratio <= 0.0:
            raise ValueError("lateral_to_vertical_force_ratio must be positive.")

        if self.min_vertical_force < 0.0:
            raise ValueError("min_vertical_force must be non-negative.")


DEFAULT_FEET_STUMBLE = FeetStumbleCfg()


@configclass
class GoalHeadingCfg:
    """
    Configuration for heading-misalignment penalties while advancing.
    """

    max_heading_error: float = 0.5
    min_forward_speed: float = 0.2
    full_forward_speed: float = 1.0

    def __post_init__(self) -> None:
        if self.max_heading_error <= 0.0:
            raise ValueError("max_heading_error must be positive.")

        if self.min_forward_speed < 0.0:
            raise ValueError("min_forward_speed must be non-negative.")

        if self.full_forward_speed <= self.min_forward_speed:
            raise ValueError(
                "full_forward_speed must be greater than min_forward_speed."
            )


DEFAULT_GOAL_HEADING = GoalHeadingCfg()


@configclass
class RootMotionChatterCfg:
    """
    Configuration for penalizing small, rapid root/trunk oscillations.

    This targets:
      - small vertical bounces that quickly reverse,
      - small roll/pitch wiggles that quickly reverse.

    It does not directly penalize large vertical motion, so larger step-up,
    jump, or obstacle traversal motions remain possible.
    """

    small_z_displacement: float = 0.035
    min_z_reversal_speed: float = 0.10
    small_tilt_change: float = 0.04
    min_roll_pitch_reversal_rate: float = 0.75
    angular_weight: float = 0.25
    reset_grace_steps: int = 1

    def __post_init__(self) -> None:
        if self.small_z_displacement <= 0.0:
            raise ValueError("small_z_displacement must be positive.")

        if self.min_z_reversal_speed < 0.0:
            raise ValueError("min_z_reversal_speed must be non-negative.")

        if self.small_tilt_change <= 0.0:
            raise ValueError("small_tilt_change must be positive.")

        if self.min_roll_pitch_reversal_rate < 0.0:
            raise ValueError("min_roll_pitch_reversal_rate must be non-negative.")

        if self.angular_weight < 0.0:
            raise ValueError("angular_weight must be non-negative.")

        if self.reset_grace_steps < 0:
            raise ValueError("reset_grace_steps must be non-negative.")


DEFAULT_ROOT_MOTION_CHATTER = RootMotionChatterCfg()


@configclass
class RootStabilityCfg:
    """
    Configuration for root-stability checks.

    Stability here means:
      - not rotating too violently in roll/pitch,
      - not tilted too far,
      - not too close to the support surface underneath the base.

    The support surface may be:
      - flat ground,
      - an obstacle top,
      - later another terrain/platform surface.
    """

    max_roll_pitch_ang_speed: float = 4.0
    max_projected_gravity_xy_norm: float = 0.75

    def __post_init__(self) -> None:
        if self.max_roll_pitch_ang_speed <= 0.0:
            raise ValueError("max_roll_pitch_ang_speed must be positive.")

        if self.max_projected_gravity_xy_norm <= 0.0:
            raise ValueError("max_projected_gravity_xy_norm must be positive.")


@configclass
class StableGoalProgressCfg:
    """
    Configuration for stable XY goal-progress reward.

    The reward has three parts:
      - positive reward for reducing XY distance to the goal,
      - negative penalty for increasing XY distance,
      - small penalty for sideways/lateral drift while making progress.
    """

    progress_scale: float = 0.03
    reset_grace_steps: int = 1
    stability: RootStabilityCfg = field(default_factory=RootStabilityCfg)

    # Do not clamp progress to [-1, 1] too early.
    # Higher caps preserve a gradient between okay progress and very direct progress.
    max_positive_reward: float = 2.0
    max_negative_penalty: float = 2.0

    # Penalizes curved/sideways motion while still allowing small corrections.
    lateral_drift_weight: float = 0.25
    max_lateral_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.progress_scale <= 0.0:
            raise ValueError("progress_scale must be positive.")

        if self.reset_grace_steps < 0:
            raise ValueError("reset_grace_steps must be non-negative.")

        if self.max_positive_reward <= 0.0:
            raise ValueError("max_positive_reward must be positive.")

        if self.max_negative_penalty <= 0.0:
            raise ValueError("max_negative_penalty must be positive.")

        if self.lateral_drift_weight < 0.0:
            raise ValueError("lateral_drift_weight must be non-negative.")

        if self.max_lateral_penalty <= 0.0:
            raise ValueError("max_lateral_penalty must be positive.")


DEFAULT_STABLE_GOAL_PROGRESS = StableGoalProgressCfg()

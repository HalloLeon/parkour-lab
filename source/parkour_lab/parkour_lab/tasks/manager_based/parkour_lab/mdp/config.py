from dataclasses import dataclass
from dataclasses import field

from isaaclab.utils import configclass

from .curriculums import DEFAULT_PARKOUR_CURRICULUM


GROUND_HEIGHT = 0.0

# ==================== OBSERVATION CONFIGURATIONS ====================


@configclass
class GoalSlotsObservationCfg:
    """
    Configuration for fixed-size goal-slot observations.

    Each goal slot contains:
        direction_body_xy: 2
        normalized_distance: 1

    So each slot contributes 3 values.
    """

    num_slots: int = 2
    max_distance: float = 5.0

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive.")

        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be positive.")


DEFAULT_GOAL_SLOTS_OBSERVATION = GoalSlotsObservationCfg()


@configclass
class HeightScanObservationCfg:
    """
    Configuration for terrain/obstacle height-scan observations.

    This is intended for teacher/critic use, not for the first deployable actor.
    """

    num_rays: int = 132
    vertical_offset: float = 0.3
    clip: float = 1.0

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError("num_rays must be positive.")

        if self.clip <= 0.0:
            raise ValueError("clip must be positive.")


DEFAULT_HEIGHT_SCAN_OBSERVATION = HeightScanObservationCfg()


# ==================== REWARD CONFIGURATIONS ====================


@dataclass(frozen=True)
class BoxSurfaceCfg:
    """Configuration for a box-shaped support surface."""

    name: str
    size: tuple[float, float, float]
    xy_margin: float = 0.02


OBSTACLE_SURFACE = BoxSurfaceCfg(
    name="obstacle",
    size=(
        max(level.obstacle_size[0] for level in DEFAULT_PARKOUR_CURRICULUM.levels),
        max(level.obstacle_size[1] for level in DEFAULT_PARKOUR_CURRICULUM.levels),
        max(level.obstacle_size[2] for level in DEFAULT_PARKOUR_CURRICULUM.levels)
    ),
    xy_margin=0.02
)


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

    def validate(self) -> None:
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
class GoalVelocityCfg:
    """
    Configuration for goal-directed XY velocity tracking.

    The robot is rewarded for matching a desired velocity along the
    direction to the goal.

    Far from the goal:
        desired speed ≈ target_speed

    Near the goal:
        desired speed smoothly decreases to avoid overshooting.
    """

    target_speed: float = 0.6
    speed_tracking_scale: float = 0.2
    slow_down_distance: float = 0.5
    min_clearance: float = 0.25

    def __post_init__(self) -> None:
        if self.target_speed < 0.0:
            raise ValueError("target_speed must be non-negative.")

        if self.speed_tracking_scale <= 0.0:
            raise ValueError("speed_tracking_scale must be positive.")

        if self.slow_down_distance <= 0.0:
            raise ValueError("slow_down_distance must be positive.")

        if self.min_clearance < 0.0:
            raise ValueError("min_clearance must be non-negative.")


DEFAULT_GOAL_VELOCITY = GoalVelocityCfg()


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
    min_clearance: float = 0.25

    def __post_init__(self) -> None:
        if self.max_roll_pitch_ang_speed <= 0.0:
            raise ValueError("max_roll_pitch_ang_speed must be positive.")

        if self.max_projected_gravity_xy_norm <= 0.0:
            raise ValueError("max_projected_gravity_xy_norm must be positive.")

        if self.min_clearance < 0.0:
            raise ValueError("min_clearance must be non-negative.")


@configclass
class StableGoalProgressCfg:
    """
    Configuration for stable XY goal-progress reward.
    """

    progress_scale: float = 0.03
    reset_grace_steps: int = 1
    stability: RootStabilityCfg = field(default_factory=RootStabilityCfg)

    def __post_init__(self) -> None:
        if self.progress_scale <= 0.0:
            raise ValueError("progress_scale must be positive.")

        if self.reset_grace_steps < 0:
            raise ValueError("reset_grace_steps must be non-negative.")


DEFAULT_STABLE_GOAL_PROGRESS = StableGoalProgressCfg()

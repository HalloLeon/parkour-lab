from dataclasses import dataclass

from isaaclab.utils import configclass


GROUND_HEIGHT = 0.0


@dataclass(frozen=True)
class BoxSurfaceCfg:
    """Configuration for a box-shaped support surface."""

    name: str
    size: tuple[float, float, float]
    xy_margin: float = 0.02


OBSTACLE_SURFACE = BoxSurfaceCfg(
    name="obstacle",
    size=(0.5, 0.5, 0.12),
    xy_margin=0.02,
)


@configclass
class FootMotionPenaltyCfg:
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


DEFAULT_FOOT_MOTION_PENALTY = FootMotionPenaltyCfg()


@dataclass(frozen=True)
class GoalVelocityTrackingCfg:
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


DEFAULT_GOAL_VELOCITY_TRACKING = GoalVelocityTrackingCfg()

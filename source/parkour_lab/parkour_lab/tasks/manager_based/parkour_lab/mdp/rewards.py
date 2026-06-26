from dataclasses import dataclass

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import torch

from . import term_cfg
from . import utils


def base_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg=SceneEntityCfg("base_contact", body_names="trunk")
) -> torch.Tensor:
    """
    Penalty signal for illegal trunk/base contact.

    Returns:
        Tensor of shape [num_envs].
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, selected_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # [num_envs]
    has_illegal_contact = torch.any(force_norm > threshold, dim=(1, 2))

    return has_illegal_contact.float()


def goal_progress_xy_stable(
    env: ManagerBasedRLEnv,
    progress_cfg: term_cfg.StableGoalProgressCfg = term_cfg.DEFAULT_STABLE_GOAL_PROGRESS,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Dense reward for stable reduction of XY distance to the goal.

    progress = previous_distance - current_distance

    Positive progress is counted only when the robot root is stable.
    Negative progress is preserved, so moving away from the goal is still
    penalized even if the robot is unstable.

    Stability includes:
      - limited roll/pitch angular velocity,
      - limited roll/pitch tilt,
      - sufficient base/root clearance above the support surface underneath it.

    The support surface may be flat ground, an obstacle top, or later another
    terrain/platform surface.

    Returns:
        [num_envs]
    """

    current_distance = utils._goal_distance_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    buffer_name = utils._private_buffer_name(
        "parkour_prev_goal_distance_xy",
        goal_cfg.name,
        asset_cfg.name
    )

    just_reset = utils._episode_start_mask(
        env,
        reference=current_distance,
        grace_steps=progress_cfg.reset_grace_steps
    )

    progress = utils._difference_from_previous_env_buffer(
        env,
        buffer_name=buffer_name,
        current_value=current_distance,
        reset_mask=just_reset
    )

    stable = utils._root_stability_mask(
        env,
        stability_cfg=progress_cfg.stability,
        asset_cfg=asset_cfg
    )

    progress = utils._gate_positive_values(
        values=progress,
        gate=stable
    )

    normalized_progress = progress / progress_cfg.progress_scale

    return torch.clamp(
        normalized_progress,
        min=-1.0,
        max=1.0
    )


def goal_heading_misalignment_l2(
    env: ManagerBasedRLEnv,
    heading_cfg: term_cfg.GoalHeadingCfg = term_cfg.DEFAULT_GOAL_HEADING,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize heading misalignment only while the robot is advancing toward
    the XY goal.

    This avoids rewarding the robot for merely staring at the goal while
    standing still.

    The penalty is active only when velocity along the goal direction is
    positive enough.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    heading_error = utils._heading_error_to_goal_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    velocity_along_goal = utils._velocity_along_goal_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    advancing_gate = utils._linear_ramp(
        value=velocity_along_goal,
        lower=heading_cfg.min_forward_speed,
        upper=heading_cfg.full_forward_speed
    )

    normalized_heading_error = torch.clamp(
        heading_error / heading_cfg.max_heading_error,
        min=0.0,
        max=1.0
    )

    return advancing_gate * normalized_heading_error.square()


def reached_goal_xy(
    env: ManagerBasedRLEnv,
    threshold: float = 0.25,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Sparse success reward based on XY goal distance.

    Returns:
        [num_envs]
    """

    dist_to_goal = utils._goal_distance_xy(env, goal_cfg, asset_cfg)
    reached = dist_to_goal < threshold

    return reached.float()


def velocity_along_goal_xy_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: term_cfg.GoalVelocityCfg = term_cfg.DEFAULT_GOAL_VELOCITY,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward tracking a desired XY velocity along the direction to the goal.

    Far from the goal:
        desired velocity is close to tracking_cfg.target_speed.

    Near the goal:
        desired velocity decreases toward zero to reduce overshooting.

    This reward does not check whether the robot is upright or has enough
    clearance. Use velocity_along_goal_xy_clearance_exp for the gated version.

    Returns:
        [num_envs]
    """

    velocity_along_goal = utils._velocity_along_goal_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    goal_dist_xy = utils._goal_distance_xy(env, goal_cfg, asset_cfg)

    slowdown_scale = torch.clamp(
        goal_dist_xy / tracking_cfg.slow_down_distance,
        min=0.0,
        max=1.0
    )

    desired_velocity = tracking_cfg.target_speed * slowdown_scale

    velocity_error = velocity_along_goal - desired_velocity

    return torch.exp(
        -velocity_error.square() / tracking_cfg.speed_tracking_scale**2
    )


def velocity_along_goal_xy_clearance_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: term_cfg.GoalVelocityCfg = term_cfg.DEFAULT_GOAL_VELOCITY,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Clearance-gated version of velocity_along_goal_xy_exp.

    The velocity reward is only paid when the robot base/root has enough
    clearance above the surface underneath it.

    The surface underneath it may be:
        - flat ground
        - obstacle top
        - later, another terrain/support surface

    This prevents rewarding forward velocity while the robot is collapsed,
    scraping, or too close to the support surface.

    Returns:
        [num_envs]
    """

    reward = velocity_along_goal_xy_exp(
        env,
        tracking_cfg=tracking_cfg,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    clearance = utils._base_clearance(
        env,
        asset_cfg=asset_cfg
    )

    has_enough_clearance = clearance > tracking_cfg.min_clearance

    return reward * has_enough_clearance.to(dtype=reward.dtype)


def base_clearance_below_l2(
    env: ManagerBasedRLEnv,
    min_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalty signal for the robot base/root being too close to the surface
    directly underneath it.

    The surface may be:
      - the ground
      - the top of an obstacle
      - later, another support surface

    This is an L2 penalty:

        penalty = max(min_clearance - clearance, 0)^2

    where:

        clearance = base_height - support_surface_height_under_base

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    clearance = utils._base_clearance(env, asset_cfg)

    clearance_error = torch.clamp(min_clearance - clearance, min=0.0)

    return clearance_error.square()


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize selected joints deviating from their default pose.

    Returns:
        [num_envs]
    """

    joint_error = utils._selected_joint_pos_error(env, asset_cfg)

    return torch.sum(joint_error.square(), dim=-1)


def feet_stumble(
    env: ManagerBasedRLEnv,
    stumble_cfg: term_cfg.FeetStumbleCfg = term_cfg.DEFAULT_FEET_STUMBLE,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Penalize feet hitting near-vertical surfaces.

    A stumble is detected when lateral contact force is large compared with
    vertical contact force.

    Returns:
        [num_envs]
    """

    contact_forces = utils._selected_contact_forces_w_history(
        env,
        sensor_cfg=sensor_cfg
    )

    lateral_force = torch.linalg.norm(contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces[..., 2])

    valid_vertical_contact = vertical_force > stumble_cfg.min_vertical_force

    stumble = torch.logical_and(
        valid_vertical_contact,
        lateral_force
        > stumble_cfg.lateral_to_vertical_force_ratio * vertical_force
    )

    return torch.any(stumble, dim=(1, 2)).float()


def no_feet_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Penalty for having no feet in contact with the ground.

    This discourages hopping/skipping in flat walking.

    Returns:
        [num_envs]
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, num_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # Has each foot contacted recently?
    # [num_envs, num_bodies]
    feet_in_contact = torch.any(force_norm > threshold, dim=1)

    # [num_envs]
    num_feet_in_contact = torch.sum(feet_in_contact.float(), dim=-1)

    no_contact = num_feet_in_contact < 1.0

    return no_contact.float()


def rapid_feet_motion_l2(
    env: ManagerBasedRLEnv,
    motion_cfg: term_cfg.FeetMotionCfg = term_cfg.DEFAULT_FOOT_MOTION_PENALTY,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Penalize excessive foot speed in a contact-aware way.

    Stance feet are expected to move slowly.
    Swing feet are allowed to move faster.

    The penalty is:

        penalty = max(foot_speed - allowed_speed, 0)^2

    where allowed_speed is:
        - max_stance_speed for feet in contact
        - max_swing_speed for feet not in contact

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    motion_cfg.validate()

    foot_speed = utils._selected_body_speed_w(env, asset_cfg)

    in_contact = utils._contact_mask(
        env,
        sensor_cfg=sensor_cfg,
        threshold=motion_cfg.contact_threshold
    )

    utils._validate_matching_shape(
        in_contact,
        foot_speed,
        lhs_name="foot contact mask",
        rhs_name="foot speed"
    )

    stance_speed_limit = torch.full_like(
        foot_speed,
        motion_cfg.max_stance_speed
    )

    swing_speed_limit = torch.full_like(
        foot_speed,
        motion_cfg.max_swing_speed
    )

    speed_limit = torch.where(
        in_contact,
        stance_speed_limit,
        swing_speed_limit
    )

    excess_speed = torch.clamp(
        foot_speed - speed_limit,
        min=0.0
    )

    penalty_per_foot = torch.clamp(
        excess_speed.square(),
        max=motion_cfg.max_penalty_per_foot
    )

    return penalty_per_foot.mean(dim=-1)


def root_chatter_l2(
    env: ManagerBasedRLEnv,
    chatter_cfg: term_cfg.RootMotionChatterCfg = term_cfg.DEFAULT_ROOT_MOTION_CHATTER,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize small, rapid root/core oscillations.

    This targets high-frequency chatter:
      - small vertical bounces that quickly reverse,
      - small roll/pitch wiggles that quickly reverse.

    It does not penalize large vertical motion directly. Larger step-up,
    jump, or obstacle traversal motions are allowed as long as they are not
    small rapid reversals.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    current = _RootChatterState.from_env(
        env,
        asset_cfg=asset_cfg
    )

    buffer_prefix = utils._private_buffer_name(
        "parkour_root_chatter",
        asset_cfg.name
    )

    previous = _RootChatterState.previous_from_env(
        env,
        buffer_prefix=buffer_prefix,
        current=current
    )

    vertical_penalty = _vertical_root_chatter_l2(
        current=current,
        previous=previous,
        chatter_cfg=chatter_cfg
    )

    angular_penalty = _angular_root_chatter_l2(
        current=current,
        previous=previous,
        chatter_cfg=chatter_cfg
    )

    penalty = vertical_penalty + chatter_cfg.angular_weight * angular_penalty

    just_reset = utils._episode_start_mask(
        env,
        reference=penalty,
        grace_steps=chatter_cfg.reset_grace_steps
    )

    penalty = torch.where(
        just_reset,
        torch.zeros_like(penalty),
        penalty
    )

    current.write_to_env(
        env,
        buffer_prefix=buffer_prefix
    )

    return penalty


@dataclass(frozen=True)
class _RootChatterState:
    """
    Root/core signals used by root_chatter_l2.

    This groups tensors that are always used together, reducing argument-heavy
    helper functions without hiding the reward logic.
    """

    root_z: torch.Tensor
    root_z_vel: torch.Tensor
    projected_gravity_xy: torch.Tensor
    roll_pitch_rate: torch.Tensor

    @classmethod
    def from_env(
        cls,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg
    ) -> "_RootChatterState":
        """
        Read current root/core signals from the environment.
        """

        return cls(
            root_z=utils._root_height_env(env, asset_cfg),
            root_z_vel=utils._root_lin_vel_z(env, asset_cfg),
            projected_gravity_xy=utils._root_projected_gravity_xy(env, asset_cfg),
            roll_pitch_rate=utils._root_roll_pitch_rate(env, asset_cfg)
        )

    @classmethod
    def previous_from_env(
        cls,
        env: ManagerBasedRLEnv,
        *,
        buffer_prefix: str,
        current: "_RootChatterState"
    ) -> "_RootChatterState":
        """
        Read previous root/core signals from environment buffers.

        Missing or stale buffers are initialized from the current state.
        """

        return cls(
            root_z=utils._get_or_init_env_buffer(
                env,
                f"{buffer_prefix}_root_z",
                current.root_z
            ),
            root_z_vel=utils._get_or_init_env_buffer(
                env,
                f"{buffer_prefix}_root_z_vel",
                current.root_z_vel
            ),
            projected_gravity_xy=utils._get_or_init_env_buffer(
                env,
                f"{buffer_prefix}_projected_gravity_xy",
                current.projected_gravity_xy
            ),
            roll_pitch_rate=utils._get_or_init_env_buffer(
                env,
                f"{buffer_prefix}_roll_pitch_rate",
                current.roll_pitch_rate
            )
        )

    def write_to_env(
        self,
        env: ManagerBasedRLEnv,
        *,
        buffer_prefix: str
    ) -> None:
        """
        Store this state as the previous-step root/core state.
        """

        utils._set_env_buffer(env, f"{buffer_prefix}_root_z", self.root_z)
        utils._set_env_buffer(env, f"{buffer_prefix}_root_z_vel", self.root_z_vel)
        utils._set_env_buffer(
            env,
            f"{buffer_prefix}_projected_gravity_xy",
            self.projected_gravity_xy
        )
        utils._set_env_buffer(
            env,
            f"{buffer_prefix}_roll_pitch_rate",
            self.roll_pitch_rate
        )


def _reversal_excess(
    current: torch.Tensor,
    previous: torch.Tensor,
    min_magnitude: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect sign reversal and compute excess reversal magnitude.

    Returns:
        reversed_direction:
            Boolean tensor.

        excess:
            max(min(abs(current), abs(previous)) - min_magnitude, 0)
    """

    reversed_direction = current * previous < 0.0

    reversal_magnitude = torch.minimum(
        torch.abs(current),
        torch.abs(previous)
    )

    excess = torch.clamp(
        reversal_magnitude - min_magnitude,
        min=0.0
    )

    return reversed_direction, excess


def _vertical_root_chatter_l2(
    current: _RootChatterState,
    previous: _RootChatterState,
    chatter_cfg: term_cfg.RootMotionChatterCfg
) -> torch.Tensor:
    """
    Penalize small vertical bounces that rapidly reverse direction.

    Returns:
        [num_envs]
    """

    z_displacement = torch.abs(current.root_z - previous.root_z)

    velocity_reversed, reversal_excess = _reversal_excess(
        current=current.root_z_vel,
        previous=previous.root_z_vel,
        min_magnitude=chatter_cfg.min_z_reversal_speed
    )

    small_displacement = z_displacement < chatter_cfg.small_z_displacement

    chatter_active = velocity_reversed & small_displacement

    return reversal_excess.square() * chatter_active.to(dtype=current.root_z.dtype)


def _angular_root_chatter_l2(
    current: _RootChatterState,
    previous: _RootChatterState,
    chatter_cfg: term_cfg.RootMotionChatterCfg
) -> torch.Tensor:
    """
    Penalize small roll/pitch wiggles that rapidly reverse direction.

    Returns:
        [num_envs]
    """

    tilt_change = torch.linalg.norm(
        current.projected_gravity_xy - previous.projected_gravity_xy,
        dim=-1
    )

    small_tilt_change = tilt_change < chatter_cfg.small_tilt_change

    angular_reversed, angular_excess = _reversal_excess(
        current=current.roll_pitch_rate,
        previous=previous.roll_pitch_rate,
        min_magnitude=chatter_cfg.min_roll_pitch_reversal_rate
    )

    chatter_active = angular_reversed & small_tilt_change[:, None]

    penalty_per_axis = angular_excess.square() * chatter_active.to(
        dtype=current.roll_pitch_rate.dtype
    )

    return torch.sum(penalty_per_axis, dim=-1)

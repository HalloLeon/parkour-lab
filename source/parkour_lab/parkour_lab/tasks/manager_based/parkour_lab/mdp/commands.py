from isaaclab.envs import ManagerBasedRLEnv
import torch


from ._shared.runtime import _all_env_ids


_TARGET_SPEED_BUFFER = "_cmd_target_speed"
_MIN_CLEARANCE_BUFFER = "_cmd_min_clearance"
_OBSTACLE_POS_BUFFER = "_cmd_obstacle_pos"
_OBSTACLE_SIZE_BUFFER = "_cmd_obstacle_size"


def ensure_parkour_commands(
    env: ManagerBasedRLEnv,
    default_target_speed: float = 0.75,
    default_min_clearance: float = 0.25,
    default_obstacle_pos: torch.Tensor | None = None,
    default_obstacle_size: torch.Tensor | None = None
) -> None:
    """
    Ensure parkour command buffers exist.

    This may be called during ObservationManager construction, before reset
    events/curriculum have initialized per-env command values.
    """

    if default_obstacle_pos is None:
        default_obstacle_pos = (2.0, 0.0, 0.025)

    if default_obstacle_size is None:
        default_obstacle_size = (0.5, 1.8, 0.05)

    needs_target_speed = (
        not hasattr(env, _TARGET_SPEED_BUFFER)
        or getattr(env, _TARGET_SPEED_BUFFER).shape != (env.num_envs,)
        or getattr(env, _TARGET_SPEED_BUFFER).device != env.device
    )

    if needs_target_speed:
        env._cmd_target_speed = torch.full(
            (env.num_envs,),
            default_target_speed,
            device=env.device,
            dtype=torch.float32
        )

    needs_min_clearance = (
        not hasattr(env, _MIN_CLEARANCE_BUFFER)
        or getattr(env, _MIN_CLEARANCE_BUFFER).shape != (env.num_envs,)
        or getattr(env, _MIN_CLEARANCE_BUFFER).device != env.device
    )

    if needs_min_clearance:
        env._cmd_min_clearance = torch.full(
            (env.num_envs,),
            default_min_clearance,
            device=env.device,
            dtype=torch.float32
        )

    needs_obstacle_pos = (
        not hasattr(env, _OBSTACLE_POS_BUFFER)
        or getattr(env, _OBSTACLE_POS_BUFFER).shape != (env.num_envs, 3)
        or getattr(env, _OBSTACLE_POS_BUFFER).device != env.device
    )

    if needs_obstacle_pos:
        env._cmd_obstacle_pos = torch.tensor(
            default_obstacle_pos,
            device=env.device,
            dtype=torch.float32
        ).unsqueeze(0).repeat(env.num_envs, 1)

    needs_obstacle_size = (
        not hasattr(env, _OBSTACLE_SIZE_BUFFER)
        or getattr(env, _OBSTACLE_SIZE_BUFFER).shape != (env.num_envs, 3)
        or getattr(env, _OBSTACLE_SIZE_BUFFER).device != env.device
    )

    if needs_obstacle_size:
        env._cmd_obstacle_size = torch.tensor(
            default_obstacle_size,
            device=env.device,
            dtype=torch.float32
        ).unsqueeze(0).repeat(env.num_envs, 1)


def initialize_commands(
    env: ManagerBasedRLEnv,
    target_speed: float,
    min_clearance: float
) -> None:
    """
    Initialize command buffers for all environments.

    This should be called by a reset/event function before observations or
    rewards require these commands.
    """

    env._cmd_target_speed = torch.full(
        (env.num_envs,),
        target_speed,
        device=env.device,
        dtype=torch.float32
    )

    env._cmd_min_clearance = torch.full(
        (env.num_envs,),
        min_clearance,
        device=env.device,
        dtype=torch.float32
    )


def set_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_speed: torch.Tensor,
    min_clearance: torch.Tensor
) -> None:
    """
    Set per-environment parkour commands.
    """

    ensure_parkour_commands(env)

    env_ids = _all_env_ids(env, env_ids)

    if not hasattr(env, _TARGET_SPEED_BUFFER) or getattr(env, _TARGET_SPEED_BUFFER).shape != (env.num_envs,):
        env._cmd_target_speed = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32
        )

    if not hasattr(env, _MIN_CLEARANCE_BUFFER) or getattr(env, _MIN_CLEARANCE_BUFFER).shape != (env.num_envs,):
        env._cmd_min_clearance = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32
        )

    env._cmd_target_speed[env_ids] = target_speed.to(
        device=env.device,
        dtype=torch.float32
    )

    env._cmd_min_clearance[env_ids] = min_clearance.to(
        device=env.device,
        dtype=torch.float32
    )


def get_target_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Current per-environment target speed.

    Returns:
        [num_envs]
    """

    ensure_parkour_commands(env)
    return env._cmd_target_speed


def get_min_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Current per-environment minimum clearance.

    Returns:
        [num_envs]
    """

    ensure_parkour_commands(env)
    return env._cmd_min_clearance


def get_obstacle_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Current per-environment obstacle position in env-local coordinates.

    Returns:
        [num_envs, 3]
    """

    ensure_parkour_commands(env)
    return env._cmd_obstacle_pos


def get_obstacle_size(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Current per-environment obstacle size.

    Returns:
        [num_envs, 3]
    """

    ensure_parkour_commands(env)
    return env._cmd_obstacle_size

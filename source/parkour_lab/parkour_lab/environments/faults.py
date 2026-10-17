"""Episode-local single-joint loss of drive after native motor limiting."""

import math
import weakref
from dataclasses import asdict, dataclass

JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


@dataclass(frozen=True)
class FaultConfig:
    """Prescribed drive factor; onset is measured from the episode's physical start."""

    joint_name: str
    factor: float
    onset_s: float

    def __post_init__(self):
        if not isinstance(self.joint_name, str) or not self.joint_name.strip():
            raise ValueError("Require one explicit motor joint name")
        if type(self.factor) not in (int, float) or self.factor not in (0, 0.5, 1):
            raise ValueError("Drive factor must be exactly 0, 0.5 or 1")
        if (
            type(self.onset_s) not in (int, float)
            or not math.isfinite(self.onset_s)
            or self.onset_s < 0
        ):
            raise ValueError("Require a finite nonnegative onset in seconds")
        object.__setattr__(self, "factor", float(self.factor))
        object.__setattr__(self, "onset_s", float(self.onset_s))

    def as_dict(self):
        return asdict(self)


class DriveFault:
    """Use the SDK physics counter; extra writes/forward calls never advance time.

    The SDK increments its counter before writing each physics interval. Thus
    counter-reset_counter-1 is the interval's start tick. The first interval
    starting at or after the requested onset applies the loss, latched until
    that row resets. No observation, action, gain or limit is changed.
    """

    def __init__(self, config, joint_names, *, num_envs, device, physics_dt, clock):
        import torch

        if not isinstance(config, FaultConfig):
            raise ValueError("Require a validated FaultConfig")
        if (
            not joint_names
            or len(set(joint_names)) != len(joint_names)
            or config.joint_name not in joint_names
        ):
            raise ValueError("Fault joint must name exactly one resolved motor")
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("Require a positive physics interval")
        self.config, self.joint_names = config, tuple(joint_names)
        self.physics_dt, self._clock = float(physics_dt), clock
        ratio = math.nextafter(config.onset_s / physics_dt, -math.inf)
        self.onset_step = math.ceil(ratio)
        self.effective_onset_s = self.onset_step * self.physics_dt
        self.joint_index = self.joint_names.index(config.joint_name)
        self.reset_steps = torch.full(
            (num_envs,), self._counter(), dtype=torch.long, device=device
        )
        self.fault_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.drive_factor = torch.ones(
            num_envs, len(joint_names), dtype=torch.float32, device=device
        )

    def _counter(self):
        step = self._clock()
        if type(step) is not int or step < 0:
            raise ValueError("Require a nonnegative native physics counter")
        return step

    def reset(self, env_ids):
        import torch

        ids = torch.as_tensor(env_ids, device=self.reset_steps.device)
        if (
            ids.ndim != 1
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.unique().numel() != ids.numel()
            or ((ids < 0) | (ids >= len(self.reset_steps))).any()
        ):
            raise ValueError("Require distinct valid integer reset rows")
        self.reset_steps[ids] = self._counter()
        self.fault_active[ids] = False
        self.drive_factor[ids] = 1.0

    def apply(self, limited_effort, columns):
        import torch

        elapsed = self._counter() - self.reset_steps - 1
        self.fault_active |= elapsed >= self.onset_step
        self.drive_factor[:, self.joint_index] = torch.where(
            self.fault_active, self.config.factor, 1.0
        )
        return limited_effort * self.drive_factor[:, columns]


def configure_fault(env):
    """Bind evaluation-only configuration to the resolved native motor groups."""
    configuration = getattr(env.cfg, "benchmark_fault", None)
    if configuration is None:
        return
    if getattr(env.cfg, "benchmark_training", True):
        raise ValueError("Timed drive loss is evaluation-only")
    if hasattr(env, "parkour_fault"):
        raise RuntimeError("Drive loss is already configured")
    from .dynamics import RandomizedDCMotor

    robot = env.scene["robot"]
    motors = tuple(robot.actuators.values())
    covered = [joint for motor in motors for joint in motor.joint_names]
    if sorted(covered) != sorted(robot.joint_names) or any(
        not isinstance(motor, RandomizedDCMotor) for motor in motors
    ):
        raise ValueError("Drive loss requires complete randomized DC motor binding")
    owner = weakref.ref(env)
    fault = DriveFault(
        FaultConfig(**configuration),
        robot.joint_names,
        num_envs=env.num_envs,
        device=env.device,
        physics_dt=env.physics_dt,
        clock=lambda: owner()._sim_step_counter,
    )
    for motor in motors:
        motor.drive_fault = fault
        motor._drive_fault_columns = [
            robot.joint_names.index(name) for name in motor.joint_names
        ]
    env.parkour_fault = fault


def reset_fault(env, env_ids):
    fault = getattr(env, "parkour_fault", None)
    if fault is not None:
        fault.reset(env_ids)


def fault_state(env):
    """Owned evaluator readback in leg-major ``JOINT_NAMES`` order.

    Actuator/state storage retains the native binding order. Only these private
    evaluator arrays are reordered; actor actions and joint sensors are untouched.
    """
    fault = getattr(env, "parkour_fault", None)
    if fault is None:
        return {}
    import torch

    limited, applied = torch.zeros_like(fault.drive_factor), torch.zeros_like(
        fault.drive_factor
    )
    for motor in env.scene["robot"].actuators.values():
        columns = motor._drive_fault_columns
        limited[:, columns] = motor.limited_effort
        applied[:, columns] = motor.applied_effort
    order = [fault.joint_names.index(name) for name in JOINT_NAMES]
    return dict(
        fault_active=fault.fault_active.detach().clone(),
        drive_factor=fault.drive_factor[:, order].detach().clone(),
        limited_effort=limited[:, order].detach().clone(),
        applied_effort=applied[:, order].detach().clone(),
    )

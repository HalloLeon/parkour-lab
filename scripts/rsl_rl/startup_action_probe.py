"""Bounded action replay and observational first-action physics telemetry.

No simulator imports, pickle loading, extra stepping, resets, or sensor updates.
PhysX getter/frame conventions follow the locally audited Isaac Lab v2.3.2.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

try:
    from .failure_trace_report import validate_failure_trace
except ImportError:
    from failure_trace_report import validate_failure_trace


INITIAL_ATOL = 2e-6
TARGET_ATOL = 1e-6
MAX_REPLAY_STEPS = 10
INITIAL_FIELDS = (
    "root_position_env_m",
    "root_orientation_wxyz",
    "linear_velocity_body_m_s",
    "linear_velocity_w_m_s",
    "angular_velocity_body_rad_s",
    "angular_velocity_w_rad_s",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_default_position_rad",
    "joint_position_target_rad",
    "joint_soft_position_limits_rad",
    "safe_joint_target_limits_rad",
    "foot_position_env_m",
    "foot_linear_velocity_w_m_s",
)
TARGET_FIELDS = (
    "environment_action",
    "delayed_raw_action",
    "affine_joint_target_rad",
    "configured_clip_joint_target_rad",
    "processed_joint_target_rad",
    "joint_position_target_rad",
)
RUNTIME_REQUIRED = (
    "checkpoint_sha256",
    "teacher_interface_sha256",
    "policy_mode",
    "reset_profile",
    "seed",
    "num_envs",
    "step_dt_s",
)
RUNTIME_OPTIONAL = ("environment_physics", "kit_args", "action_clip", "action_noise")


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if type(value) in (int, float):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return value is None or type(value) in (str, bool)


def _close(actual, expected, label, atol):
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: shape mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _close(left, right, f"{label}[{index}]", atol)
    elif type(expected) in (int, float):
        if (
            type(actual) not in (int, float)
            or not _finite(actual)
            or abs(actual - expected) > atol
        ):
            raise ValueError(f"{label}: mismatch (absolute tolerance {atol:g})")
    elif actual != expected:
        raise ValueError(f"{label}: mismatch")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def _copy_tensor(value, *, batch=True):
    """Own the PhysX buffer before another getter/physics step can overwrite it."""
    value = value.detach().clone()
    if batch:
        if value.ndim < 1 or value.shape[0] != 1:
            raise ValueError("Physics probe requires exactly one environment")
        value = value[0]
    result = value.cpu().tolist()
    if not _finite(result):
        raise ValueError("Nonfinite physics telemetry")
    return result


def _unavailable(error):
    return {"status": "UNAVAILABLE", "reason": f"{type(error).__name__}: {error}"}


def _checked_tensor(value, shape, label):
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{label}: expected tensor shape {shape}, got {tuple(value.shape)}"
        )
    return _copy_tensor(value)


def _optional_tensor(getter, shape, label):
    try:
        value = getter()
    except (AttributeError, RuntimeError, NotImplementedError) as error:
        return _unavailable(error)
    # A present but malformed/nonfinite measurement is not 'unavailable'.
    return _checked_tensor(value, shape, label)


def _action_order(value, joint_ids, raw_count, *, suffix=()):
    shape = (1, raw_count, *suffix)
    if tuple(value.shape) != shape:
        raise ValueError(
            f"Raw joint tensor: expected shape {shape}, got {tuple(value.shape)}"
        )
    return value[:, joint_ids]


class StartupActionProbe:
    """Replay at most ten source actions, failing closed on contract mismatch."""

    def __init__(self, source_path, max_steps=MAX_REPLAY_STEPS, *, solver_probe=None):
        if solver_probe not in (None, "tgs", "pgs"):
            raise ValueError("Solver probe must be explicitly tgs or pgs")
        self.solver_probe = solver_probe
        if type(max_steps) is not int or not 1 <= max_steps <= MAX_REPLAY_STEPS:
            raise ValueError("Replay prefix must contain 1 to 10 actions")
        self.source_path = Path(source_path).resolve()
        raw = self.source_path.read_bytes()
        self.source = json.loads(raw, object_pairs_hook=_unique_object)
        validate_failure_trace(self.source)
        meta = self.source["metadata"]
        if any(
            key in meta
            for key in (
                "action_probe",
                "action_replay",
                "action_replay_probe",
                "startup_action_probe",
                "replay",
            )
        ) or any("replayed_action" in sample for sample in self.source["samples"]):
            raise ValueError("Nested action replay is not an ordinary policy source")
        if meta.get("action_source", "policy") not in ("policy", "policy_action"):
            raise ValueError("Replay requires an ordinary policy-action source")
        if meta["policy_mode"] not in ("history_mean", "privileged_mean"):
            raise ValueError("Replay requires a deterministic mean-policy source")
        if len(self.source["samples"]) < max_steps or any(
            sample["post"]["done"] for sample in self.source["samples"][:max_steps]
        ):
            raise ValueError(
                "Source lacks the requested complete nonterminal action prefix"
            )
        self.max_steps = max_steps
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        self.physics_substeps = []
        self.physical_metadata = {}
        self._runtime_validated = False
        self._initial_validated = False
        self._next_step = 0
        self._awaiting_post = None
        self._physics_captured = False
        # The base validator checks the common schema; these extra fields are
        # necessary for replay's stronger initial-state contract.
        initial = self.source["samples"][0]["pre"]["state"]
        for key in INITIAL_FIELDS:
            if key not in initial or not isinstance(initial[key], list):
                raise ValueError(f"Source initial state lacks {key}")
        if len(initial["joint_default_position_rad"]) != 12 or not all(
            type(value) in (int, float)
            for value in initial["joint_default_position_rad"]
        ):
            raise ValueError("Source joint_default_position_rad must have 12 entries")

    def metadata(self):
        meta = self.source["metadata"]
        return {
            "kind": "startup_action_replay_probe",
            "action_source": "recorded_policy_action",
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_checkpoint_sha256": meta["checkpoint_sha256"],
            "source_teacher_interface_sha256": meta["teacher_interface_sha256"],
            "prefix_steps": self.max_steps,
            "initial_state_abs_tolerance": INITIAL_ATOL,
            "target_abs_tolerance": TARGET_ATOL,
            "runtime_validated": self._runtime_validated,
            "initial_state_validated": self._initial_validated,
            "limitations": [
                "Open-loop replay is a diagnostic intervention, not policy evaluation.",
                "Course and runtime interface may differ intentionally; checkpoint contract must match.",
                "Initial telemetry does not expose every simulator/contact-solver internal state.",
                "Contact force vectors are not a measurement of contact-point slip or a friction test.",
            ],
        }

    def validate_runtime(self, metadata):
        self._runtime_validated = False
        source = self.source["metadata"]
        if not isinstance(metadata, dict) or not _finite(metadata):
            raise ValueError("Runtime metadata must be finite JSON data")
        if self.solver_probe is None and "solver_probe" in metadata:
            raise ValueError("Solver intervention requires explicit replay opt-in")
        if self.solver_probe is not None:
            try:
                from .startup_solver_probe import validate_solver_environment_physics
            except ImportError:
                from startup_solver_probe import validate_solver_environment_physics
            solver_meta = metadata.get("solver_probe", {})
            if solver_meta.get("requested_solver") != self.solver_probe.upper():
                raise ValueError("Requested replay solver does not match readback")
            validate_solver_environment_physics(
                source.get("environment_physics"),
                metadata.get("environment_physics"),
                solver_meta,
            )
        for key in RUNTIME_REQUIRED + RUNTIME_OPTIONAL:
            if key == "environment_physics" and self.solver_probe is not None:
                continue  # The explicit single-field exception was checked above.
            if key in source or key in RUNTIME_REQUIRED:
                if key not in metadata or metadata[key] != source[key]:
                    raise ValueError(f"Runtime metadata.{key}: missing or mismatch")
        for key in ("joint_names", "foot_names", "contact_body_names"):
            expected = source["capture_metadata"].get(key)
            if (
                expected is not None
                and metadata.get("capture_metadata", {}).get(key) != expected
            ):
                raise ValueError(f"Runtime capture_metadata.{key}: missing or mismatch")
        source_action = source.get("runtime_teacher_interface", {}).get("action")
        if source_action is not None and (
            metadata.get("runtime_teacher_interface", {}).get("action") != source_action
        ):
            raise ValueError(
                "Runtime teacher interface action mapping: missing or mismatch"
            )
        self._runtime_validated = True

    def action(self, step, policy_action, pre):
        if not self._runtime_validated:
            raise ValueError("Validate runtime metadata before replay")
        if type(step) is not int or step != self._next_step or step >= self.max_steps:
            raise ValueError(
                "Replay action steps must be consecutive within the bounded prefix"
            )
        if self._awaiting_post is not None:
            raise ValueError(
                "Validate the preceding replay target before another action"
            )
        if tuple(policy_action.shape) != (1, 12):
            raise ValueError("Replay requires policy action shape (1, 12)")
        if not policy_action.is_floating_point():
            raise ValueError("Replay requires a floating-point policy action")
        _copy_tensor(policy_action)
        if step == 0:
            expected = self.source["samples"][0]["pre"]
            for key in INITIAL_FIELDS:
                _close(
                    pre["state"].get(key),
                    expected["state"][key],
                    f"Initial {key}",
                    INITIAL_ATOL,
                )
            _close(
                pre["observations"].get("dynamics"),
                expected["observations"]["dynamics"],
                "Initial dynamics",
                INITIAL_ATOL,
            )
            self._initial_validated = True
        # Preserve the wrapper's action mapping and dtype/device. Never modify
        # the actual policy output, which the collector records separately.
        selected = policy_action.new_tensor(
            [self.source["samples"][step]["policy_action"]]
        )
        _close(
            _copy_tensor(selected),
            self.source["samples"][step]["policy_action"],
            "Replay action dtype conversion",
            TARGET_ATOL,
        )
        self._awaiting_post = step
        return selected

    def validate_post(self, step, state):
        if self._awaiting_post != step:
            raise ValueError("No matching replay action awaits target validation")
        expected = self.source["samples"][step]["post"]["state"]
        for key in TARGET_FIELDS:
            _close(
                state.get(key), expected[key], f"Replay step {step} {key}", TARGET_ATOL
            )
        _close(
            state.get("joint_position_target_rad"),
            state.get("processed_joint_target_rad"),
            "Actual versus processed target",
            TARGET_ATOL,
        )
        self._awaiting_post = None
        self._next_step += 1

    def _generalized_dynamics(self, asset):
        """Raw floating-base arrays; never drop the root or permute just one axis.

        Getter names/shapes follow Omni Physics 107.3's ArticulationView API.
        No kinematic refresh or simulator step is used to obtain these arrays.
        """
        names = list(asset.joint_names)
        size = len(names) + 6
        view = asset.root_physx_view
        result = {
            "raw_dof_names": names,
            "floating_base_root_components": 6,
            "ordering": "Raw PhysX generalized order: root six components then raw_dof_names. No action-order permutation or root-frame conversion.",
        }
        for name, getter, shape in (
            ("mass_matrix", "get_generalized_mass_matrices", (1, size, size)),
            ("gravity_compensation", "get_gravity_compensation_forces", (1, size)),
            (
                "coriolis_centrifugal_compensation",
                "get_coriolis_and_centrifugal_compensation_forces",
                (1, size),
            ),
        ):
            result[name] = _optional_tensor(
                lambda method=getter: getattr(view, method)(), shape, name
            )
        return result

    def _physics_state(self, env, asset, action, joint_ids, *, generalized=False):
        view = asset.root_physx_view
        result = {
            "joint_position_rad": _copy_tensor(view.get_dof_positions()[:, joint_ids]),
            "joint_velocity_rad_s": _copy_tensor(
                view.get_dof_velocities()[:, joint_ids]
            ),
            "joint_position_target_rad": _copy_tensor(
                asset.data.joint_pos_target[:, joint_ids]
            ),
            "processed_joint_target_rad": _copy_tensor(action.processed_actions),
            "joint_computed_torque_nm": _copy_tensor(
                asset.data.computed_torque[:, joint_ids]
            ),
            "joint_applied_torque_nm": _copy_tensor(
                asset.data.applied_torque[:, joint_ids]
            ),
        }
        for name, getter in (
            (
                "joint_submitted_effort_nm",
                lambda: _action_order(
                    asset._joint_effort_target_sim, joint_ids, len(asset.joint_names)
                ),
            ),
            (
                "joint_physx_actuation_force_nm",
                lambda: _action_order(
                    view.get_dof_actuation_forces(), joint_ids, len(asset.joint_names)
                ),
            ),
        ):
            result[name] = _optional_tensor(getter, (1, 12), name)
        if generalized:
            result["generalized_dynamics"] = self._generalized_dynamics(asset)
        for key, getter in (
            ("root_transform_w_xyzw", "get_root_transforms"),
            ("root_com_velocity_w_m_s_rad_s", "get_root_velocities"),
            ("link_transform_w_xyzw", "get_link_transforms"),
            ("link_com_velocity_w_m_s_rad_s", "get_link_velocities"),
        ):
            try:
                result[key] = _copy_tensor(getattr(view, getter)())
            except (AttributeError, RuntimeError, NotImplementedError) as error:
                result[key] = _unavailable(error)
        contacts = {}
        for name in ("feet_contact", "undesired_contact", "chassis_contact"):
            try:
                sensor = env.scene[name]
                forces = sensor.contact_physx_view.get_net_contact_forces(
                    dt=env.physics_dt
                )
                forces = forces.reshape(1, len(sensor.body_names), 3)
                contacts[name] = {
                    "body_names": list(sensor.body_names),
                    "net_force_w_n": _copy_tensor(forces),
                }
            except (
                KeyError,
                AttributeError,
                RuntimeError,
                NotImplementedError,
            ) as error:
                contacts[name] = _unavailable(error)
        result["contacts"] = contacts
        return result

    def _physical_settings(self, env, asset, action, joint_ids):
        properties = {}
        # All getter names below are used by Isaac Lab v2.3.2 itself. Joint
        # properties are reordered into the exact action order, not USD order.
        for key, getter, joints in (
            ("body_mass_kg", "get_masses", False),
            ("body_inertia_kg_m2", "get_inertias", False),
            ("body_com_pose", "get_coms", False),
            (
                "shape_material_static_dynamic_restitution",
                "get_material_properties",
                False,
            ),
            ("joint_armature", "get_dof_armatures", True),
            ("joint_physx_stiffness", "get_dof_stiffnesses", True),
            ("joint_physx_damping", "get_dof_dampings", True),
            ("joint_max_velocity_rad_s", "get_dof_max_velocities", True),
            ("joint_max_force_nm", "get_dof_max_forces", True),
            ("joint_position_limits_rad", "get_dof_limits", True),
        ):
            try:
                values = getattr(asset.root_physx_view, getter)()
                properties[key] = _copy_tensor(
                    values[:, joint_ids] if joints else values
                )
            except (AttributeError, RuntimeError, NotImplementedError) as error:
                properties[key] = _unavailable(error)
        for name in ("stiffness", "damping"):
            try:
                values = getattr(asset.data, f"default_joint_{name}").detach().clone()
                for actuator in asset.actuators.values():
                    values[:, actuator.joint_indices] = getattr(actuator, name)
                properties[f"joint_operative_{name}"] = _copy_tensor(
                    values[:, joint_ids]
                )
            except (AttributeError, RuntimeError, NotImplementedError) as error:
                properties[f"joint_operative_{name}"] = _unavailable(error)
        properties["joint_friction_static_dynamic_viscous"] = _optional_tensor(
            lambda: _action_order(
                asset.root_physx_view.get_dof_friction_properties(),
                joint_ids,
                len(asset.joint_names),
                suffix=(3,),
            ),
            (1, 12, 3),
            "joint_friction_static_dynamic_viscous",
        )
        properties["joint_legacy_friction_coefficient"] = _optional_tensor(
            lambda: _action_order(
                asset.root_physx_view.get_dof_friction_coefficients(),
                joint_ids,
                len(asset.joint_names),
            ),
            (1, 12),
            "joint_legacy_friction_coefficient",
        )
        return {
            "joint_names": list(action._joint_names),
            "body_names": list(asset.body_names),
            "foot_names": list(
                self.source["metadata"]["capture_metadata"]["foot_names"]
            ),
            "environment_origin_w_m": _copy_tensor(env.scene.env_origins),
            "physics_dt_s": env.physics_dt,
            "initial_authoritative_state": self._physics_state(
                env, asset, action, joint_ids, generalized=True
            ),
            "properties": properties,
            "unavailable": {
                "contact_offsets": "UNKNOWN: no verified per-shape getter used"
            },
            "frames": {
                "transforms": "PhysX raw world link pose xyz + quaternion xyzw; body_names order",
                "velocities": "PhysX world COM linear then angular velocity; not link-origin/contact-point velocity",
                "body_com_pose": "PhysX local COM pose relative to each body link: xyz + quaternion xyzw",
                "contacts": "Direct PhysX net force query at physics_dt; sensor body_names order; not friction/slip measurement",
                "torques": "Explicit actuator computed/applied caches written immediately before the physics step",
                "joint_submitted_effort_nm": "Isaac Lab _joint_effort_target_sim submission buffer in action joint order; not a measured net joint torque",
                "joint_physx_actuation_force_nm": "PhysX get_dof_actuation_forces backend command in action joint order; excludes implicit drives and constraint/contact forces; not net joint torque",
                "joint_friction_static_dynamic_viscous": "Per action joint: static friction effort (Nm), dynamic friction effort (Nm), viscous friction coefficient; not surface friction",
                "joint_legacy_friction_coefficient": "Deprecated backend coefficient in action joint order; independent fallback channel when the new friction parameters are zero; not surface friction",
                "generalized_dynamics": "Raw floating-base mass and compensation arrays at initial capture and first pre-physics only; optional availability, no refresh/step and no causal verdict",
                "targets": "Articulation target buffer and action processed target; explicit PD uses these, not a PhysX position drive",
            },
        }

    @contextmanager
    def capture_physics(self, env, step):
        """Observe sim.step calls already made by the first control step only."""
        if step != 0:
            yield
            return
        if self._physics_captured:
            raise ValueError("First-action physics capture may only run once")
        if not self._initial_validated or self._awaiting_post != 0:
            raise ValueError(
                "Select the validated first replay action before physics capture"
            )
        env = getattr(env, "unwrapped", env)
        if env.num_envs != 1:
            raise ValueError("Physics probe requires exactly one environment")
        count = env.step_dt / env.physics_dt
        if not math.isfinite(count) or not math.isclose(count, 4.0, abs_tol=1e-9):
            raise ValueError(
                "First-action probe requires exactly four physics substeps"
            )
        asset = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        if (
            list(action._joint_names)
            != self.source["metadata"]["capture_metadata"]["joint_names"]
        ):
            raise ValueError("Live action joint order differs from source")
        joint_ids = [asset.joint_names.index(name) for name in action._joint_names]
        self.physical_metadata = self._physical_settings(env, asset, action, joint_ids)
        sim = env.sim
        original = sim.step
        had_override = "step" in vars(sim)
        prior_override = vars(sim).get("step")
        self._physics_captured = True

        def observe_step(*args, **kwargs):
            index = len(self.physics_substeps)
            if index >= 4:
                raise ValueError("Unexpected extra physics step in replay capture")
            record = {
                "control_step": 0,
                "physics_substep": index,
                "physics_dt_s": env.physics_dt,
                "time_before_s": index * env.physics_dt,
                "time_after_s": (index + 1) * env.physics_dt,
                "pre": self._physics_state(
                    env, asset, action, joint_ids, generalized=index == 0
                ),
            }
            self.physics_substeps.append(record)
            result = original(*args, **kwargs)
            record["post"] = self._physics_state(env, asset, action, joint_ids)
            return result

        sim.step = observe_step
        try:
            yield
            if len(self.physics_substeps) != 4:
                raise ValueError(
                    "First action did not execute exactly four physics substeps"
                )
        finally:
            if had_override:
                sim.step = prior_override
            else:
                del sim.step

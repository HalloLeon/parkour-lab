"""Pinned DC torque arithmetic and deterministic native-API dynamics boundaries."""

import ast
import importlib.util
import unittest
from contextlib import contextmanager
from copy import deepcopy
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.control.motor_contract import make_motor_contract, verify_runtime_motor
from parkour_lab.environments.faults import DriveFault, FaultConfig
from parkour_lab.runtime.motor import _runtime_motor_binding
from tests.support import ROOT
from tests.support.isaaclab import modules, pinned_source_root
from tests.support.native import make_native_env


@contextmanager
def native_dynamics():
    """Execute actual SDK actuator code; only unavailable simulator imports are doubled."""
    root = pinned_source_root()
    replacements = {}
    for name in (
        "isaaclab",
        "isaaclab.actuators",
        "isaaclab.utils",
        "isaaclab.utils.string",
    ):
        replacements[name] = ModuleType(name)
        replacements[name].__path__ = []
    replacements["isaaclab.utils"].DelayBuffer = object
    replacements["isaaclab.utils"].LinearInterpolation = object
    replacements["isaaclab.utils.types"] = NS(ArticulationActions=NS)
    replacements["isaaclab.managers"] = NS(EventTermCfg=NS)
    paths = {
        "isaaclab.actuators.actuator_base": root / "actuators/actuator_base.py",
        "isaaclab.actuators.actuator_pd": root / "actuators/actuator_pd.py",
        "parkour_lab.environments.dynamics": ROOT
        / "source/parkour_lab/parkour_lab/environments/dynamics.py",
    }
    specs = {
        name: importlib.util.spec_from_file_location(name, path)
        for name, path in paths.items()
    }
    replacements.update(
        {name: importlib.util.module_from_spec(spec) for name, spec in specs.items()}
    )
    with modules(replacements):
        for name, spec in specs.items():
            spec.loader.exec_module(replacements[name])
            if name == "isaaclab.actuators.actuator_pd":
                replacements["isaaclab.actuators"].DCMotor = replacements[name].DCMotor
        module = replacements["parkour_lab.environments.dynamics"]
        with patch.object(
            module,
            "_material_combine_modes",
            return_value={"/robot/material": "average"},
        ):
            yield module


def actuator(module, count=4):
    cfg = NS(
        stiffness=25.0,
        damping=0.5,
        effort_limit=23.5,
        velocity_limit=30.0,
        effort_limit_sim=1e9,
        velocity_limit_sim=1000.0,
        saturation_effort=23.5,
        armature=0.0,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    return module.RandomizedDCMotor(
        cfg, [f"j{i}" for i in range(12)], slice(None), count, "cpu"
    )


def environment(module, *, dynamics="randomized", count=4):
    motor = actuator(module, count)
    values = dict(
        masses=torch.tensor([[6.0, 1.0]]).repeat(count, 1),
        inertias=torch.eye(3).flatten().repeat(count, 2, 1),
        coms=torch.zeros(count, 2, 7),
        materials=torch.zeros(count, 8, 3),
        dof_limits=torch.tensor([-3.0, 3.0]).repeat(count, 12, 1),
    )
    values["coms"][..., 6] = 1.0  # PhysX quaternions use xyzw order.
    view = NS()
    for name in values:
        setattr(
            view,
            "get_" + ("material_properties" if name == "materials" else name),
            lambda n=name: values[n].clone(),
        )
    for name in ("masses", "inertias", "materials"):

        def write(value, ids, n=name):
            values[n][ids] = value[ids]

        setattr(
            view,
            "set_" + ("material_properties" if name == "materials" else name),
            write,
        )
    root = torch.zeros(count, 13)
    root[:, 2], root[:, 3] = 0.4, 1.0
    data = NS(
        default_mass=values["masses"].clone(),
        default_inertia=values["inertias"].clone(),
        default_root_state=root,
        default_joint_pos=torch.zeros(count, 12),
        soft_joint_pos_limits=values["dof_limits"].clone(),
    )
    robot = NS(
        data=data,
        joint_names=motor.joint_names,
        body_names=["base", "leg"],
        actuators={"legs": motor},
        root_physx_view=view,
    )
    writes = {}
    robot.write_root_pose_to_sim = lambda value, env_ids: writes.update(
        pose=value.clone(), ids=env_ids.clone()
    )
    robot.write_root_velocity_to_sim = lambda value, env_ids: writes.update(
        velocity=value.clone()
    )
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: writes.update(
        joint_pos=pos.clone(), joint_vel=vel.clone()
    )

    class Scene(dict):
        env_origins = (
            torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
        )

    env = NS(
        num_envs=count,
        device="cpu",
        scene=Scene(robot=robot),
        cfg=NS(
            parkour_task=dict(num_envs=count, seed=123, dynamics=dynamics),
            parkour_evaluation=False,
        ),
    )
    return env, motor, values, writes


def sdk_function(relative, name, namespace, *, owner=None):
    """Run one unmodified pinned SDK function without importing native plugins."""
    source = (pinned_source_root() / relative).read_text()
    tree = ast.parse(source)
    body = (
        tree.body
        if owner is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    )
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(
        "from __future__ import annotations\n" + ast.get_source_segment(source, node),
        namespace,
    )
    return namespace[name]


class DynamicsTests(unittest.TestCase):
    def test_drive_loss_follows_native_limit_at_every_speed_and_torque_sign(self):
        with native_dynamics() as module:
            healthy = actuator(module, count=5)
            position = torch.zeros(5, 12)
            velocity = torch.tensor([0.0, 15.0, 29.0, 40.0, -40.0])[:, None].expand(
                -1, 12
            )
            clock = NS(step=0)
            for factor in (1, 0.5, 0):
                for request in (0.1, 10.0, -10.0):
                    with self.subTest(factor=factor, request=request):
                        motor = actuator(module, count=5)
                        motor.motor_strength.fill_(1.1)
                        healthy.motor_strength.fill_(1.1)
                        motor.drive_fault = DriveFault(
                            FaultConfig("j7", factor, 0.01),
                            motor.joint_names,
                            num_envs=5,
                            device="cpu",
                            physics_dt=0.005,
                            clock=lambda: clock.step,
                        )
                        motor._drive_fault_columns = list(range(12))

                        def command():
                            return NS(
                                joint_positions=torch.full_like(position, request),
                                joint_velocities=velocity.clone(),
                                joint_efforts=torch.zeros_like(position),
                            )

                        nominal = healthy.compute(
                            command(), position, velocity
                        ).joint_efforts.clone()
                        clock.step = motor.drive_fault.reset_steps[0].item() + 2
                        before = motor.compute(command(), position, velocity)
                        self.assertTrue(torch.equal(before.joint_efforts, nominal))
                        clock.step += 1
                        delivered = motor.compute(command(), position, velocity)
                        expected = nominal.clone()
                        expected[:, 7] *= factor
                        self.assertTrue(torch.equal(motor.limited_effort, nominal))
                        self.assertTrue(torch.equal(delivered.joint_efforts, expected))
                        self.assertTrue(
                            torch.equal(motor.computed_effort, healthy.computed_effort)
                        )
                        self.assertIs(delivered.joint_efforts, motor.applied_effort)
                        self.assertTrue(torch.equal(motor.stiffness, healthy.stiffness))
                        self.assertTrue(
                            torch.equal(motor.effort_limit, healthy.effort_limit)
                        )
                        self.assertTrue(
                            torch.equal(motor.velocity_limit, healthy.velocity_limit)
                        )

    def test_sdk_publishes_final_drive_to_torque_buffers_and_reward(self):
        with native_dynamics() as module:
            motor = actuator(module, count=2)
            motor.drive_fault = DriveFault(
                FaultConfig("j7", 0, 0),
                motor.joint_names,
                num_envs=2,
                device="cpu",
                physics_dt=0.005,
                clock=lambda: 0,
            )
            motor._drive_fault_columns = list(range(12))
            motor.drive_fault._clock = lambda: 1
            target = torch.full((2, 12), 10.0)
            data = NS(
                joint_pos_target=target.clone(),
                joint_vel_target=torch.zeros(2, 12),
                joint_effort_target=torch.zeros(2, 12),
                joint_pos=torch.zeros(2, 12),
                joint_vel=torch.zeros(2, 12),
                computed_torque=torch.zeros(2, 12),
                applied_torque=torch.zeros(2, 12),
                soft_joint_vel_limits=torch.zeros(2, 12),
            )
            robot = NS(
                actuators={"legs": motor},
                _data=data,
                _joint_effort_target_sim=torch.zeros(2, 12),
                _joint_pos_target_sim=torch.zeros(2, 12),
                _joint_vel_target_sim=torch.zeros(2, 12),
            )
            apply = sdk_function(
                "assets/articulation/articulation.py",
                "_apply_actuator_model",
                dict(ArticulationActions=NS),
                owner="Articulation",
            )
            apply(robot)
            self.assertTrue(
                torch.equal(data.applied_torque, robot._joint_effort_target_sim)
            )
            self.assertTrue(torch.equal(data.applied_torque, motor.applied_effort))
            self.assertTrue((data.applied_torque[:, 7] == 0).all())
            self.assertTrue((data.computed_torque[:, 7] > 0).all())
            self.assertTrue(torch.equal(data.joint_pos_target, target))
            reward = sdk_function(
                "envs/mdp/rewards.py",
                "joint_torques_l2",
                dict(
                    torch=torch,
                    SceneEntityCfg=lambda name: NS(name=name, joint_ids=slice(None)),
                ),
            )
            actual = reward(NS(scene={"robot": NS(data=data)}))
            self.assertTrue(torch.equal(actual, data.applied_torque.square().sum(-1)))
            self.assertTrue((actual < data.computed_torque.square().sum(-1)).all())

    def test_rough_shared_reset_uses_changed_supported_origin_and_fixed_dynamics(self):
        from parkour_lab.runtime.native import causal_sensor_noise

        with native_dynamics() as dynamics:
            env, motor, physical, writes = environment(dynamics)
            env.cfg.parkour_task = dict(num_envs=4, seed=123, dynamics="randomized")
            dynamics.initialize_dynamics(env)
            dynamics.reset_state(env, torch.arange(4))
            self.assertTrue(
                torch.equal(writes["pose"][:, 2], env.scene.env_origins[:, 2] + 0.4)
            )
            fixed_gains = motor.stiffness.clone()
            fixed_mass = physical["masses"].clone()
            env.scene.env_origins[0] = torch.tensor([8.0, 16.0, -1.61])
            dynamics.reset_state(env, torch.tensor([0]))
            self.assertEqual(writes["ids"].tolist(), [0])
            self.assertAlmostEqual(float(writes["pose"][0, 2]), -1.21, places=6)
            self.assertTrue(torch.equal(writes["velocity"], torch.zeros(1, 6)))
            self.assertTrue(torch.equal(writes["joint_vel"], torch.zeros(1, 12)))
            self.assertTrue((writes["joint_pos"].abs() <= 0.05).all())
            self.assertTrue(torch.equal(motor.stiffness, fixed_gains))
            self.assertTrue(torch.equal(physical["masses"], fixed_mass))
            dynamics.dynamics_report(env)
            noise = causal_sensor_noise(env)
            self.assertEqual(noise.shape, (4, 30))
            self.assertTrue(
                (
                    noise.abs()
                    <= torch.tensor([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)
                ).all()
            )

    def check_connected_start(self, support):
        with native_dynamics() as module:
            env, _, _, writes = environment(module, dynamics="nominal")
            env.scene.env_origins.zero_()
            env.scene.terrain = NS(
                start_position=[3.0, 7.0, support], start_heading=1.2
            )
            module.initialize_dynamics(env)
            module.reset_state(env, torch.arange(env.num_envs))
            self.assertTrue(torch.equal(env.scene.env_origins, torch.zeros(4, 3)))
            self.assertTrue(
                torch.allclose(writes["pose"][:, 2], torch.full((4,), support + 0.4))
            )
            self.assertTrue(torch.equal(writes["velocity"], torch.zeros(4, 6)))
            self.assertTrue(torch.equal(writes["joint_vel"], torch.zeros(4, 12)))
            data = env.scene["robot"].data
            data.root_pos_w, data.root_quat_w = (
                writes["pose"][:, :3],
                writes["pose"][:, 3:],
            )
            data.root_state_w = torch.cat([writes["pose"], writes["velocity"]], dim=1)
            data.joint_pos, data.joint_vel = writes["joint_pos"], writes["joint_vel"]
            module.start_report(env)
            for i, attempt in enumerate(
                env.parkour_randomization.manifest()["attempts"]
            ):
                start = attempt["start"]
                self.assertEqual(start["nominal_heading"], 1.2)
                self.assertTrue(
                    torch.allclose(
                        data.root_pos_w[i, :2],
                        torch.tensor(start["start_xy"]) + torch.tensor([3.0, 7.0]),
                    )
                )
            data.root_pos_w[0, 2] -= 0.1
            with self.assertRaisesRegex(ValueError, "first start"):
                module.start_report(env)

    def test_inertia_setter_can_represent_principal_axes_differently(self):
        # A sign change and a principal-axis permutation can both preserve the
        # body-frame tensor; comparing quaternions (even modulo sign) is wrong.
        for rotation in ((0, 0, 0, -1), (0, 0, 2**-0.5, 2**-0.5)):
            with self.subTest(rotation=rotation), native_dynamics() as module:
                env, _, values, _ = environment(module)
                robot = env.scene["robot"]
                tensor = torch.diag(torch.tensor([0.1, 0.2, 0.3])).flatten()
                values["inertias"][:] = tensor
                robot.data.default_inertia.copy_(values["inertias"])
                setter = robot.root_physx_view.set_inertias

                def set_inertias(data, ids):
                    setter(data, ids)
                    values["coms"][..., 3:] = torch.tensor(rotation)

                with patch.object(robot.root_physx_view, "set_inertias", set_inertias):
                    module.initialize_dynamics(env)
                report = module.dynamics_report(env)
                self.assertNotEqual(
                    report["requested_physics"]["coms"],
                    report["physical_readback"]["coms"],
                )
                # The accepted representation must still remain fixed afterward.
                values["coms"][..., 3:].neg_()
                with self.assertRaisesRegex(ValueError, "after initialization: coms"):
                    module.dynamics_report(env)

    def test_startup_rejects_physical_mismatch_and_identifies_field(self):
        for key, index in (
            ("masses", (0, 0)),
            ("inertias", (0, 0, 0)),
            ("coms", (0, 0, 0)),
            ("materials", (0, 0, 0)),
            ("joint_limits", (0, 0, 0)),
        ):
            with self.subTest(key=key), native_dynamics() as module:
                env, _, values, _ = environment(module)
                view = env.scene["robot"].root_physx_view
                setter = view.set_material_properties

                def set_materials(data, ids):
                    setter(data, ids)
                    values["dof_limits" if key == "joint_limits" else key][
                        index
                    ] += 0.01

                with (
                    patch.object(view, "set_material_properties", set_materials),
                    self.assertRaisesRegex(
                        ValueError, f"readback mismatch for {key}.*expected .*got"
                    ),
                ):
                    module.initialize_dynamics(env)

    def test_startup_rejects_nonfinite_principal_axes(self):
        with native_dynamics() as module:
            env, _, values, _ = environment(module)
            view = env.scene["robot"].root_physx_view
            setter = view.set_inertias

            def set_inertias(data, ids):
                setter(data, ids)
                values["coms"][0, 0, 3] = float("nan")

            with (
                patch.object(view, "set_inertias", set_inertias),
                self.assertRaisesRegex(ValueError, "Nonfinite.*coms"),
            ):
                module.initialize_dynamics(env)

    def test_native_binding_is_portable_across_independent_gain_draws(self):
        with native_dynamics() as module:
            env = make_native_env(4)
            robot = env.scene["robot"]
            motor = actuator(module)
            motor._joint_names = robot.joint_names
            motor.cfg.to_dict = lambda: {"class_type": "RandomizedDCMotor"}
            robot.actuators["legs"] = motor
            binding, digest = _runtime_motor_binding(env)
            limits = robot.data.soft_joint_pos_limits[0].tolist()
            manifest = dict(
                joint_names=robot.joint_names,
                period_s=0.02,
                configuration=dict(
                    default_position_rad=binding["default_position_rad"],
                    action_clip=dict(
                        version="joint_limits_v1", target_limits_rad=limits
                    ),
                ),
                actuator_profile="native_motor_sha256:" + digest,
            )
            contract = make_motor_contract(
                binding, manifest["actuator_profile"], target_limits_rad=limits
            )
            motor.stiffness_scale[:, ::2] = 0.91
            motor.damping_scale[:, 1::2] = 1.09
            motor.stiffness.copy_(motor.nominal_stiffness * motor.stiffness_scale)
            motor.damping.copy_(motor.nominal_damping * motor.damping_scale)
            motor.motor_strength[0] = 1.1
            changed, changed_digest = _runtime_motor_binding(env)
            self.assertEqual(digest, changed_digest)
            self.assertNotEqual(binding, changed)
            verify_runtime_motor(contract, manifest, changed)
            motor.effort_limit += 0.1
            changed, _ = _runtime_motor_binding(env)
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_runtime_motor(contract, manifest, changed)

    def test_strength_scales_before_real_native_torque_speed_clipping(self):
        with native_dynamics() as module:
            motor = actuator(module)
            motor.motor_strength[:] = torch.tensor([0.9, 1.0, 1.1, 1.1])[:, None]
            position = torch.zeros(4, 12)
            velocity = (
                torch.tensor([0.0, 15.0, 29.0, -29.0])[:, None].expand(-1, 12).clone()
            )
            command = NS(
                joint_positions=torch.full_like(position, 0.5),
                joint_velocities=velocity.clone(),
                joint_efforts=torch.zeros_like(position),
            )
            motor.compute(command, position, velocity)
            expected_raw = torch.full_like(position, 12.5) * motor.motor_strength
            speed_cap = torch.minimum(
                torch.full_like(position, 23.5), 23.5 * (1 - velocity / 30)
            )
            self.assertTrue(torch.equal(motor.computed_effort, expected_raw))
            self.assertTrue(
                torch.equal(
                    command.joint_efforts, torch.minimum(expected_raw, speed_cap)
                )
            )
            self.assertEqual(motor.effort_limit.unique().tolist(), [23.5])
            self.assertEqual(motor.velocity_limit.unique().tolist(), [30.0])
            self.assertIsNone(command.joint_positions)

    def test_fixed_draw_readback_stock_mass_inertia_rule_and_resets(self):
        with native_dynamics() as module:
            env, motor, values, writes = environment(module)
            nominal = deepcopy(values)
            module.initialize_dynamics(env)
            draw = module.get_randomization(env).dynamics()
            self.assertTrue(
                torch.equal(values["masses"][:, 1], nominal["masses"][:, 1])
            )
            self.assertTrue(
                torch.equal(values["inertias"][:, 1], nominal["inertias"][:, 1])
            )
            self.assertTrue(
                torch.equal(
                    values["inertias"][:, 0],
                    nominal["inertias"][:, 0] * (values["masses"][:, 0] / 6)[:, None],
                )
            )
            self.assertTrue(
                torch.equal(values["materials"][:, 0], values["materials"][:, -1])
            )
            self.assertTrue(
                torch.equal(
                    motor.stiffness,
                    motor.nominal_stiffness
                    * torch.tensor(draw["kp_scale"], dtype=torch.float32),
                )
            )
            fixed = deepcopy(values)
            for ids in (torch.tensor([0, 2]), torch.tensor([2, 0])):
                module.reset_state(env, ids)
                self.assertTrue(torch.equal(writes["velocity"], torch.zeros(2, 6)))
                self.assertTrue(torch.equal(writes["joint_vel"], torch.zeros(2, 12)))
                self.assertTrue((writes["joint_pos"].abs() <= 0.05).all())
                self.assertTrue(
                    torch.equal(
                        writes["pose"][:, 2], env.scene.env_origins[ids, 2] + 0.4
                    )
                )
                for key in values:
                    self.assertTrue(torch.equal(values[key], fixed[key]))
            module.dynamics_report(env)
            values["materials"][1, 3, 0] += 0.1
            with self.assertRaisesRegex(ValueError, "readback"):
                module.dynamics_report(env)

    def test_nominal_dynamics_and_gain_drift_rejection(self):
        with native_dynamics() as module:
            env, motor, values, _ = environment(module, dynamics="nominal")
            module.initialize_dynamics(env)
            self.assertEqual(values["masses"][:, 0].tolist(), [6.0] * 4)
            self.assertTrue(torch.equal(motor.stiffness, motor.nominal_stiffness))
            self.assertTrue(torch.equal(motor.motor_strength, torch.ones(4, 12)))
            motor.damping[0, 0] += 0.01
            with self.assertRaisesRegex(ValueError, "PD gains"):
                module.dynamics_report(env)

    def test_actual_first_start_is_checked_not_only_sampled_metadata(self):
        with native_dynamics() as module:
            env, _, _, writes = environment(module)
            module.initialize_dynamics(env)
            module.reset_state(env, torch.arange(env.num_envs))
            data = env.scene["robot"].data
            data.root_pos_w = writes["pose"][:, :3].clone()
            data.root_quat_w = writes["pose"][:, 3:].clone()
            data.root_state_w = torch.cat([writes["pose"], writes["velocity"]], dim=1)
            data.joint_pos, data.joint_vel = writes["joint_pos"], writes["joint_vel"]
            report = module.start_report(env)
            self.assertEqual(report["root_position"], data.root_pos_w.tolist())
            data.root_quat_w.neg_()
            module.start_report(env)
            data.joint_vel[0, 0] = 0.001
            with self.assertRaisesRegex(ValueError, "first start"):
                module.start_report(env)

    def test_configuration_retains_physical_limits_and_replaces_stock_events(self):
        with native_dynamics() as module:
            original = actuator(module).cfg
            original.class_type = module.DCMotor
            cfg = NS(
                events=NS(),
                scene=NS(
                    robot=NS(actuators={"legs": original}),
                    terrain=NS(physics_material=NS()),
                ),
                sim=NS(),
            )
            module.configure_dynamics(
                cfg, dict(num_envs=4, seed=0, dynamics="randomized")
            )
            self.assertIs(original.class_type, module.RandomizedDCMotor)
            self.assertEqual(original.effort_limit, 23.5)
            self.assertEqual(original.velocity_limit, 30.0)
            self.assertIsNone(cfg.events.base_com)
            self.assertIsNone(cfg.events.reset_robot_joints)
            self.assertEqual(cfg.events.dynamics.mode, "startup")
            self.assertIs(cfg.events.reset_base.func, module.reset_state)
            self.assertEqual(
                cfg.scene.terrain.physics_material.friction_combine_mode, "multiply"
            )


if __name__ == "__main__":
    unittest.main()

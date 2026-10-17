"""Fixed native Go2 base wiring with CPU configuration doubles."""

import ast
import unittest
from types import SimpleNamespace as NS

import torch

from parkour_lab.environments.configuration import build_environment_config
from parkour_lab.runtime.native import foot_contacts
from tests.support.isaaclab import (
    EpisodeCommand,
    native_config_modules,
    pinned_source_root,
    stock_config,
    stock_rough_config,
)


class EnvironmentConfigurationTests(unittest.TestCase):
    def build(self, *, training=True):
        with native_config_modules():
            return build_environment_config(
                num_envs=7, seed=97, device="cpu", training=training
            )

    def test_fixed_clock_contact_span_native_actions_and_seed(self):
        cfg = self.build()
        stock = stock_config()
        self.assertEqual((cfg.scene.num_envs, cfg.seed, cfg.sim.device), (7, 97, "cpu"))
        self.assertEqual(
            (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval, cfg.episode_length_s),
            (0.005, 4, 4, 20.0),
        )
        self.assertEqual(cfg.actions, stock.actions)
        self.assertEqual(cfg.scene.robot.soft_joint_pos_limit_factor, 0.9)
        self.assertEqual(cfg.scene.contact_forces.update_period, 0.005)
        self.assertEqual(cfg.scene.contact_forces.history_length, 3)
        self.assertEqual(
            (cfg.scene.contact_forces.history_length - 1) * cfg.sim.dt, 0.01
        )
        self.assertIsNone(cfg.curriculum.terrain_levels)
        self.assertEqual(
            cfg.parkour_task, dict(num_envs=7, seed=97, dynamics="randomized")
        )

    def test_only_causal_49d_group_and_native_termination_terms(self):
        cfg = self.build()
        actor = cfg.observations.proprio
        self.assertIsNone(cfg.observations.policy)
        self.assertFalse(hasattr(cfg.observations, "terrain"))
        self.assertIsNone(actor.base_lin_vel)
        self.assertFalse(actor.enable_corruption)
        terms = [
            name
            for name, value in vars(actor).items()
            if value is not None
            and name not in ("enable_corruption", "concatenate_terms")
        ]
        self.assertEqual(
            terms,
            [
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "actions",
                "foot_contacts",
            ],
        )
        self.assertIs(actor.foot_contacts.func, foot_contacts)
        self.assertFalse(getattr(actor.foot_contacts, "noise", None))
        self.assertEqual(actor.base_ang_vel.noise, (-0.2, 0.2))
        self.assertFalse(cfg.terminations.base_contact.time_out)
        self.assertTrue(cfg.terminations.time_out.time_out)
        self.assertEqual(set(vars(cfg.terminations)), {"base_contact", "time_out"})

    def test_every_role_uses_native_rough_rewards_without_overrides(self):
        expected = stock_rough_config().rewards
        for training in (True, False):
            cfg = self.build(training=training)
            self.assertEqual(vars(cfg.rewards), vars(expected))
            self.assertIsNot(cfg.rewards.feet_air_time, expected.feet_air_time)
            self.assertFalse(hasattr(cfg.rewards, "joint_posture"))
            self.assertEqual(cfg.parkour_evaluation, not training)

    def test_common_command_class_with_explicit_external_evaluation_ranges(self):
        for training in (True, False):
            cfg = self.build(training=training)
            command = cfg.commands.base_velocity
            self.assertIs(command.class_type, EpisodeCommand)
            self.assertEqual(command.resampling_time_range, (20.0, 20.0))
            self.assertFalse(command.heading_command)
            self.assertEqual(
                (command.rel_heading_envs, command.rel_standing_envs), (0.0, 0.0)
            )
            self.assertIsNone(command.ranges.heading)
            self.assertFalse(command.debug_vis)
            expected = (
                ((-0.2, 0.5), (-0.2, 0.2), (-0.5, 0.5))
                if training
                else ((0.0, 0.0),) * 3
            )
            self.assertEqual(
                (
                    command.ranges.lin_vel_x,
                    command.ranges.lin_vel_y,
                    command.ranges.ang_vel_z,
                ),
                expected,
            )

    def test_privileged_raycasters_retain_native_measurement_geometry(self):
        cfg = self.build()
        scan = cfg.scene.height_scanner
        self.assertEqual(
            (
                scan.pattern_cfg.resolution,
                scan.pattern_cfg.size,
                scan.pattern_cfg.ordering,
            ),
            (0.15, (1.65, 1.50), "xy"),
        )
        self.assertEqual(scan.offset.pos, (0.0, 0.0, 20.0))
        self.assertEqual(scan.mesh_prim_paths, [cfg.scene.terrain.prim_path])
        self.assertEqual(scan.ray_alignment, "yaw")
        self.assertEqual(scan.max_distance, 25.0)
        self.assertEqual(scan.update_period, 0.0)
        self.assertFalse(scan.debug_vis)

    def test_invalid_parameters_fail_before_native_imports(self):
        valid = dict(num_envs=7, seed=97, device="cpu", training=True)
        for changes in (
            {"num_envs": True},
            {"num_envs": 0},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": False},
            {"device": " "},
            {"training": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_environment_config(**{**valid, **changes})

    def test_pinned_lazy_rays_stay_fresh_without_substep_casts_or_duplicate_reads(self):
        # Execute SDK scheduling/cache/reset and RayCaster.data, not a reimplementation.
        methods = []
        for relative, class_name, names in (
            (
                "sensors/sensor_base.py",
                "SensorBase",
                {"update", "reset", "_update_outdated_buffers"},
            ),
            ("sensors/ray_caster/ray_caster.py", "RayCaster", {"data"}),
        ):
            path = pinned_source_root() / relative
            tree = ast.parse(path.read_text())
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            methods.extend(
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            )
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *methods,
            ],
            type_ignores=[],
        )
        namespace = {}
        exec(
            compile(ast.fix_missing_locations(code), "pinned_lazy_ray_methods", "exec"),
            namespace,
        )

        def cast(sensor, ids):
            sensor._data[ids] = sensor.state[ids]
            sensor.casts.append(ids.tolist())

        sensor_type = type(
            "LazyRay",
            (),
            {
                **{
                    name: namespace[name]
                    for name in ("update", "reset", "_update_outdated_buffers", "data")
                },
                "_update_buffers_impl": cast,
            },
        )

        def run(period, rate, steps):
            sensor = sensor_type()
            sensor.cfg = NS(update_period=period, history_length=0)
            sensor._is_visualizing = False
            sensor._timestamp = torch.zeros(2)
            sensor._timestamp_last_update = torch.zeros(2)
            sensor._is_outdated = torch.ones(2, dtype=torch.bool)
            sensor.state = torch.zeros(2, dtype=torch.long)
            sensor._data = torch.full((2,), -1, dtype=torch.long)
            sensor.casts = []
            self.assertTrue(torch.equal(sensor.data, sensor.state))
            stale = []
            for step in range(steps):
                calls = len(sensor.casts)
                for _ in range(rate // 50):
                    sensor.state += 1
                    sensor.update(1 / rate)
                    self.assertEqual(len(sensor.casts), calls)
                sample = sensor.data.clone()
                if not torch.equal(sample, sensor.state):
                    stale.append(step)
                calls = len(sensor.casts)
                self.assertTrue(torch.equal(sensor.data, sample))
                self.assertEqual(len(sensor.casts), calls)
            return sensor, stale

        _, stale = run(0.02, 400, 556)
        self.assertEqual(stale, list(range(400, 556, 2)))
        _, stale = run(0.02, 200, 1500)
        self.assertEqual(stale, list(range(800, 1500, 2)))
        for rate in (200, 400):
            cfg = self.build()
            with self.subTest(rate=rate):
                sensor, stale = run(cfg.scene.height_scanner.update_period, rate, 1500)
                self.assertEqual(stale, [])
                self.assertEqual(
                    len(sensor.casts), 1501
                )  # initial + one per control boundary
                # Outgoing terminal read is cached; reset invalidates only ended rows.
                calls = len(sensor.casts)
                outgoing = sensor.data.clone()
                self.assertEqual(len(sensor.casts), calls)
                sensor.state[0] = -1
                sensor.reset(torch.tensor([0]))
                self.assertEqual(len(sensor.casts), calls)
                self.assertTrue(torch.equal(sensor.data, sensor.state))
                self.assertEqual(sensor.casts[-1], [0])
                self.assertEqual(len(sensor.casts), calls + 1)
                self.assertTrue(torch.equal(sensor.data, sensor.state))
                self.assertEqual(len(sensor.casts), calls + 1)
                self.assertNotEqual(outgoing[0], sensor.data[0])
                self.assertEqual(outgoing[1], sensor.data[1])

"""Current task wiring with lightweight native config doubles, not physics tests."""

import ast
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.config import ExperimentConfig, TaskConfig
from parkour_lab.environments.configuration import build_environment_config
from parkour_lab.runtime.native import foot_contacts
from tests.support.isaaclab import (
    FlatCommand,
    NativeCommand,
    RoughCommand,
    TerrainCommand,
    com_velocity,
    native_config_modules,
    pinned_source_root,
    scan,
    stock_config,
    stock_rough_config,
)


class EnvironmentConfigurationTests(unittest.TestCase):
    def test_rough_defaults_round_trip_without_changing_existing_defaults(self):
        task = TaskConfig(terrain="rough")
        self.assertEqual((task.num_rows, task.difficulty_range), (10, (0.0, 1.0)))
        self.assertEqual(task.rough_pivot_fraction, 0.0)
        config = ExperimentConfig(task=task)
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        config.validate_training()
        self.assertEqual(
            (TaskConfig().num_rows, TaskConfig().difficulty_range), (1, (0.05, 0.15))
        )

    def test_rough_pivot_fraction_validation_round_trip_and_external_bypass(self):
        for fraction in (0, 0.2, 0.9):
            config = ExperimentConfig(
                task=TaskConfig(
                    terrain="rough",
                    rough_pivot_fraction=fraction,
                )
            )
            self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        for fraction in (-0.01, 0.901, float("nan"), float("inf"), True, "0.2"):
            with (
                self.subTest(fraction=fraction),
                self.assertRaisesRegex(ValueError, "rough_pivot_fraction"),
            ):
                TaskConfig(terrain="rough", rough_pivot_fraction=fraction)
        for terrain in ("rough", "flat"):
            task = TaskConfig(terrain=terrain, rough_pivot_fraction=0.2)
            with native_config_modules():
                cfg = build_environment_config(task, evaluation=True)
            self.assertEqual(cfg.parkour_task["rough_pivot_fraction"], 0.2)
            self.assertIs(cfg.commands.base_velocity.class_type, NativeCommand)
            self.assertEqual(cfg.commands.base_velocity.ranges.ang_vel_z, (0.0, 0.0))

    def test_rough_reuses_curriculum_bounded_dynamics_and_causal_sensing(self):
        original = stock_rough_config()
        for evaluation in (False, True):
            for rate in (200, 400):
                with (
                    self.subTest(evaluation=evaluation, rate=rate),
                    native_config_modules(rough=original),
                ):
                    task = TaskConfig(
                        terrain="rough",
                        physics_hz=rate,
                        rewards={"joint_posture": {"weight": -0.2}},
                    )
                    cfg = build_environment_config(task, evaluation=evaluation)
                generator = cfg.scene.terrain.terrain_generator
                self.assertEqual(
                    (
                        generator.num_rows,
                        generator.num_cols,
                        generator.difficulty_range,
                        generator.seed,
                    ),
                    (10, 25, (0.0, 1.0), task.seed),
                )
                self.assertTrue(generator.curriculum)
                self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
                self.assertEqual(
                    list(generator.sub_terrains),
                    [*original.scene.terrain.terrain_generator.sub_terrains, "flat"],
                )
                self.assertEqual(generator.sub_terrains["flat"].proportion, 0.25)
                self.assertEqual(cfg.parkour_task["terrain"], "rough")
                self.assertEqual(cfg.parkour_evaluation, evaluation)
                self.assertEqual(cfg.sim.dt * cfg.decimation, 0.02)
                self.assertIsNone(cfg.observations.proprio.base_lin_vel)
                self.assertIsNone(cfg.observations.proprio.height_scan)
                self.assertIsNone(cfg.observations.policy.height_scan)
                self.assertFalse(cfg.observations.proprio.enable_corruption)
                self.assertFalse(cfg.observations.policy.enable_corruption)
                self.assertIs(
                    cfg.observations.proprio.foot_contacts.func, foot_contacts
                )
                self.assertEqual(
                    cfg.observations.terrain.height_scan.params["obs_cfg"].num_rays, 132
                )
                self.assertEqual(cfg.rewards.joint_posture.weight, -0.2)
                self.assertEqual(cfg.rewards.flat_orientation_l2.weight, -2.5)
                self.assertFalse(hasattr(cfg.terminations, "procedural_workspace"))
                self.assertEqual(
                    cfg.terminations.procedural_physical_failure.params[
                        "minimum_surface_z_m"
                    ],
                    -4.0,
                )
                if evaluation:
                    self.assertIsNone(cfg.curriculum.terrain_levels)
                    self.assertIs(cfg.commands.base_velocity.class_type, NativeCommand)
                else:
                    self.assertIs(
                        cfg.curriculum.terrain_levels.func,
                        original.curriculum.terrain_levels.func,
                    )
                    self.assertIs(cfg.commands.base_velocity.class_type, RoughCommand)
                    self.assertFalse(cfg.commands.base_velocity.heading_command)
                    self.assertEqual(
                        cfg.commands.base_velocity.resampling_time_range, (4.0, 4.0)
                    )
        self.assertEqual(original.scene.terrain.terrain_generator.num_cols, 20)
        self.assertIsNotNone(original.observations.policy.height_scan)

    def test_connected_reuses_causal_dynamics_and_excludes_procedural_generator(self):
        task = TaskConfig(
            terrain="connected",
            num_envs=1,
            dynamics="nominal",
            physics_hz=400,
            world=dict(
                target_family="ramp",
                tier=10,
                coarse_seed=17,
                fine_seed=23,
                reverse=True,
                world_yaw=0.37,
            ),
        )
        with patch("parkour_lab.environments.terrains.connected.configure") as install:
            cfg = self.build(task, evaluation=True)
        install.assert_called_once_with(cfg, task)
        self.assertEqual(cfg.parkour_task["world"], task.world)
        self.assertTrue(cfg.parkour_evaluation)
        self.assertIsNone(cfg.scene.terrain.terrain_generator)
        self.assertFalse(hasattr(cfg.terminations, "procedural_workspace"))
        self.assertEqual(
            cfg.terminations.procedural_physical_failure.params["minimum_surface_z_m"],
            -0.02,
        )
        self.assertFalse(cfg.observations.proprio.enable_corruption)
        self.assertIs(cfg.commands.base_velocity.class_type, NativeCommand)
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_x, (0.0, 0.0))
        self.assertEqual((cfg.sim.dt, cfg.decimation), (0.0025, 8))
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            self.build(task)

    def build(self, task=None, **kwargs):
        with native_config_modules():
            return build_environment_config(task or TaskConfig(), **kwargs)

    def test_physics_rate_is_explicit_integer_with_historical_default(self):
        self.assertEqual(ExperimentConfig.from_dict({}).task.physics_hz, 200)
        for rate in (200, 400):
            config = ExperimentConfig(task=TaskConfig(physics_hz=rate))
            self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        for rate in (True, False, 200.0, 400.0, "400", None, 100, 800):
            with (
                self.subTest(rate=rate),
                self.assertRaisesRegex(ValueError, "physics_hz"),
            ):
                TaskConfig(physics_hz=rate)

    def test_both_physics_rates_preserve_control_scan_render_and_contact_span(self):
        for rate, dt, decimation, history in (
            (200, 0.005, 4, 3),
            (400, 0.0025, 8, 5),
        ):
            with self.subTest(rate=rate):
                cfg = self.build(TaskConfig(physics_hz=rate))
                self.assertEqual((cfg.sim.dt, cfg.decimation), (dt, decimation))
                self.assertEqual(cfg.sim.dt * cfg.decimation, 0.02)
                self.assertEqual(cfg.sim.render_interval, decimation)
                self.assertEqual(cfg.scene.height_scanner.update_period, 0.0)
                self.assertEqual(cfg.scene.base_height_scanner.update_period, 0.0)
                self.assertEqual(cfg.scene.contact_forces.update_period, dt)
                self.assertEqual(cfg.scene.contact_forces.history_length, history)
                self.assertEqual((history - 1) * dt, 0.01)

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
            cfg = self.build(TaskConfig(physics_hz=rate))
            for name in ("height_scanner", "base_height_scanner"):
                with self.subTest(rate=rate, scanner=name):
                    sensor, stale = run(
                        getattr(cfg.scene, name).update_period, rate, 1500
                    )
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

    def test_fresh_defaults_preserve_native_physics_and_current_acquisition_values(
        self,
    ):
        original = stock_config()
        with native_config_modules(original):
            cfg = build_environment_config(
                TaskConfig(num_envs=7, seed=97, device="cpu", episode_length_s=12.5)
            )
        self.assertEqual(
            (cfg.scene.num_envs, cfg.seed, cfg.sim.device, cfg.episode_length_s),
            (7, 97, "cpu", 12.5),
        )
        self.assertEqual(cfg.actions, original.actions)
        self.assertEqual((cfg.sim.dt, cfg.decimation), (0.005, 4))
        self.assertEqual(cfg.events.add_base_mass, original.events.add_base_mass)
        self.assertEqual(cfg.events.physics_material, original.events.physics_material)
        self.assertEqual(
            cfg.events.reset_base.params["velocity_range"],
            original.events.reset_base.params["velocity_range"],
        )
        self.assertIsNone(cfg.events.base_com)
        self.assertIsNone(cfg.events.push_robot)
        self.assertEqual(
            cfg.rewards.track_lin_vel_xy_exp, original.rewards.track_lin_vel_xy_exp
        )
        self.assertEqual(
            cfg.rewards.track_ang_vel_z_exp, original.rewards.track_ang_vel_z_exp
        )
        self.assertEqual(
            (
                cfg.rewards.dof_pos_limits.weight,
                cfg.rewards.flat_orientation_l2.weight,
                cfg.rewards.feet_air_time.weight,
            ),
            (-10.0, -2.5, 0.01),
        )
        self.assertEqual(
            cfg.events.reset_base.params["pose_range"],
            {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
        )
        self.assertEqual(original.rewards.dof_pos_limits.weight, 0.0)
        self.assertIs(cfg.commands.base_velocity.class_type, TerrainCommand)
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_x, (-0.3, 0.7))
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_y, (-0.2, 0.2))
        self.assertEqual(cfg.commands.base_velocity.ranges.ang_vel_z, (-0.8, 0.8))
        self.assertEqual(cfg.commands.base_velocity.resampling_time_range, (2.0, 12.0))

    def test_actor_is_noisy_49d_order_with_noiseless_contacts_and_clean_critic(self):
        cfg = self.build()
        actor, critic = cfg.observations.proprio, cfg.observations.policy
        self.assertIsNone(actor.base_lin_vel)
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
        self.assertEqual(sum((3, 3, 3, 12, 12, 12, 4)), 49)
        self.assertIs(cfg.observations.proprio.foot_contacts.func, foot_contacts)
        self.assertIs(cfg.observations.policy.foot_contacts.func, foot_contacts)
        self.assertFalse(getattr(cfg.observations.proprio.foot_contacts, "noise", None))
        self.assertEqual(cfg.scene.contact_forces.update_period, 0.005)
        self.assertTrue(actor.enable_corruption)
        self.assertFalse(critic.enable_corruption)
        self.assertIs(critic.base_lin_vel.func, com_velocity)
        self.assertIsNot(actor.base_ang_vel, critic.base_ang_vel)
        self.assertEqual(actor.base_ang_vel.noise, (-0.2, 0.2))
        self.assertFalse(hasattr(actor, "contacts"))

    def test_tracking_widths_change_only_existing_reward_parameters(self):
        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation):
                reference = self.build(
                    TaskConfig(terrain="flat"), evaluation=evaluation
                )
                candidate = self.build(
                    TaskConfig(
                        terrain="flat",
                        linear_tracking_std=0.25,
                        angular_tracking_std=0.3,
                    ),
                    evaluation=evaluation,
                )
                for name, width in (
                    ("track_lin_vel_xy_exp", 0.25),
                    ("track_ang_vel_z_exp", 0.3),
                ):
                    term = getattr(candidate.rewards, name)
                    self.assertEqual(term.params["std"], width)
                    term.params["std"] = 0.5
                # Restoring just the two widths gives the identical native config.
                candidate.parkour_task["linear_tracking_std"] = 0.5
                candidate.parkour_task["angular_tracking_std"] = 0.5
                self.assertEqual(vars(candidate), vars(reference))

    def test_reward_overrides_preserve_native_functions_selectors_and_unrelated_task(
        self,
    ):
        overrides = {
            "dof_torques_l2": {"weight": -0.0001},
            "action_rate_l2": {"weight": 0.0},
            "feet_air_time": {"weight": 0.02, "params": {"threshold": 0.3}},
            "joint_posture": {"weight": -0.2, "params": {"stand_still_scale": 3.0}},
        }
        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation):
                reference = self.build(
                    TaskConfig(terrain="flat"), evaluation=evaluation
                )
                candidate = self.build(
                    TaskConfig(terrain="flat", rewards=overrides), evaluation=evaluation
                )
                for name, settings in overrides.items():
                    before, after = (
                        getattr(reference.rewards, name),
                        getattr(candidate.rewards, name),
                    )
                    self.assertIs(after.func, before.func)
                    self.assertEqual(after.weight, settings["weight"])
                    for key, value in settings.get("params", {}).items():
                        self.assertEqual(after.params[key], value)
                    fixed = before.params.keys() - settings.get("params", {}).keys()
                    self.assertEqual(
                        {key: after.params[key] for key in fixed},
                        {key: before.params[key] for key in fixed},
                    )
                    setattr(candidate.rewards, name, before)
                candidate.parkour_task["rewards"] = reference.parkour_task["rewards"]
                self.assertEqual(vars(candidate), vars(reference))

    def test_default_reward_recipe_preserves_acquisition_weights_with_posture_disabled(
        self,
    ):
        cfg = self.build(TaskConfig(terrain="flat"))
        self.assertEqual(
            {
                name: term.weight
                for name, term in vars(cfg.rewards).items()
                if term is not None
            },
            {
                "track_lin_vel_xy_exp": 1.5,
                "track_ang_vel_z_exp": 0.75,
                "lin_vel_z_l2": -2.0,
                "ang_vel_xy_l2": -0.05,
                "dof_torques_l2": -0.0002,
                "dof_acc_l2": -2.5e-7,
                "action_rate_l2": -0.01,
                "feet_air_time": 0.01,
                "flat_orientation_l2": -2.5,
                "dof_pos_limits": -10.0,
                "joint_posture": 0.0,
            },
        )

    def test_real_264d_scan_and_physical_failures_precede_workspace_timeout(self):
        cfg = self.build()
        scanner = cfg.scene.height_scanner
        self.assertEqual(
            (
                scanner.pattern_cfg.resolution,
                scanner.pattern_cfg.size,
                scanner.pattern_cfg.ordering,
            ),
            (0.15, (1.65, 1.50), "xy"),
        )
        self.assertEqual(
            (
                scanner.offset.pos,
                scanner.ray_alignment,
                scanner.max_distance,
                scanner.update_period,
            ),
            ((0.0, 0.0, 20.0), "yaw", 25.0, 0.0),
        )
        scan_cfg = cfg.observations.terrain.height_scan.params["obs_cfg"]
        self.assertEqual(
            (scan_cfg.num_rays * 2, scan_cfg.vertical_offset, scan_cfg.clip),
            (264, 0.3, 0.5),
        )
        self.assertEqual(cfg.scene.base_height_scanner.pattern_cfg.size, (0.0, 0.0))
        physical = cfg.terminations.procedural_physical_failure
        self.assertEqual(
            physical.params,
            {
                "minimum_m": 0.12,
                "minimum_surface_z_m": -0.32 * 0.15,
                "fall_margin_m": 0.5,
                "max_tilt_rad": math.pi / 4,
            },
        )
        self.assertFalse(physical.time_out)
        self.assertEqual(
            list(vars(cfg.terminations))[-2:],
            ["procedural_physical_failure", "procedural_workspace"],
        )
        self.assertTrue(cfg.terminations.procedural_workspace.time_out)

    def test_all_configured_rows_are_available_without_manager_curriculum(self):
        for rows in (1, 3, 5):
            cfg = self.build(TaskConfig(num_rows=rows, difficulty_range=(0.1, 0.4)))
            self.assertEqual(cfg.scene.terrain.max_init_terrain_level, rows - 1)
            self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, rows)
            self.assertEqual(
                cfg.scene.terrain.terrain_generator.difficulty_range, (0.1, 0.4)
            )
            self.assertIsNone(cfg.curriculum.terrain_levels)

    def test_flat_uses_real_plane_scan_and_current_flat_sampler(self):
        cfg = self.build(TaskConfig(terrain="flat"))
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertIsNone(cfg.scene.terrain.terrain_generator)
        self.assertIs(cfg.commands.base_velocity.class_type, FlatCommand)
        self.assertEqual(cfg.scene.height_scanner.mesh_prim_paths, ["/World/ground"])
        self.assertIs(cfg.observations.terrain.height_scan.func, scan)
        self.assertEqual(
            cfg.observations.terrain.height_scan.params["obs_cfg"].num_rays, 132
        )
        self.assertEqual(
            cfg.terminations.procedural_physical_failure.params["minimum_surface_z_m"],
            0.0,
        )
        self.assertFalse(hasattr(cfg.terminations, "procedural_workspace"))

    def test_steps_use_exact_current_callbacks_and_three_exposed_rows(self):
        from parkour_lab.environments.terrains.step_field import step_field_terrain

        cfg = self.build(TaskConfig(terrain="steps"))
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 2)
        generator = cfg.scene.terrain.terrain_generator
        self.assertEqual(generator.difficulty_range, (0.15, 0.55))
        terrains = list(generator.sub_terrains.values())
        self.assertTrue(
            all(item.function is step_field_terrain for item in terrains[12:16])
        )
        self.assertTrue(
            all(
                item.function is not step_field_terrain
                for item in terrains[:12] + terrains[16:]
            )
        )

    def test_evaluation_owns_commands_and_cleans_actor_without_removing_randomization(
        self,
    ):
        for terrain in ("flat", "procedural", "steps", "traversal"):
            with self.subTest(terrain=terrain):
                cfg = self.build(TaskConfig(terrain=terrain), evaluation=True)
                command = cfg.commands.base_velocity
                self.assertIs(command.class_type, NativeCommand)
                self.assertEqual(command.ranges.lin_vel_x, (0.0, 0.0))
                self.assertEqual(command.ranges.lin_vel_y, (0.0, 0.0))
                self.assertEqual(command.ranges.ang_vel_z, (0.0, 0.0))
                self.assertEqual(command.resampling_time_range, (1e9, 1e9))
                self.assertFalse(command.heading_command)
                self.assertEqual(
                    (command.rel_heading_envs, command.rel_standing_envs), (0.0, 0.0)
                )
                self.assertFalse(cfg.observations.proprio.enable_corruption)
                self.assertEqual(
                    cfg.events.add_base_mass.params["mass_distribution_params"],
                    (-1.0, 3.0),
                )

    def test_traversal_layout_is_explicit_fixed_geometry(self):
        from parkour_lab.environments.terrains.traversal import profiles_for_layout

        for layout in ("standard", "step_ladder"):
            cfg = self.build(
                TaskConfig(terrain="traversal", traversal_layout=layout),
                evaluation=True,
            )
            self.assertEqual(
                tuple(cfg.scene.terrain.terrain_generator.sub_terrains),
                profiles_for_layout(layout),
            )
            self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
            self.assertEqual(
                cfg.scene.terrain.terrain_generator.difficulty_range, (1.0, 1.0)
            )

    def test_unsupported_settings_fail_before_native_import(self):
        for task, evaluation in (
            (TaskConfig(num_rows=2), False),
            (TaskConfig(terrain="steps", num_rows=1), False),
            (TaskConfig(terrain="steps", difficulty_range=(0.1, 0.55)), False),
            (TaskConfig(terrain="traversal"), False),
            (TaskConfig(terrain="traversal", num_rows=3), True),
            (TaskConfig(terrain="traversal", difficulty_range=(0.0, 1.0)), True),
            ({}, False),
            (TaskConfig(), "yes"),
        ):
            with self.subTest(task=task), self.assertRaises(ValueError):
                build_environment_config(task, evaluation=evaluation)


if __name__ == "__main__":
    unittest.main()

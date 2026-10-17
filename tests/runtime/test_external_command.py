"""Real pinned native command lifecycle with CPU scene/import dependencies.

The Isaac Lab command/base classes are loaded unmodified from the
local 2.3.2 checkout. Only simulator imports and configuration containers are
stubbed. These tests do not start PhysX or establish robot traversal behavior.
"""

import importlib.util
import inspect
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

import parkour_lab.runtime.native as runtime
from parkour_lab.benchmark import Go2Config, build_config
from parkour_lab.control.controller import ControllerSpec, JointTargets
from parkour_lab.control.motor_contract import make_motor_contract
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.runtime.motor import _runtime_motor_binding
from tests.support import SOURCE_ROOT
from tests.support.isaaclab import pinned_source_root
from tests.support.native import NativeEnvironmentFixture
from tests.support.public_task import public_configs


@contextmanager
def native_command_classes():
    """Import actual implementations without importing Kit or simulator assets."""
    source = pinned_source_root()
    modules = {}

    def module(name, **attributes):
        result = ModuleType(name)
        result.__path__ = []
        result.__dict__.update(attributes)
        modules[name] = result
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, result)
        return result

    forbidden = Mock(
        side_effect=AssertionError("Unexpected simulator or heading dependency")
    )
    module("omni")
    module("omni.timeline")
    module("omni.kit")
    module("omni.kit.app", get_app_interface=forbidden)
    module("prettytable", PrettyTable=forbidden)
    module("isaaclab")
    module(
        "isaaclab.utils",
        class_to_dict=vars,
        string_to_callable=forbidden,
        configclass=lambda cls: cls,
    )
    module("isaaclab.utils.string")
    module("isaaclab.utils.math", wrap_to_pi=forbidden)
    module("isaaclab.assets", Articulation=NS)
    module("isaaclab.markers", VisualizationMarkers=forbidden)
    managers = module("isaaclab.managers")
    module(
        "isaaclab.managers.manager_term_cfg", ManagerTermBaseCfg=NS, CommandTermCfg=NS
    )
    module("isaaclab.managers.scene_entity_cfg", SceneEntityCfg=NS)
    module("isaaclab.terrains", SubTerrainBaseCfg=NS, TerrainGeneratorCfg=NS)
    module("isaaclab.envs")
    module("isaaclab.envs.mdp")
    commands = module("isaaclab.envs.mdp.commands")
    module(
        "parkour_lab.environments.dynamics",
        get_randomization=lambda env: env.parkour_randomization,
    )

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    loaded_names = (
        "isaaclab.managers.manager_base",
        "isaaclab.managers.command_manager",
        "isaaclab.envs.mdp.commands.velocity_command",
        "parkour_lab.environments.commands._episode_native_test",
    )
    missing = object()
    originals = {
        name: sys.modules.get(name, missing) for name in (*modules, *loaded_names)
    }
    # Restore only these dependencies, not all of sys.modules: Torch can
    # legitimately lazy-import extension registrations during the lifecycle.
    sys.modules.update(modules)
    try:
        base = load(
            "isaaclab.managers.manager_base", source / "managers/manager_base.py"
        )
        manager = load(
            "isaaclab.managers.command_manager", source / "managers/command_manager.py"
        )
        managers.ManagerTermBase = base.ManagerTermBase
        managers.CommandTerm = manager.CommandTerm
        velocity = load(
            "isaaclab.envs.mdp.commands.velocity_command",
            source / "envs/mdp/commands/velocity_command.py",
        )
        commands.UniformVelocityCommand = velocity.UniformVelocityCommand
        episode = load(
            "parkour_lab.environments.commands._episode_native_test",
            SOURCE_ROOT / "parkour_lab/environments/commands/episode.py",
        )
        yield NS(
            source=source,
            ManagerTermBase=base.ManagerTermBase,
            CommandTerm=manager.CommandTerm,
            UniformVelocityCommand=velocity.UniformVelocityCommand,
            EpisodeVelocityCommand=episode.EpisodeVelocityCommand,
            forbidden=forbidden,
        )
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class ExternalCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pinned_source_root()

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(143)

    def assert_external_config(self, cfg, command_class):
        self.assertIs(cfg.class_type, command_class)
        self.assertFalse(cfg.heading_command)
        self.assertEqual((cfg.rel_heading_envs, cfg.rel_standing_envs), (0.0, 0.0))
        self.assertIsNone(cfg.ranges.heading)
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            self.assertEqual(getattr(cfg.ranges, name), (0.0, 0.0))
        self.assertEqual(cfg.resampling_time_range, (20.0, 20.0))
        self.assertFalse(cfg.debug_vis)
        self.assertEqual(cfg.asset_name, "robot")

    def configure_fixture(self, native):
        with public_configs():
            cfg = build_config(
                Go2Config(num_envs=4, seed=143, device="cpu"), training=False
            )
        # The small stock-config double omits this SDK-inherited asset selector.
        cfg.commands.base_velocity.asset_name = "robot"
        cfg.commands.base_velocity.class_type = native.EpisodeVelocityCommand
        self.assert_external_config(
            cfg.commands.base_velocity, native.EpisodeVelocityCommand
        )
        fixture = NativeEnvironmentFixture(count=4)
        fixture.env.cfg = cfg
        fixture.env.parkour_randomization = TaskRandomization(
            cfg.parkour_task, evaluation=True
        )
        term = native.EpisodeVelocityCommand(cfg.commands.base_velocity, fixture.env)
        self.assertIsInstance(term, native.CommandTerm)
        self.assertIsInstance(term, native.ManagerTermBase)
        fixture.command_term = term
        fixture.command = term.command
        fixture.env.command_manager.get_term = lambda name: term
        fixture.env.command_manager.get_command = lambda name: term.command
        fixture.on_reset = lambda: term.reset(torch.arange(4))
        fixture.next_terminated = torch.zeros(4, dtype=torch.bool)
        fixture.next_timed_out = torch.zeros(4, dtype=torch.bool)
        return fixture, term

    def test_public_config_constructs_real_episode_on_four_flat_replicas(self):
        with native_command_classes() as native:
            fixture, term = self.configure_fixture(native)
            self.assertEqual(
                Path(inspect.getfile(type(term))),
                SOURCE_ROOT / "parkour_lab/environments/commands/episode.py",
            )
            self.assertEqual(
                Path(inspect.getfile(native.CommandTerm)),
                native.source / "managers/command_manager.py",
            )
            metrics = term.reset(torch.arange(4))
            self.assertEqual(set(metrics), {"error_vel_xy", "error_vel_yaw"})
            self.assertTrue(torch.isfinite(term.command).all())
            self.assertTrue((term.time_left == 20.0).all())
            self.assertTrue((term.command_counter == 1).all())
            self.assertFalse(term.is_heading_env.any())
            self.assertEqual(fixture.env.cfg.benchmark_task_id, "Go2-Flat-v1")
            native.forbidden.assert_not_called()

    def test_real_compute_preserves_signed_external_commands_without_resampling(self):
        with native_command_classes() as native:
            _, term = self.configure_fixture(native)
            term.reset(torch.arange(4))
            counter = term.command_counter.clone()
            term.time_left.fill_(float("inf"))
            term.is_standing_env.fill_(False)
            term.heading_target.fill_(100.0)
            term.is_heading_env.fill_(
                True
            )  # Disabled cfg must prevent heading steering.
            for desired in ((0.4, 0.1, 0.5), (-0.2, -0.1, -0.5), (0.0, 0.0, 0.0)):
                expected = torch.tensor(desired).expand(4, 3)
                term.vel_command_b.copy_(expected)
                for _ in range(100):
                    term.compute(0.02)
                    self.assertTrue(torch.equal(term.command, expected))
                    self.assertTrue(torch.isfinite(term.command).all())
                self.assertTrue(torch.equal(term.command_counter, counter))
                self.assertTrue((term.time_left > 0).all())
            native.forbidden.assert_not_called()

    def test_real_partial_reset_preserves_outgoing_rows_then_resumes_external_tape(
        self,
    ):
        with native_command_classes() as native:
            fixture, term = self.configure_fixture(native)
            fixture.env.reset(seed=143)
            binding, digest = _runtime_motor_binding(fixture.env)
            profile = "native_motor_sha256:" + digest
            limits = fixture.data.soft_joint_pos_limits[0].tolist()
            raw = torch.linspace(-0.2, 0.2, 12).repeat(4, 1)
            default = fixture.data.default_joint_pos
            spec = ControllerSpec(
                name="external_command_fixture",
                artifact_sha256="a" * 64,
                preprocessing_version="named_inputs_v1",
                joint_names=tuple(binding["joint_names"]),
                period_s=0.02,
                actuator_profile=profile,
                sensors={},
                raw_action_meaning="none",
                configuration={
                    "default_position_rad": binding["default_position_rad"],
                    "action_clip": {
                        "version": "joint_limits_v1",
                        "target_limits_rad": limits,
                    },
                },
            )
            controller = NS(
                spec=spec,
                reset=lambda mask: None,
                act=lambda inputs: JointTargets(spec.joint_names, default + 0.25 * raw),
            )
            host = runtime.NativeControllerSession(
                fixture.env,
                controller,
                make_motor_contract(binding, profile, target_limits_rad=limits),
            )
            first = torch.ones(4, dtype=torch.bool)
            initial_command = torch.tensor((0.4, 0.0, 0.5)).expand(4, 3)
            targets = host.act(initial_command, time_s=0.0, reset_mask=first)
            self.assertTrue(torch.isinf(term.time_left).all())
            selected = torch.tensor([False, True, False, True])
            fixture.next_terminated = selected.clone()
            reset_commands = []

            def native_command_update():
                done = fixture.next_terminated | fixture.next_timed_out
                if done.any():
                    term.reset(done.nonzero().flatten())
                    reset_commands.append(term.command.clone())
                term.compute(fixture.env.step_dt)

            fixture.on_step = native_command_update
            raw = torch.linspace(-0.2, 0.2, 12).repeat(4, 1)
            next_command = (-0.2, 0.1, -0.5)
            delivered = host.motor.encode(targets)
            observations, _, terminated, truncated, extras = fixture.env.step(delivered)
            host.motor.verify_delivery(terminated, truncated)
            done = terminated | truncated
            targets = host.act(
                torch.tensor(next_command).expand(4, 3), time_s=0.02, reset_mask=done
            )
            observations = fixture.observations()
            self.assertTrue(torch.equal(done, selected))
            self.assertEqual(len(reset_commands), 1)
            self.assertTrue(torch.isfinite(reset_commands[0][selected]).all())
            torch.testing.assert_close(
                extras["final_observation"]["proprio"][:, 6:9],
                initial_command[selected],
            )
            self.assertTrue(
                torch.equal(
                    reset_commands[0][~selected],
                    torch.tensor((0.4, 0.0, 0.5)).expand(2, 3),
                )
            )
            expected = torch.tensor(next_command).expand(4, 3)
            self.assertTrue(torch.equal(term.command, expected))
            self.assertTrue(torch.equal(observations["proprio"][:, 6:9], expected))
            self.assertTrue(torch.equal(observations["policy"][:, 9:12], expected))
            self.assertFalse(observations["proprio"][selected, 33:45].any())
            self.assertTrue(
                torch.equal(
                    observations["proprio"][~selected, 33:45], delivered[~selected]
                )
            )
            self.assertTrue(
                torch.equal(
                    extras["final_observation"]["proprio"][:, 33:45],
                    delivered[selected],
                )
            )
            self.assertFalse(term.is_heading_env.any())
            self.assertFalse(term.is_standing_env.any())
            self.assertTrue(torch.isinf(term.time_left).all())
            fixture.next_terminated.zero_()
            counter = term.command_counter.clone()
            for index, command in enumerate(
                ((0.0, 0.0, 0.0), (0.3, 0.0, 0.4), (0.0, 0.0, -0.4)), start=2
            ):
                delivered = host.motor.encode(targets)
                _, _, terminated, truncated, _ = fixture.env.step(delivered)
                host.motor.verify_delivery(terminated, truncated)
                done = terminated | truncated
                targets = host.act(
                    torch.tensor(command).expand(4, 3),
                    time_s=index * 0.02,
                    reset_mask=done,
                )
                observations = fixture.observations()
                self.assertFalse(done.any())
                self.assertTrue(
                    torch.equal(
                        observations["proprio"][:, 6:9],
                        torch.tensor(command).expand(4, 3),
                    )
                )
                self.assertTrue(torch.equal(term.command_counter, counter))
                self.assertTrue(torch.isinf(term.time_left).all())
            native.forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()

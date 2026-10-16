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
from unittest.mock import Mock, patch

import parkour_lab.environments.traversal as terrain
import parkour_lab.runtime.native as runtime
import torch
from parkour_lab.runtime.training import TrainingHost
from parkour_lab.methods.roa.environment import ROAEnvironment

from tests.support.isaaclab import pinned_source_root
from tests.support.native import TrainingEnvironmentFixture


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
        yield NS(
            source=source,
            ManagerTermBase=base.ManagerTermBase,
            CommandTerm=manager.CommandTerm,
            UniformVelocityCommand=velocity.UniformVelocityCommand,
            forbidden=forbidden,
        )
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def inherited_command_config():
    """Nonzero archived sampling defaults that external ownership must replace."""
    return NS(
        class_type="training_20_column_sampler",
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=0.7,
        rel_standing_envs=0.2,
        ranges=NS(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(2.0, 3.0),
        debug_vis=True,
        untouched_source_field="retained",
    )


def traversal_config():
    return NS(
        commands=NS(base_velocity=inherited_command_config()),
        scene=NS(terrain=NS(terrain_generator=None, max_init_terrain_level=0)),
        events=NS(reset_base=NS(params={"pose_range": {}})),
        terminations=NS(
            procedural_physical_failure=NS(params={"minimum_surface_z_m": -1})
        ),
    )


class ExternalCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pinned_source_root()
        cls.fixtures = terrain.preflight(1043)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(143)
        receipts = patch.object(terrain, "_RECEIPTS", [])
        receipts.start()
        self.addCleanup(receipts.stop)

    def assert_external_config(self, cfg, command_class):
        self.assertIs(cfg.class_type, command_class)
        self.assertFalse(cfg.heading_command)
        self.assertEqual((cfg.rel_heading_envs, cfg.rel_standing_envs), (0.0, 0.0))
        self.assertIsNone(cfg.ranges.heading)
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            self.assertEqual(getattr(cfg.ranges, name), (0.0, 0.0))
        self.assertEqual(cfg.resampling_time_range, (1e9, 1e9))
        self.assertFalse(cfg.debug_vis)
        self.assertEqual(cfg.untouched_source_field, "retained")
        self.assertEqual(cfg.heading_control_stiffness, 0.5)
        self.assertEqual(cfg.asset_name, "robot")

    def configure_fixture(self, native):
        cfg = traversal_config()
        terrain.configure(cfg, 1043, self.fixtures)
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_cols, 4)
        self.assert_external_config(
            cfg.commands.base_velocity, native.UniformVelocityCommand
        )
        fixture = TrainingEnvironmentFixture(count=4)
        fixture.env.cfg.scene = cfg.scene
        fixture.env.scene["terrain"] = NS(
            terrain_types=torch.arange(4),
            terrain_levels=torch.zeros(4, dtype=torch.long),
        )
        term = cfg.commands.base_velocity.class_type(
            cfg.commands.base_velocity, fixture.env
        )
        self.assertIs(type(term), native.UniformVelocityCommand)
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

    def test_helper_accepts_injected_class_without_importing_simulator(self):
        cfg = inherited_command_config()
        with native_command_classes() as native:
            selected = native.UniformVelocityCommand
        # The source class survives module cleanup; this branch must not need
        # the lazy default native import at all.
        with patch.dict(sys.modules, {"isaaclab.envs.mdp.commands": None}):
            runtime.configure_external_command(cfg, command_class=selected)
        self.assert_external_config(cfg, selected)

    def test_traversal_config_constructs_real_command_on_four_columns(self):
        with native_command_classes() as native:
            fixture, term = self.configure_fixture(native)
            self.assertEqual(
                Path(inspect.getfile(type(term))),
                native.source / "envs/mdp/commands/velocity_command.py",
            )
            self.assertEqual(
                Path(inspect.getfile(native.CommandTerm)),
                native.source / "managers/command_manager.py",
            )
            metrics = term.reset(torch.arange(4))
            self.assertEqual(set(metrics), {"error_vel_xy", "error_vel_yaw"})
            self.assertTrue((term.command == 0).all())
            self.assertTrue((term.time_left == 1e9).all())
            self.assertTrue((term.command_counter == 1).all())
            self.assertFalse(term.is_heading_env.any())
            self.assertFalse(term.is_standing_env.any())
            self.assertEqual(
                fixture.env.cfg.scene.terrain.terrain_generator.num_cols, 4
            )
            native.forbidden.assert_not_called()

    def test_real_compute_preserves_signed_external_commands_without_resampling(self):
        with native_command_classes() as native:
            _, term = self.configure_fixture(native)
            term.reset(torch.arange(4))
            counter = term.command_counter.clone()
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

    def test_real_partial_reset_zeroes_only_reset_rows_then_resumes_external_tape(self):
        with native_command_classes() as native:
            fixture, term = self.configure_fixture(native)
            host = ROAEnvironment(TrainingHost(fixture.env, fixture.app))
            observations, reset = host.reset(seed=143, command=(0.4, 0.0, 0.5))
            self.assertTrue(reset.all())
            original_history = observations["history"].reshape(4, 25, 49).clone()
            self.assertTrue(
                torch.equal(original_history[:, -1], observations["policy"])
            )
            self.assertTrue(
                torch.equal(
                    original_history, observations["policy"][:, None].repeat(1, 25, 1)
                )
            )
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
            observations, _, done, _ = host.step(
                raw, "external_command_regression", next_command=next_command
            )
            self.assertTrue(torch.equal(done, selected))
            self.assertEqual(len(reset_commands), 1)
            self.assertTrue((reset_commands[0][selected] == 0).all())
            self.assertTrue(
                torch.equal(
                    reset_commands[0][~selected],
                    torch.tensor((0.4, 0.0, 0.5)).expand(2, 3),
                )
            )
            expected = torch.tensor(next_command).expand(4, 3)
            self.assertTrue(torch.equal(term.command, expected))
            self.assertTrue(torch.equal(observations["policy"][:, 6:9], expected))
            self.assertTrue(
                torch.equal(observations["critic_state"][:, 9:12], expected)
            )
            history = observations["history"].reshape(4, 25, 49)
            self.assertTrue(torch.equal(history[:, -1], observations["policy"]))
            self.assertTrue(
                torch.equal(
                    history[selected],
                    observations["policy"][selected, None].repeat(1, 25, 1),
                )
            )
            self.assertTrue(
                torch.equal(history[~selected, :-1], original_history[~selected, 1:])
            )
            self.assertTrue((observations["policy"][selected, 33:45] == 0).all())
            self.assertTrue(
                torch.equal(observations["policy"][~selected, 33:45], raw[~selected])
            )
            self.assertFalse(term.is_heading_env.any())
            self.assertFalse(term.is_standing_env.any())
            self.assertTrue(torch.isinf(term.time_left).all())
            fixture.next_terminated.zero_()
            counter = term.command_counter.clone()
            for command in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.4), (0.0, 0.0, -0.4)):
                observations, _, done, _ = host.step(
                    raw, "external_command_regression", next_command=command
                )
                self.assertFalse(done.any())
                self.assertTrue(
                    torch.equal(
                        observations["policy"][:, 6:9],
                        torch.tensor(command).expand(4, 3),
                    )
                )
                self.assertTrue(torch.equal(term.command_counter, counter))
                self.assertTrue(torch.isinf(term.time_left).all())
            native.forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()

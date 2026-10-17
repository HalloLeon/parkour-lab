"""Terminal capture around the pinned SDK step/reset code; no physics engine."""

import ast
import importlib.util
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from tests.support import SOURCE_ROOT
from tests.support.isaaclab import modules, pinned_source_root


class NativeTransitionTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        # Execute the installed SDK's real control/reset ordering with tiny
        # manager doubles, rather than reproducing that order in the fixture.
        source = pinned_source_root() / "envs/manager_based_rl_env.py"
        tree = ast.parse(source.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLEnv"
        )
        methods = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in ("step", "_reset_idx")
        ]
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *methods,
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
        self.base = type(
            "ManagerBasedRLEnv",
            (),
            {
                "__init__": lambda env: self.initialize(env),
                "close": lambda env: setattr(env, "closed", True),
                **{name: namespace[name] for name in ("step", "_reset_idx")},
            },
        )
        self.noise_cfg = type("NoiseCfg", (), {})
        self.stateful_term = type("ManagerTermBase", (), {})
        spec = importlib.util.spec_from_file_location(
            "native_transition_runtime",
            SOURCE_ROOT / "parkour_lab/environments/runtime.py",
        )
        module = importlib.util.module_from_spec(spec)
        with modules(
            {
                "isaaclab.envs": NS(ManagerBasedRLEnv=self.base),
                "isaaclab.managers": NS(ManagerTermBase=self.stateful_term),
                "isaaclab.utils.noise": NS(NoiseCfg=self.noise_cfg),
            }
        ):
            spec.loader.exec_module(module)
        self.environment = module.LocomotionEnv

    def initialize(self, env):
        self.env = env
        env.closed = False
        env.device, env.num_envs = "cpu", 4
        env.step_dt, env.physics_dt = 0.02, 0.005
        env.cfg = NS(decimation=4, sim=NS(render_interval=4), num_rerenders_on_reset=0)
        env._sim_step_counter = env.common_step_counter = 0
        env.episode_length_buf = torch.zeros(4, dtype=torch.long)
        env.extras = {}
        self.state = torch.arange(4, dtype=torch.float32)[:, None]
        self.command = torch.ones(4, 1)
        self.action = torch.zeros(4, 1)
        self.terminated = torch.tensor([True, False, True, False])
        self.truncated = torch.tensor([False, True, True, False])
        self.reads = []
        self.reset_noise = None

        def noop(*args, **kwargs):
            pass

        def reset(*args, **kwargs):
            return {}

        env.sim = NS(
            has_gui=lambda: False,
            has_rtx_sensors=lambda: False,
            step=lambda **kwargs: self.state.add_(1),
        )
        env.scene = NS(write_data_to_sim=noop, update=noop, reset=self.reset_scene)

        def reset_action(ids):
            self.action[ids] = 0
            return {}

        env.action_manager = NS(
            process_action=self.action.copy_, apply_action=noop, reset=reset_action
        )
        env.termination_manager = NS(
            compute=lambda: self.terminated | self.truncated,
            terminated=self.terminated,
            time_outs=self.truncated,
            reset=reset,
        )
        env.reward_manager = NS(
            compute=lambda **kwargs: self.state[:, 0].clone(), reset=reset
        )
        env.command_manager = NS(
            compute=lambda **kwargs: self.command.add_(10), reset=reset
        )
        env.curriculum_manager = NS(compute=noop, reset=reset)
        env.event_manager = NS(available_modes=[], reset=reset)
        env.recorder_manager = NS(
            active_terms=[],
            reset=reset,
            record_pre_step=noop,
            record_post_physics_decimation_step=noop,
            record_pre_reset=noop,
            record_post_reset=noop,
        )
        term = NS(history_length=0, modifiers=None, func=noop, noise=self.noise_cfg())
        group = NS(history_length=None, concatenate_terms=True, frame=term)
        env.observation_manager = NS(
            active_terms={"proprio": ["frame"]},
            cfg=NS(proprio=group),
            compute=self.observe,
            reset=reset,
        )

    def reset_scene(self, ids):
        self.state[ids] = -100
        self.command[ids] = -1
        self.reset_noise = torch.rand(len(ids))

    def observe(self, *, update_history=False):
        self.reads.append(update_history)
        return {
            "proprio": torch.cat(
                (self.state, self.command, self.action, torch.rand(4, 1)), dim=1
            )
        }

    def test_final_rows_precede_reset_commands_and_own_their_buffers(self):
        env = self.environment()
        action = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        observations, _, terminated, truncated, info = env.step(action)
        final = info["final_observation"]["proprio"]
        self.assertEqual(info["final_env_ids"].tolist(), [0, 1, 2])
        torch.testing.assert_close(final[:, 0], torch.tensor([4.0, 5.0, 6.0]))
        torch.testing.assert_close(final[:, 1], torch.ones(3))
        torch.testing.assert_close(final[:, 2], action[:3, 0])
        torch.testing.assert_close(
            observations["proprio"][:3, 0], torch.full((3,), -100.0)
        )
        self.assertEqual(self.reads, [False, True])
        self.assertEqual(
            env.transition_counts,
            dict(
                control_steps=1,
                terminated_rows=2,
                truncated_rows=2,
                simultaneous_rows=1,
                final_observation_rows=3,
            ),
        )
        snapshot = final.clone()
        self.terminated.zero_()
        self.truncated.zero_()
        _, _, _, _, following = env.step(action)
        self.assertIsNone(following["final_observation"])
        self.assertEqual(following["final_env_ids"].numel(), 0)
        torch.testing.assert_close(final, snapshot, rtol=0, atol=0)
        self.assertEqual(terminated.tolist(), [True, False, True, False])
        self.assertEqual(truncated.tolist(), [False, True, True, False])
        env._reset_idx(torch.arange(4))
        self.assertIsNone(env._final)
        self.assertFalse(env._capturing)
        self.assertEqual(self.reads, [False, True, True])
        self.assertEqual(env.interactions.session, 8)

    def test_terminal_sample_does_not_advance_reset_or_regular_observation_rng(self):
        action = torch.ones(4, 1)
        torch.manual_seed(974)
        plain = self.base()
        expected = plain.step(action)[0]["proprio"]
        reset_noise = self.reset_noise.clone()
        state = torch.random.get_rng_state()
        torch.manual_seed(974)
        env = self.environment()
        actual = env.step(action)[0]["proprio"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.reset_noise, reset_noise, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_training_information_owns_compact_outgoing_rows_before_reset(self):
        env = self.environment()
        env.cfg.benchmark_training = True
        env.capture_training_information = True
        # Return live buffers deliberately: the native hook must own its
        # selected terminal rows before the SDK mutates them during reset.
        with patch(
            "parkour_lab.benchmark.information.capture_information",
            side_effect=lambda _: {
                "physical_state": self.state,
                "command": self.command,
                "episode_length": env.episode_length_buf[:, None],
            },
        ) as capture:
            observations, _, terminated, truncated, info = env.step(torch.ones(4, 1))
            final = info["final_training_information"]
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(info["final_env_ids"].tolist(), [0, 1, 2])
            torch.testing.assert_close(
                final["physical_state"], torch.tensor([[4.0], [5.0], [6.0]])
            )
            torch.testing.assert_close(final["command"], torch.ones(3, 1))
            torch.testing.assert_close(
                final["episode_length"], torch.ones(3, 1, dtype=torch.long)
            )
            torch.testing.assert_close(
                observations["proprio"][:3, 0], torch.full((3,), -100.0)
            )
            self.assertEqual(env.episode_length_buf.tolist(), [0, 0, 0, 1])
            self.assertEqual(terminated.tolist(), [True, False, True, False])
            self.assertEqual(truncated.tolist(), [False, True, True, False])
            self.assertEqual(env.interactions.cumulative, 4)
            snapshot = {name: value.clone() for name, value in final.items()}
            self.state.fill_(500)
            self.command.fill_(700)
            self.terminated.zero_()
            self.truncated.zero_()
            following = env.step(torch.ones(4, 1))[-1]
            self.assertIsNone(following["final_training_information"])
            env._reset_idx(torch.arange(4))
            self.assertEqual(capture.call_count, 1)
        for name, value in final.items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        self.assertIsNone(env._final_information)
        self.assertEqual(env.interactions.cumulative, 8)

    def test_evaluation_cannot_enable_training_capture_with_runtime_flag(self):
        env = self.environment()
        env.cfg.benchmark_training = False
        env.capture_training_information = True
        with patch("parkour_lab.benchmark.information.capture_information") as capture:
            info = env.step(torch.ones(4, 1))[-1]
        capture.assert_not_called()
        self.assertNotIn("final_training_information", info)
        self.assertEqual(info["final_env_ids"].tolist(), [0, 1, 2])
        self.assertEqual(env.interactions.cumulative, 4)

    def test_training_capture_failure_keeps_completed_native_collection(self):
        env = self.environment()
        env.cfg.benchmark_training = True
        env.capture_training_information = True
        with patch(
            "parkour_lab.benchmark.information.capture_information",
            side_effect=RuntimeError("training capture failed"),
        ) as capture:
            with self.assertRaisesRegex(RuntimeError, "training capture failed"):
                env.step(torch.ones(4, 1))
            self.assertEqual(
                env.interactions.state_dict(),
                dict(session=4, cumulative=4, uncertain=0),
            )
            self.assertEqual(env.common_step_counter, 1)
            self.assertEqual(env._sim_step_counter, 4)
            self.assertEqual(env.transition_counts["control_steps"], 0)
            self.assertFalse(env._capturing)
            self.assertIsNone(env._final)
            self.assertIsNone(env._final_information)
            env._reset_idx(torch.arange(4))
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(env.interactions.cumulative, 4)

    def test_motion_capture_retains_terminal_physics_and_is_opt_in(self):
        env = self.environment()
        with patch("parkour_lab.runtime.native.motion_state") as capture:
            env.step(torch.ones(4, 1))
            capture.assert_not_called()
            env.capture_motion = True
            capture.side_effect = lambda _: {
                "position_w": self.state.clone(),
                "foot_net_forces_w": self.state[:, None].expand(-1, 4, 3).clone(),
            }
            _, _, _, _, info = env.step(torch.ones(4, 1))
            motion = info["motion_state"]["position_w"]
            torch.testing.assert_close(
                info["motion_state"]["foot_net_forces_w"][:, 0, 0], motion[:, 0]
            )
            torch.testing.assert_close(
                motion[:, 0], torch.tensor([-96.0, -96.0, -96.0, 11.0])
            )
            torch.testing.assert_close(
                self.state[:, 0], torch.tensor([-100.0, -100.0, -100.0, 11.0])
            )
            saved = motion.clone()
            self.terminated.zero_()
            self.truncated.zero_()
            following = env.step(torch.ones(4, 1))[-1]
            self.assertEqual(following["motion_state"]["position_w"][0, 0], -96)
            torch.testing.assert_close(saved, motion, rtol=0, atol=0)
            env._reset_idx(torch.arange(4))
            self.assertIsNone(env._final_physics)

    def test_contacts_follow_actual_sdk_physics_capture_and_reset_order(self):
        from parkour_lab.control.proprioception import FOOT_NAMES
        from parkour_lab.runtime.native import foot_contacts

        for dt, decimation in ((0.005, 4), (0.0025, 8)):
            with self.subTest(dt=dt):
                env = self.environment()
                env.physics_dt, env.cfg.decimation = dt, decimation
                env.cfg.sim.render_interval = decimation
                forces = torch.zeros(4, 4, 3)
                sensor = NS(
                    body_names=list(reversed(FOOT_NAMES)), data=NS(net_forces_w=forces)
                )
                scene_type = type(
                    "Scene", (NS,), {"__getitem__": lambda scene, key: sensor}
                )
                env.scene = scene_type(**vars(env.scene))
                physics_ticks = []

                def update(dt):
                    physics_ticks.append(dt)
                    forces[:, :, 0] = 2 if len(physics_ticks) % decimation == 0 else 0

                env.scene.update = update
                env.observation_manager.compute = lambda **kwargs: {
                    "proprio": foot_contacts(env)
                }
                self.assertTrue((foot_contacts(env) == 0).all())
                observations, _, _, _, info = env.step(torch.ones(4, 1))
                self.assertEqual(physics_ticks, [dt] * decimation)
                self.assertAlmostEqual(sum(physics_ticks), 0.02)
                self.assertTrue((info["final_observation"]["proprio"] == 1).all())
                self.assertTrue((observations["proprio"][:3] == 0).all())
                self.assertTrue((observations["proprio"][3] == 1).all())
                self.assertTrue((forces[:, :, 0] == 2).all())
                self.terminated.zero_()
                self.truncated.zero_()
                self.assertTrue((env.step(torch.ones(4, 1))[0]["proprio"] == 1).all())
                self.assertEqual(physics_ticks, [dt] * (2 * decimation))

    def test_capture_exception_clears_context_before_manual_reset(self):
        env = self.environment()

        def invalid(**kwargs):
            torch.rand(5)
            raise RuntimeError("sensor failed")

        env.observation_manager.compute = invalid
        state = torch.random.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "sensor failed"):
            env.step(torch.ones(4, 1))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertFalse(env._capturing)
        self.assertIsNone(env._final)
        self.assertEqual(
            env.interactions.state_dict(), dict(session=4, cumulative=4, uncertain=0)
        )
        env._reset_idx(torch.arange(4))
        self.assertEqual(env.interactions.cumulative, 4)

    def test_partial_physics_failure_charges_all_replicas_as_uncertain(self):
        env = self.environment()

        def fail(**kwargs):
            raise RuntimeError("physics failed during its first attempted tick")

        env.sim.step = fail
        with self.assertRaisesRegex(RuntimeError, "physics failed"):
            env.step(torch.ones(4, 1))
        self.assertEqual(
            env.interactions.state_dict(), dict(session=4, cumulative=4, uncertain=4)
        )

    def test_optional_survivor_capture_failure_keeps_completed_native_collection(self):
        env = self.environment()
        self.terminated.zero_()
        self.truncated.zero_()
        env.capture_motion = True
        with (
            patch(
                "parkour_lab.runtime.native.motion_state",
                side_effect=RuntimeError("capture failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            env.step(torch.ones(4, 1))
        self.assertEqual(
            env.interactions.state_dict(), dict(session=4, cumulative=4, uncertain=0)
        )

    def test_stateful_observation_processing_is_rejected(self):
        original = self.initialize
        for defect in ("group_history", "term_history", "modifiers", "noise", "term"):

            def initialize(env):
                original(env)
                group = env.observation_manager.cfg.proprio
                if defect == "group_history":
                    group.history_length = 2
                elif defect == "term_history":
                    group.frame.history_length = 2
                elif defect == "modifiers":
                    group.frame.modifiers = [object()]
                elif defect == "noise":
                    group.frame.noise = object()
                else:
                    group.frame.func = self.stateful_term()

            self.initialize = initialize
            with (
                self.subTest(defect=defect),
                self.assertRaisesRegex(ValueError, "stateless|adapter"),
            ):
                self.environment()
            self.assertTrue(self.env.closed)


if __name__ == "__main__":
    unittest.main()

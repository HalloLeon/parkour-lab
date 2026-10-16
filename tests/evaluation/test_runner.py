"""Connected first-attempt evidence through the existing frozen evaluator."""

import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from parkour_lab.evaluation.runner import evaluate
from parkour_lab.control.command_tape import validate_tape
from tests.support.native import make_native_env
from tests.support.isaaclab import modules


class ConnectedPlaybackTests(unittest.TestCase):
    def test_bound_attempt_archives_global_id_and_stops_after_first_ending(self):
        from parkour_lab.config import TaskConfig
        from parkour_lab.environments.randomization import TaskRandomization

        task = TaskConfig(
            terrain="connected",
            num_envs=1,
            episode_length_s=31,
            terrain_group="ramp/down/10deg",
            terrain_attempt_index=50,
        )
        env = make_native_env(1)
        env.cfg.seed = task.seed
        env.cfg.parkour_task = asdict(task)
        env.max_episode_length = 2
        env.scene["robot"].body_names = [
            "base",
            "FR_foot",
            "FL_foot",
            "RR_foot",
            "RL_foot",
        ]
        env.scene["height_scanner"] = NS(data=NS(ray_hits_w=torch.zeros(1, 132, 3)))
        env.scene.terrain = NS(
            world={
                "vertices": np.zeros((3, 3)),
                "faces": np.array([[0, 1, 2]]),
                "metadata": {},
            },
            metadata={},
            native_receipt={},
        )
        sampler = env.parkour_randomization = TaskRandomization(
            task, evaluation=True, nominal_heading=task.world["world_yaw"] - np.pi / 2
        )
        sampler.dynamics()
        env.reset = lambda **kwargs: (sampler.sample_start(None), {})
        controller = NS(spec=NS(sensors={}), state_sha256=lambda: "fixed")
        host = Mock(last_sensor_noise=None)
        host.motor.encode.return_value = torch.zeros(1, 12)
        host.motor.progress.return_value = {}
        report = {}
        commands = (
            [[0.0, 0.0, 0.0]] * 100
            + [[0.35, 0.0, 0.0]] * 1250
            + [[0.0, 0.0, 0.0]] * 150
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            modules(
                {
                    "parkour_lab.environments.dynamics": NS(
                        dynamics_report=lambda env: {
                            "randomization": sampler.manifest()
                        },
                        start_report=lambda env: {},
                    )
                }
            ),
            patch(
                "parkour_lab.evaluation.runner.NativeControllerSession",
                return_value=host,
            ),
        ):
            evaluate(
                env,
                NS(is_running=lambda: True),
                NS(controller=controller, motor_contract={}),
                commands,
                folder,
                report,
            )
            manifest = json.loads((Path(folder) / "manifest.json").read_text())
            tape = json.loads((Path(folder) / "commands.json").read_text())
            with np.load(Path(folder) / "tracking.npz") as tracking:
                self.assertEqual(set(tracking.files), {"root_com_velocity"})
        self.assertEqual(manifest["group_id"], task.terrain_group)
        self.assertEqual(manifest["attempts"][0]["attempt_index"], 50)
        self.assertEqual(manifest["attempts"][0]["stratum"], "randomized")
        self.assertEqual(
            report["terrain_assignment"]["streams"], manifest["attempts"][0]["streams"]
        )
        self.assertEqual(report["requested_control_steps"], 1500)
        self.assertEqual(report["control_steps"], 2)
        self.assertEqual(host.act.call_count, 2)
        self.assertFalse(tape["complete"])
        self.assertEqual(report["status"], "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED")

    def test_bound_attempt_rejects_wrong_tape_before_host_or_policy_consumption(self):
        from parkour_lab.config import TaskConfig

        task = TaskConfig(
            terrain="connected",
            num_envs=1,
            episode_length_s=31,
            terrain_group="ramp/down/10deg",
            terrain_attempt_index=50,
        )
        env = NS(cfg=NS(parkour_task=asdict(task)), num_envs=1, step_dt=0.02)
        with (
            patch("parkour_lab.evaluation.runner.NativeControllerSession") as host,
            self.assertRaisesRegex(ValueError, "assigned"),
        ):
            evaluate(env, None, None, [[0.0, 0.0, 0.0]] * 1500, None, {}, profile=None)
        host.assert_not_called()

    def test_complete_ending_and_error_preserve_distinct_first_attempt_evidence(self):
        for ending, missing_estimate in (
            (None, None),
            ("terminated", None),
            ("truncated", None),
            ("both", None),
            ("error", None),
            (None, 1),
        ):
            with (
                self.subTest(ending=ending, missing_estimate=missing_estimate),
                tempfile.TemporaryDirectory() as folder,
            ):
                env = make_native_env(1)
                env.cfg.seed = 42
                env.cfg.parkour_task = {"terrain": "connected"}
                robot = env.scene["robot"]
                robot.data.root_lin_vel_b[:] = torch.tensor([0.1, 0.2, 0.3])
                robot.data.root_ang_vel_b[:] = torch.tensor([4.0, 5.0, 6.0])
                robot.body_names = ["base", "FR_foot", "FL_foot", "RR_foot", "RL_foot"]
                env.scene["height_scanner"] = NS(
                    data=NS(ray_hits_w=torch.zeros(1, 132, 3))
                )
                env.scene.terrain = NS(
                    world={
                        "vertices": np.zeros((3, 3)),
                        "faces": np.array([[0, 1, 2]]),
                        "metadata": {},
                    },
                    metadata={"source": "synthetic"},
                    native_receipt={"native": "synthetic"},
                )
                env.reset = lambda **kwargs: ({}, {})
                original_step = env.step

                def step(action):
                    # Native cached tensors and the controller's buffer can change in place.
                    robot.data.root_lin_vel_b[:] = torch.tensor(
                        [env.index + 1.0, env.index + 2.0, env.index + 3.0]
                    )
                    if controller.last_estimate is not None:
                        controller.last_estimate.fill_(-777)
                    if env.index == 2 and ending == "error":
                        raise RuntimeError("injected step failure")
                    robot.data.root_pos_w.fill_(env.index + 1)
                    env.scene["contact_forces"].data.net_forces_w.fill_(env.index + 1)
                    values = list(original_step(action))
                    if env.index == 2 and ending in ("terminated", "truncated", "both"):
                        values[2].fill_(ending != "truncated")
                        values[3].fill_(ending != "terminated")
                        # Outgoing extras own their arrays before this SDK-style reset.
                        robot.data.root_pos_w.fill_(-99)
                        robot.data.root_lin_vel_b.fill_(-99)
                        env.scene["contact_forces"].data.net_forces_w.zero_()
                    return tuple(values)

                env.step = step
                controller = NS(spec=NS(sensors={}), state_sha256=lambda: "fixed")
                host = Mock(last_sensor_noise=None)

                def act(command, *, time_s, reset_mask):
                    controller.last_estimate = (
                        None
                        if env.index == missing_estimate
                        else torch.tensor(
                            [
                                [
                                    env.index + 10.0,
                                    env.index + 20.0,
                                    env.index + 30.0,
                                    99.0,
                                ]
                            ]
                        )
                    )
                    return "unchanged-targets"

                host.act.side_effect = act
                host.motor.encode.return_value = torch.zeros(1, 12)
                host.motor.progress.return_value = {"delivery": "synthetic"}
                report = {}
                with patch(
                    "parkour_lab.evaluation.runner.NativeControllerSession",
                    return_value=host,
                ):
                    if ending == "error":
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            evaluate(
                                env,
                                NS(is_running=lambda: True),
                                NS(controller=controller, motor_contract={}),
                                [(0, 0, 0)] * 5,
                                folder,
                                report,
                            )
                    else:
                        evaluate(
                            env,
                            NS(is_running=lambda: True),
                            NS(controller=controller, motor_contract={}),
                            [(0, 0, 0)] * 5,
                            folder,
                            report,
                        )
                tape = json.loads((Path(folder) / "commands.json").read_text())
                validate_tape(tape, require_complete=False)
                expected_steps = 5 if ending is None else 2
                self.assertEqual(tape["steps"], expected_steps)
                self.assertEqual(tape["complete"], ending is None)
                self.assertEqual(env.index, expected_steps)
                self.assertFalse(env.capture_motion)
                self.assertFalse(env.capture_diagnostics)
                with np.load(Path(folder) / "tracking.npz") as tracking:
                    expected_linear = np.array(
                        [
                            [[0.1, 0.2, 0.3]],
                            *[
                                [[step, step + 1, step + 2]]
                                for step in range(1, expected_steps)
                            ],
                        ],
                        dtype=np.float32,
                    )
                    expected_estimate = np.array(
                        [
                            [[step + 10, step + 20, step + 30]]
                            for step in range(expected_steps)
                        ],
                        dtype=np.float32,
                    )
                    if missing_estimate is not None:
                        expected_estimate[missing_estimate] = np.nan
                    np.testing.assert_array_equal(
                        tracking["root_com_linear_velocity_b"], expected_linear
                    )
                    np.testing.assert_array_equal(
                        tracking["estimated_linear_velocity_b"], expected_estimate
                    )
                    np.testing.assert_array_equal(
                        tracking["root_com_velocity"][:, :, :2],
                        expected_linear[:, :, :2],
                    )
                    np.testing.assert_array_equal(
                        tracking["root_com_velocity"][:, :, 2], 6
                    )
                self.assertEqual(
                    set(host.act.call_args.kwargs), {"time_s", "reset_mask"}
                )
                self.assertTrue(
                    torch.equal(env.action_manager.action, torch.zeros(1, 12))
                )
                with np.load(Path(folder) / "motion.npz") as motion:
                    self.assertEqual(len(motion["position_w"]), expected_steps)
                    np.testing.assert_array_equal(
                        motion["position_w"][-1], expected_steps
                    )
                    np.testing.assert_array_equal(
                        motion["foot_net_forces_w"][-1], expected_steps
                    )
                    self.assertIn("initial_base_surface_hits_w", motion)
                self.assertIn("world.npz", report["evidence_sha256"])
                if ending != "error":
                    self.assertEqual(report["control_steps"], expected_steps)
                    self.assertEqual(report["requested_control_steps"], 5)
                    self.assertTrue(report["policy_unchanged"])
                    self.assertEqual(
                        report["status"],
                        "EVALUATION_COMPLETE_NOT_QUALIFIED"
                        if ending is None
                        else "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED",
                    )


if __name__ == "__main__":
    unittest.main()

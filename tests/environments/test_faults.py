"""Deterministic substep scheduling and private evaluator drive-loss readback."""

import json
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch

from parkour_lab.environments.faults import (
    JOINT_NAMES,
    DriveFault,
    FaultConfig,
    configure_fault,
    fault_state,
    reset_fault,
)
from tests.environments.test_dynamics import environment, native_dynamics


class FaultTests(unittest.TestCase):
    def test_config_roundtrip_and_rejected_ambiguous_values(self):
        cfg = FaultConfig("FR_calf_joint", 0, 8)
        self.assertEqual(FaultConfig(**json.loads(json.dumps(cfg.as_dict()))), cfg)
        self.assertIs(type(cfg.factor), float)
        for values in (
            ("", 0.5, 8),
            (None, 0.5, 8),
            ("joint", True, 8),
            ("joint", 0.7, 8),
            ("joint", float("nan"), 8),
            ("joint", 0.5, -1),
            ("joint", 0.5, float("inf")),
            ("joint", 0.5, True),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FaultConfig(*values)

    def test_interval_start_clock_persistence_and_partial_reset(self):
        clock = NS(step=0)
        fault = DriveFault(
            FaultConfig("j1", 0.5, 0.01),
            ("j0", "j1"),
            num_envs=2,
            device="cpu",
            physics_dt=0.005,
            clock=lambda: clock.step,
        )
        effort = torch.tensor([[20.0, -12.0], [8.0, 16.0]])
        self.assertEqual(fault.onset_step, 2)
        for step in (0, 0, 1, 2):
            clock.step = step
            torch.testing.assert_close(fault.apply(effort, [0, 1]), effort)
        clock.step = 3
        expected = effort * torch.tensor([1.0, 0.5])
        torch.testing.assert_close(fault.apply(effort, [0, 1]), expected)
        torch.testing.assert_close(fault.apply(effort, [0, 1]), expected)
        fault.reset(torch.tensor([0]))
        for step in (3, 4, 5):
            clock.step = step
            result = fault.apply(effort, [0, 1])
            torch.testing.assert_close(result[0], effort[0])
            torch.testing.assert_close(result[1], expected[1])
        clock.step = 6
        torch.testing.assert_close(fault.apply(effort, [0, 1]), expected)
        self.assertEqual(fault.reset_steps.tolist(), [3, 0])

    def test_quantized_onset_and_no_random_draws(self):
        before_numpy, before_torch = np.random.get_state(), torch.random.get_rng_state()
        fault = DriveFault(
            FaultConfig("j", 0, 0.07),
            ("j",),
            num_envs=1,
            device="cpu",
            physics_dt=0.005,
            clock=lambda: 14,
        )
        self.assertEqual(fault.onset_step, 14)
        self.assertAlmostEqual(fault.effective_onset_s, 0.07)
        fault.apply(torch.ones(1, 1), [0])
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before_numpy[1], after[1]))
        self.assertEqual(before_numpy[2:], after[2:])
        self.assertTrue(torch.equal(before_torch, torch.random.get_rng_state()))
        for ids in ([0, 0], [-1], [1], [0.0], [[0]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                fault.reset(ids)

    def test_configuration_binding_canonical_readback_and_no_fault_fast_path(self):
        self.assertEqual(fault_state(NS()), {})
        configure_fault(NS(cfg=NS()))
        reset_fault(NS(), torch.tensor([0]))
        with native_dynamics() as module:
            original, motor, _, _ = environment(module)

            class Env(NS):
                pass

            env = Env(**vars(original), physics_dt=0.005, _sim_step_counter=0)
            native_names = tuple(
                f"{leg}_{joint}_joint"
                for joint in ("hip", "thigh", "calf")
                for leg in ("FL", "FR", "RL", "RR")
            )
            env.scene["robot"].joint_names = native_names
            motor._joint_names = native_names
            env.cfg.benchmark_training = False
            env.cfg.benchmark_fault = FaultConfig("FR_calf_joint", 0.5, 0.01).as_dict()
            configure_fault(env)
            self.assertIs(motor.drive_fault, env.parkour_fault)
            env._sim_step_counter = 3
            motor.applied_effort = motor._clip_effort(
                torch.arange(1, 13).expand(4, -1).float()
            )
            state = fault_state(env)
            self.assertEqual(
                state["drive_factor"][0].tolist(),
                [1 if name != "FR_calf_joint" else 0.5 for name in JOINT_NAMES],
            )
            order = [native_names.index(name) for name in JOINT_NAMES]
            torch.testing.assert_close(
                state["applied_effort"], motor.applied_effort[:, order]
            )
            frozen = {key: value.clone() for key, value in state.items()}
            reset_fault(env, torch.tensor([0]))
            self.assertEqual(
                env.parkour_fault.fault_active.tolist(), [False, True, True, True]
            )
            self.assertTrue((env.parkour_fault.drive_factor[0] == 1).all())
            for key in state:
                self.assertTrue(torch.equal(state[key], frozen[key]))
            with self.assertRaisesRegex(RuntimeError, "already configured"):
                configure_fault(env)

    def test_split_motor_groups_share_one_clock_and_map_named_effort(self):
        with native_dynamics() as module:
            original, motor, _, _ = environment(module)

            class Env(NS):
                pass

            env = Env(**vars(original), physics_dt=0.005, _sim_step_counter=0)
            robot = env.scene["robot"]
            names = tuple(reversed(JOINT_NAMES))
            robot.joint_names = names
            robot.actuators = {
                str(offset): module.RandomizedDCMotor(
                    motor.cfg, names[offset::2], list(range(offset, 12, 2)), 4, "cpu"
                )
                for offset in (0, 1)
            }
            env.cfg.benchmark_training = False
            env.cfg.benchmark_fault = FaultConfig("FL_hip_joint", 0, 0.005).as_dict()
            configure_fault(env)
            for step in (0, 1, 2):
                env._sim_step_counter = step
                for actuator in robot.actuators.values():
                    actuator.applied_effort = actuator._clip_effort(torch.ones(4, 6))
                expected_active = step == 2
                self.assertEqual(
                    env.parkour_fault.fault_active.tolist(), [expected_active] * 4
                )
            state = fault_state(env)
            self.assertTrue((state["applied_effort"][:, 0] == 0).all())
            self.assertTrue((state["applied_effort"][:, 1:] == 1).all())
            self.assertTrue((state["limited_effort"] == 1).all())

    def test_training_fault_and_unknown_resolved_joint_are_rejected(self):
        with native_dynamics() as module:
            original, _, _, _ = environment(module)

            class Env(NS):
                pass

            env = Env(**vars(original), physics_dt=0.005, _sim_step_counter=0)
            env.cfg.benchmark_fault = FaultConfig("missing", 0, 8).as_dict()
            env.cfg.benchmark_training = True
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                configure_fault(env)
            env.cfg.benchmark_training = False
            with self.assertRaisesRegex(ValueError, "resolved motor"):
                configure_fault(env)


if __name__ == "__main__":
    unittest.main()

"""Bounded delivery/feedback and explicit artifact conversion; CPU, not physics."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import torch

from parkour_lab import artifacts
from parkour_lab.__main__ import parse_args, resolve_config
from parkour_lab.config import ExperimentConfig, MethodConfig, TaskConfig
from parkour_lab.control.action import BOUNDED_MODE, JointActionTransform
from parkour_lab.control.controller import ControllerSession
from parkour_lab.methods import roa
from parkour_lab.methods.roa.runtime import ROAHistoryController, roa_tensor_sha256
from parkour_lab.runtime.motor import NativeJointTargetBridge
from parkour_lab.runtime.training import TrainingHost
from tests.support.native import TrainingEnvironmentFixture
from tests.support.roa import packet


def fixture_with_limits():
    fixture = TrainingEnvironmentFixture(count=4)
    default = fixture.data.default_joint_pos
    width = torch.linspace(0.4, 2.0, 12)
    mechanical = torch.stack((default - width, default + 1.2 * width), dim=-1)
    center = (mechanical[..., 0] + mechanical[..., 1]) / 2
    half = 0.5 * (mechanical[..., 1] - mechanical[..., 0]) * 0.9
    fixture.data.soft_joint_pos_limits = torch.stack((center - half, center + half), -1)
    fixture.data.joint_limits = mechanical
    return fixture


def tiny_options():
    return MethodConfig(
        options={
            "rollout_steps": 2,
            "history_steps": 2,
            "history_interval": 1,
            "adaptation_epochs": 1,
            "adaptation_batches": 2,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "contact_conditioned": False,
        }
    )


class ActionBoundaryTests(unittest.TestCase):
    def test_extreme_requests_reach_bounded_targets_feedback_and_action_rate(self):
        fixture = fixture_with_limits()
        original_step = fixture.env.step
        fixture.env.reward_manager.active_terms = ["action_rate_l2", "unused"]

        def step(action):
            previous = fixture.env.action_manager.action.clone()
            result = original_step(action)
            rate = -0.01 * (action - previous).square().sum(-1)
            fixture.env.reward_manager._step_reward[:, 0] = rate
            fixture.env.reward_manager._step_reward[:, 1] = 0
            return result[0], rate * fixture.env.step_dt, *result[2:]

        fixture.env.step = step
        host = TrainingHost(fixture.env, fixture.app, action_mode=BOUNDED_MODE)
        host.reset()
        limits = fixture.data.soft_joint_pos_limits
        delivered_previous = torch.zeros(4, 12)
        for index, sign in enumerate((1, -1)):
            raw = torch.full((4, 12), sign * 4.46e8)
            preserved = raw.clone()
            if index:
                fixture.next_terminated = torch.tensor([True, False, False, False])
            native, reward, done, extras = host.step(raw, "boundary")
            delivered = fixture.deliveries[-1]
            target = fixture.data.joint_pos_target
            self.assertTrue(torch.equal(raw, preserved))
            self.assertTrue((target >= limits[..., 0]).all())
            self.assertTrue((target <= limits[..., 1]).all())
            self.assertLess(float(delivered.abs().max()), 20)
            expected_reward = (
                -0.01 * (delivered - delivered_previous).square().sum(-1) * 0.02
            )
            torch.testing.assert_close(reward, expected_reward, rtol=0, atol=0)
            self.assertLess(float(reward.abs().max()), 2)
            self.assertTrue(
                torch.equal(native["proprio"][~done, 33:45], delivered[~done])
            )
            self.assertTrue((native["proprio"][done, 33:45] == 0).all())
            if done.any():
                self.assertTrue(
                    torch.equal(
                        extras["final_observation"]["proprio"][:, 33:45],
                        delivered[done],
                    )
                )
            delivered_previous = delivered.clone()
        report = host.bridge.progress()["action_boundary"]
        self.assertEqual(
            report["max_requested_action_abs"], float(preserved.abs().max())
        )
        self.assertEqual(report["saturated_joint_commands"], 96)
        self.assertEqual(report["saturated_row_commands"], 8)
        self.assertEqual(report["total_joint_commands"], 96)

    def test_ppo_keeps_raw_samples_and_exact_likelihood_replay(self):
        fixture = fixture_with_limits()
        host = TrainingHost(fixture.env, fixture.app, action_mode=BOUNDED_MODE)
        with contextlib.redirect_stdout(io.StringIO()):
            learner = roa.create(host, tiny_options().options, 601)
        with torch.no_grad():
            motor = learner.method.policy.actor.motor
            motor.tail[-1].weight.zero_()
            motor.tail[-1].bias.fill_(4.46e8)
        learner.method.collect(learner.environment, learner.observations)
        storage = learner.method.algorithm.storage
        self.assertTrue((storage.actions == 4.46e8).all())
        self.assertLess(float(torch.stack(fixture.deliveries).abs().max()), 20)
        self.assertTrue(
            torch.equal(
                storage.observations[1]["policy"][:, 33:45], fixture.deliveries[0]
            )
        )
        self.assertEqual(
            learner.method.algorithm.verify_first_replay()["ratio_min"], 1.0
        )

    def test_frozen_controller_matches_training_delivery_and_native_limits(self):
        fixture = fixture_with_limits()
        host = TrainingHost(fixture.env, fixture.app, action_mode=BOUNDED_MODE)
        with contextlib.redirect_stdout(io.StringIO()):
            learner = roa.create(host, tiny_options().options, 602)
        actor = learner.method.policy.actor
        with torch.no_grad():
            actor.motor.tail[-1].weight.zero_()
            actor.motor.tail[-1].bias.copy_(torch.linspace(-4.46e8, 4.46e8, 12))
        controller = ROAHistoryController(
            actor.motor,
            actor.estimator,
            joint_names=host.manifest["joint_names"],
            default_position_rad=fixture.data.default_joint_pos[0],
            artifact_sha256=roa_tensor_sha256(actor.motor, actor.estimator),
            actuator_profile=host.manifest["actuator_profile"],
            target_limits_rad=host.motor_contract["target_limits_rad"],
        )
        session = ControllerSession(
            controller,
            joint_names=host.bridge.joint_names,
            actuator_profile=host.manifest["actuator_profile"],
        )
        observations = learner.observations
        reset = torch.ones(4, dtype=torch.bool)
        for step in range(2):
            result = session.step(
                **packet(controller, observations["policy"], step, reset)
            )
            expected_raw = actor.history_action(
                observations["policy"], observations["history"].reshape(4, 25, 49)
            ).detach()
            self.assertTrue(torch.equal(result.raw_action, expected_raw))
            observations, _, reset, _ = learner.environment.step(expected_raw, "parity")
            self.assertTrue(
                torch.equal(result.position_rad, fixture.data.joint_pos_target)
            )
        fixture.data.soft_joint_pos_limits[:, 0, 1] += 0.01
        with self.assertRaisesRegex(ValueError, "native soft joint limits"):
            NativeJointTargetBridge(
                fixture.env, host.motor_contract, controller.spec.manifest()
            )

    def test_legacy_transform_remains_exact_and_unbounded(self):
        default = torch.linspace(-0.3, 0.3, 12)
        raw = torch.full((4, 12), 4.46e8)
        delivered, target = JointActionTransform(default)(raw)
        self.assertTrue(torch.equal(delivered, raw))
        self.assertTrue(torch.equal(target, default + 0.25 * raw))


class ActionArtifactTests(unittest.TestCase):
    def test_conversion_preserves_learner_and_export_then_rejects_implicit_override(
        self,
    ):
        fixture = fixture_with_limits()
        host = TrainingHost(fixture.env, fixture.app)
        config = ExperimentConfig(
            task=TaskConfig(num_envs=4, device="cpu"), method=tiny_options()
        )
        with contextlib.redirect_stdout(io.StringIO()):
            learner = roa.create(host, config.method.options, 603)
            learner.advance()
        state = learner.state_dict()
        self.assertTrue(state["ppo_optimizer"]["state"])
        self.assertTrue(state["adaptation_optimizer"]["state"])
        controller = roa.controller(
            roa.export(state, config.method.options),
            config.method.options,
            host.manifest,
            "cpu",
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, bounded, actor = (
                folder / name for name in ("source.plab", "bounded.plab", "actor.plab")
            )
            artifacts.save_artifact(
                source,
                kind="training",
                config=config,
                state=state,
                motor_contract=host.motor_contract,
                motor_manifest=controller.spec.manifest(),
                updates=1,
            )
            report = folder / "report.json"
            realization = {
                "joint_names": fixture.robot.joint_names,
                "physical_readback": {
                    "joint_limits": fixture.data.joint_limits.tolist()
                },
            }
            report.write_text(json.dumps({"task_realization": realization}))
            source_bytes = source.read_bytes()
            receipt = artifacts.bound_actions(source, bounded, report=report)
            self.assertTrue(receipt["learning_payload_unchanged"])
            self.assertFalse(receipt["qualified"])
            self.assertEqual(source.read_bytes(), source_bytes)
            with ZipFile(source) as old, ZipFile(bounded) as new:
                self.assertEqual(old.read("payload"), new.read("payload"))
            data = artifacts.load_artifact(bounded, kind="training")
            self.assertEqual(data["updates"], 1)
            self.assertEqual(data["config"]["method"], config.to_dict()["method"])
            self.assertEqual(
                ExperimentConfig.from_dict(data["config"]).task, config.task
            )
            self.assertEqual(data["config"]["action_mode"], BOUNDED_MODE)
            torch.testing.assert_close(data["state"], state, rtol=0, atol=0)
            artifacts.export_actor(bounded, actor)
            loaded = artifacts.load_actor(actor)
            self.assertEqual(
                loaded.controller.state_sha256(), controller.state_sha256()
            )
            self.assertEqual(
                data["motor_manifest"],
                json.loads(json.dumps(loaded.controller.spec.manifest())),
            )
            NativeJointTargetBridge(
                fixture.env, loaded.motor_contract, loaded.controller.spec.manifest()
            )
            override = folder / "override.json"
            for checkpoint, mode in ((source, BOUNDED_MODE), (bounded, "unbounded")):
                override.write_text(json.dumps({"action_mode": mode}))
                args = parse_args(
                    [
                        "train",
                        "--checkpoint",
                        str(checkpoint),
                        "--config",
                        str(override),
                    ]
                )
                with self.assertRaisesRegex(ValueError, "use bound-actions"):
                    resolve_config(args)
            wrong = copy.deepcopy(realization)
            wrong["joint_names"].reverse()
            report.write_text(json.dumps({"task_realization": wrong}))
            with self.assertRaisesRegex(ValueError, "joint order"):
                artifacts.bound_actions(source, folder / "wrong.plab", report=report)
            self.assertFalse((folder / "wrong.plab").exists())


if __name__ == "__main__":
    unittest.main()

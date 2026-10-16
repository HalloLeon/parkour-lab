"""Native math and observation-boundary tests; no PhysX/contact-generation claim."""

import ast
import contextlib
import math
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from tensordict import TensorDict

import parkour_lab.methods.roa.contacts as contacts
from parkour_lab.methods.roa.environment import ROAEnvironment
from parkour_lab.runtime.training import TrainingHost
from tests.support.isaaclab import pinned_source_root
from tests.support.native import TrainingEnvironmentFixture


@contextlib.contextmanager
def native_inverse_rotation():
    """Run the pinned Isaac Lab inverse rotation, not a replacement formula."""
    path = pinned_source_root() / "utils/math.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) == "quat_apply_inverse"
    )
    # CPU execution uses the unmodified body without a JIT import dependency.
    function.decorator_list = []
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    with patch.dict(
        sys.modules,
        {"isaaclab.utils.math": NS(quat_apply_inverse=namespace["quat_apply_inverse"])},
    ):
        yield


def attach_contacts(native):
    """Distinct sensor/robot ordering and stale reset buffers mimic native APIs."""
    native.robot.body_names = ["base", *reversed(contacts.FEET)]
    names = ["FR_foot", "unused_body", "RR_foot", "FL_foot", "RL_foot"]
    forces = torch.arange(native.env.num_envs * 15, dtype=torch.float32).reshape(
        -1, 5, 3
    )
    native.env.scene["contact_forces"] = NS(
        body_names=names, data=NS(net_forces_w=forces)
    )
    native.data.root_link_quat_w = torch.tensor([1.0, 0, 0, 0]).repeat(
        native.env.num_envs, 1
    )
    # A different alias detects accidental use of root COM rather than link data.
    native.data.root_quat_w = torch.tensor([0.0, 0, 0, 1]).repeat(
        native.env.num_envs, 1
    )
    return forces


class ContactFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.native = TrainingEnvironmentFixture()
        self.forces = attach_contacts(self.native)
        self.rotation = native_inverse_rotation()
        self.rotation.__enter__()
        self.addCleanup(self.rotation.__exit__, None, None, None)
        self.reset = torch.zeros(8, dtype=torch.bool)

    def test_named_sensor_order_body_inverse_and_no_source_aliasing(self):
        native = self.native
        native.data.root_link_quat_w[1] = torch.tensor(
            [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
        )
        native.data.root_link_quat_w[2] = torch.tensor(
            [math.sqrt(0.5), 0, math.sqrt(0.5), 0]
        )
        original = self.forces.clone()
        state = contacts.ContactFeatures(native.env)
        self.assertEqual(state.ids, [3, 0, 4, 2])
        expected = original[:, state.ids].clone()
        world = expected.clone()
        expected[1] = torch.stack((world[1, :, 1], -world[1, :, 0], world[1, :, 2]), -1)
        expected[2] = torch.stack((-world[2, :, 2], world[2, :, 1], world[2, :, 0]), -1)
        value = state.sample(self.reset)
        torch.testing.assert_close(
            value, torch.tanh(expected.flatten(1) / 100), atol=2e-7, rtol=0
        )
        self.assertEqual(value.shape, (8, 12))
        self.assertFalse(value.requires_grad)
        value.zero_()
        self.assertTrue(torch.equal(self.forces, original))
        self.assertEqual(state.report()["observation_calls"], 1)

    def test_reset_masks_stale_forces_without_mutating_sensor_or_other_rows(self):
        self.reset[[1, 5]] = True
        self.forces.requires_grad_(True)
        before = self.forces.detach().clone()
        state = contacts.ContactFeatures(self.native.env)
        result = state.sample(self.reset)
        expected = torch.tanh(before[:, state.ids].flatten(1) / 100)
        expected[self.reset] = 0
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.equal(self.forces, before))
        self.assertEqual(state.report()["reset_rows_zeroed"], 2)
        self.assertEqual(state.report()["nonzero_rows"], 6)
        self.assertLessEqual(state.report()["maximum_absolute_feature"], 1)

    def test_missing_or_duplicated_named_feet_reject(self):
        for owner in (self.native.robot, self.native.env.scene["contact_forces"]):
            original = owner.body_names
            for names in (
                [name for name in original if name != "FL_foot"],
                original + ["FL_foot"],
            ):
                with self.subTest(owner=owner, names=names):
                    owner.body_names = names
                    with self.assertRaisesRegex(ValueError, "named teacher foot"):
                        contacts.ContactFeatures(self.native.env)
            owner.body_names = original

    def test_invalid_force_quaternion_or_reset_observation_rejects_before_counting(
        self,
    ):
        state = contacts.ContactFeatures(self.native.env)
        data = self.native.data
        sensor = self.native.env.scene["contact_forces"].data
        for field, value in (
            ("force", torch.zeros(7, 5, 3)),
            ("force", torch.full((8, 5, 3), float("nan"))),
            ("force", torch.full((8, 5, 3), float("inf"))),
            ("quat", torch.zeros(8, 4)),
            ("quat", torch.ones(8, 4)),
            ("quat", torch.full((8, 4), float("nan"))),
            ("quat", torch.zeros(8, 3)),
            ("reset", torch.zeros(8)),
            ("reset", torch.zeros(7, dtype=torch.bool)),
        ):
            with self.subTest(field=field, shape=tuple(value.shape)):
                sensor.net_forces_w = self.forces
                data.root_link_quat_w = torch.tensor([1.0, 0, 0, 0]).repeat(8, 1)
                reset = self.reset
                if field == "force":
                    sensor.net_forces_w = value
                elif field == "quat":
                    data.root_link_quat_w = value
                else:
                    reset = value
                with self.assertRaisesRegex(
                    ValueError, "Invalid native teacher contact"
                ):
                    state.sample(reset)
                self.assertEqual(state.calls, 0)

    def test_teacher_contacts_disabled_still_requires_causal_foot_indicators(self):
        host = ROAEnvironment(TrainingHost(self.native.env, self.native.app))
        observation, _ = host.reset()
        self.assertIsInstance(observation, TensorDict)
        self.assertIsNone(host.contacts)
        self.assertNotIn("contacts", observation)
        self.assertEqual(observation["policy"].shape, (8, 49))

    def test_host_partial_reset_and_history_keep_contact_channel_privileged(self):
        host = ROAEnvironment(
            TrainingHost(self.native.env, self.native.app), contact_conditioned=True
        )
        obs, reset = host.reset()
        self.assertTrue(reset.all())
        self.assertTrue((obs["contacts"] == 0).all())
        for _ in range(30):
            previous_history = host.history.frames.clone()
            self.forces.add_(1)
            before = self.forces.clone()
            raw = torch.full((8, 12), 0.2)
            obs, _, done, _ = host.step(raw, "history_adaptation")
            expected = torch.tanh(before[:, host.contacts.ids].flatten(1) / 100)
            expected[done] = 0
            torch.testing.assert_close(obs["contacts"], expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(self.forces, before))
            history = obs["history"].reshape(8, 25, 49)
            self.assertTrue(torch.equal(history[:, -1], obs["policy"]))
            self.assertTrue(
                torch.equal(history[~done, :-1], previous_history[~done, 1:])
            )
            self.assertTrue(
                torch.equal(history[done], obs["policy"][done, None].repeat(1, 25, 1))
            )
            self.assertEqual(obs["policy"].shape, (8, 49))
            self.assertEqual(obs["history"].shape, (8, 1225))
            self.assertTrue((history[done, :, 45:] == 0).all())
            self.assertTrue((obs["policy"][~done, 45:] == 1).all())
            self.assertEqual(obs["dynamics"].shape, (8, 7))
        self.assertGreater(host.host.partial_reset_steps, 0)
        self.assertEqual(host.contacts.calls, 31)


if __name__ == "__main__":
    unittest.main()

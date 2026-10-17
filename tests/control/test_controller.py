"""Causal controller boundary: timestamps, privilege, row validity and ownership."""

import unittest
from dataclasses import replace

import torch

from parkour_lab.control.controller import (
    ControllerSession,
    ControllerSpec,
    JointTargets,
    Sample,
    SensorSpec,
)


class Actor:
    def __init__(self):
        self.spec = ControllerSpec(
            "fixture",
            "a" * 64,
            "frame-v1",
            ("left", "right"),
            0.02,
            "motor",
            {
                "required": SensorSpec((2,), "rad", "joint", 0.02),
                "optional": SensorSpec((1,), "unitless", "body", 0.02, required=False),
            },
            "private diagnostic vector",
            {"nested": [1, 2]},
        )
        self.memory = torch.ones(2, 1)
        self.inputs, self.resets = [], []
        self.output = JointTargets(
            self.spec.joint_names, torch.zeros(2, 2), torch.ones(2, 1)
        )

    def reset(self, mask):
        self.resets.append(mask.clone())
        self.memory[mask] = 0
        mask.zero_()

    def act(self, inputs):
        self.inputs.append(inputs)
        self.memory += 1
        return self.output


def decision(time=0.0, *, reset=None):
    return dict(
        time_s=time,
        command=torch.ones(2, 3),
        command_time_s=time,
        sensors={
            "required": Sample(
                torch.ones(2, 2), time, torch.ones(2, dtype=torch.bool), "rad", "joint"
            ),
            "optional": Sample(
                torch.tensor([[5.0], [99.0]]),
                time,
                torch.tensor([True, False]),
                "unitless",
                "body",
            ),
        },
        reset_mask=(
            torch.full((2,), time == 0, dtype=torch.bool) if reset is None else reset
        ),
    )


def session(actor):
    return ControllerSession(
        actor, joint_names=actor.spec.joint_names, actuator_profile="motor"
    )


class ControllerSessionTests(unittest.TestCase):
    def test_projected_inputs_and_outputs_own_storage_and_invalid_optional_rows_are_zero(
        self,
    ):
        actor, values = Actor(), decision()
        values["sensors"]["oracle"] = replace(
            values["sensors"]["required"], privileged=True
        )
        host = session(actor)
        output = host.step(**values)
        received = actor.inputs[0]
        self.assertEqual(set(received.sensors), {"required", "optional"})
        torch.testing.assert_close(
            received.sensors["optional"].value, torch.tensor([[5.0], [0.0]])
        )
        self.assertEqual(values["sensors"]["optional"].value[1, 0], 99)
        with torch.inference_mode():
            received.command.zero_()
            received.sensors["required"].value.zero_()
            received.sensors["required"].valid.zero_()
        self.assertTrue(values["command"].all())
        self.assertTrue(values["sensors"]["required"].value.all())
        self.assertTrue(values["sensors"]["required"].valid.all())
        self.assertTrue(values["reset_mask"].all())
        actor.output.position_rad.fill_(9)
        actor.output.raw_action.zero_()
        self.assertFalse(output.position_rad.any())
        self.assertTrue(output.raw_action.all())
        with self.assertRaises(TypeError):
            received.sensors["oracle"] = None

    def test_episode_memory_resets_only_explicit_rows_before_inference(self):
        actor = Actor()
        host = session(actor)
        host.step(**decision())
        host.step(**decision(0.02))
        host.step(**decision(0.04, reset=torch.tensor([False, True])))
        torch.testing.assert_close(actor.memory, torch.tensor([[3.0], [1.0]]))
        torch.testing.assert_close(
            host.reset_at_s, torch.tensor([0.0, 0.04], dtype=torch.float64)
        )
        self.assertTrue(actor.resets[0].all())
        self.assertFalse(actor.resets[1].any())

    def test_privileged_specs_and_declared_provider_samples_always_reject(self):
        actor = Actor()
        actor.spec = replace(
            actor.spec,
            sensors={"oracle": SensorSpec((2,), "rad", "joint", 0.02, privileged=True)},
        )
        with self.assertRaisesRegex(ValueError, "Privileged sensor"):
            session(actor)
        actor, values = Actor(), decision()
        values["sensors"]["required"] = replace(
            values["sensors"]["required"], privileged=True
        )
        host = session(actor)
        with self.assertRaisesRegex(ValueError, "Privileged provider"):
            host.step(**values)
        self.assertTrue(host.faulted)
        self.assertFalse(actor.inputs)

    def test_invalid_inputs_latch_without_inference_or_implicit_recovery(self):
        mutations = (
            lambda row: row.update(command_time_s=-0.03),
            lambda row: row.update(command_time_s=0.01),
            lambda row: row.update(reset_mask=torch.tensor([True, False])),
            lambda row: row["sensors"].pop("required"),
            lambda row: row["sensors"].update(
                required=replace(row["sensors"]["required"], units="deg")
            ),
            lambda row: row["sensors"].update(
                required=replace(row["sensors"]["required"], frame="world")
            ),
            lambda row: row["sensors"].update(
                required=replace(row["sensors"]["required"], acquired_at_s=-0.03)
            ),
            lambda row: row["sensors"].update(
                required=replace(
                    row["sensors"]["required"], valid=torch.tensor([True, False])
                )
            ),
        )
        for mutate in mutations:
            actor, values = Actor(), decision()
            mutate(values)
            host = session(actor)
            with self.assertRaises(ValueError):
                host.step(**values)
            self.assertTrue(host.faulted)
            self.assertFalse(actor.inputs)
            self.assertFalse(actor.resets)
            with self.assertRaisesRegex(RuntimeError, "faulted"):
                host.step(**decision())

    def test_sensor_cannot_predate_new_episode_even_when_within_age_limit(self):
        actor = Actor()
        host = session(actor)
        host.step(**decision())
        values = decision(0.02, reset=torch.tensor([False, True]))
        values["sensors"]["required"] = replace(
            values["sensors"]["required"], acquired_at_s=0.0
        )
        with self.assertRaisesRegex(ValueError, "predates episode reset"):
            host.step(**values)
        self.assertEqual(len(actor.inputs), 1)

    def test_changed_cadence_batch_dtype_or_frozen_spec_rejects(self):
        mutations = (
            lambda actor, row: row.update(time_s=0.03),
            lambda actor, row: row.update(command=torch.ones(1, 3)),
            lambda actor, row: row.update(
                command=torch.ones(2, 3, dtype=torch.float64)
            ),
            lambda actor, row: actor.spec.configuration["nested"].__setitem__(0, 9),
        )
        for mutate in mutations:
            actor = Actor()
            host = session(actor)
            host.step(**decision())
            values = decision(0.02)
            mutate(actor, values)
            with self.assertRaises(ValueError):
                host.step(**values)
            self.assertTrue(host.faulted)

    def test_named_output_and_finite_private_vector_are_validated(self):
        for output in (
            JointTargets(("right", "left"), torch.zeros(2, 2)),
            JointTargets(("left", "right"), torch.full((2, 2), torch.nan)),
            JointTargets(("left", "right"), torch.zeros(2, 2, dtype=torch.float64)),
            JointTargets(("left", "right"), torch.zeros(2, 2), torch.zeros(2, 0)),
            JointTargets(
                ("left", "right"), torch.zeros(2, 2), torch.full((2, 1), torch.nan)
            ),
        ):
            actor = Actor()
            actor.output = output
            host = session(actor)
            with self.assertRaises(ValueError):
                host.step(**decision())
            self.assertTrue(host.faulted)

    def test_missing_optional_sensor_is_none_and_delivery_is_detached(self):
        actor, values = Actor(), decision()
        values["sensors"].pop("optional")
        values["command"].requires_grad_()
        values["sensors"]["required"].value.requires_grad_()
        output = session(actor).step(**values)
        self.assertIsNone(actor.inputs[0].sensors["optional"])
        self.assertFalse(actor.inputs[0].command.requires_grad)
        self.assertFalse(actor.inputs[0].sensors["required"].value.requires_grad)
        self.assertFalse(output.position_rad.requires_grad)


if __name__ == "__main__":
    unittest.main()

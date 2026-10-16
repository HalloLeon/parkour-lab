"""Current-only learning/actor artifacts; tensor fixtures are not native evidence."""

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

import torch

from parkour_lab import artifacts
from parkour_lab.config import ExperimentConfig, MethodConfig, TaskConfig
from parkour_lab.control.controller import ControllerSession
from parkour_lab.control.motor_contract import (
    make_motor_contract,
    nominal_binding_sha256,
)
from parkour_lab.methods.roa import model, runtime
from parkour_lab.methods.roa.training import ROATrainingMethod
from tests.support import installed_backends
from tests.support.native import motor_binding
from tests.support.roa import TensorHost, observations, packet, random_frames


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(825)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.config = ExperimentConfig(
            task=TaskConfig(num_envs=8, device="cpu"),
            method=MethodConfig(
                options=dict(
                    rollout_steps=2,
                    history_steps=2,
                    history_interval=1,
                    adaptation_epochs=1,
                    adaptation_batches=2,
                    num_learning_epochs=1,
                    num_mini_batches=2,
                    contact_conditioned=False,
                )
            ),
            updates=1,
            save_interval=1,
        )
        obs = observations(8)
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy, _ = model.build_policy(obs)
        options = dict(self.config.method.options)
        options.pop("contact_conditioned")
        options.pop("initial_action_std")
        self.method = ROATrainingMethod(
            self.policy, obs, num_envs=8, device="cpu", **options
        )
        self.method.advance(TensorHost(obs), obs)
        self.names = tuple(f"j{i}" for i in range(12))
        self.default = torch.linspace(-0.5, 0.5, 12)
        binding = motor_binding(self.names, self.default.tolist(), count=3)
        parameters = binding["actuators"]["legs"]["resolved_parameters"]
        for name, rows in parameters.items():
            parameters[name] = [rows[0].copy() for _ in range(8)]
        self.profile = "native_motor_sha256:" + nominal_binding_sha256(binding)
        self.contract = make_motor_contract(binding, self.profile)
        actor = self.policy.actor
        self.controller = runtime.ROAHistoryController(
            actor.motor,
            actor.estimator,
            joint_names=self.names,
            default_position_rad=self.default,
            artifact_sha256=runtime.roa_tensor_sha256(actor.motor, actor.estimator),
            actuator_profile=self.profile,
        )
        self.checkpoint = self.root / "checkpoint.pt"
        self.actor_path = self.root / "actor.pt"
        self.save_training(self.checkpoint)

    def save_training(self, path, **overrides):
        arguments = dict(
            kind="training",
            config=self.config,
            state=self.method.state_dict(),
            motor_contract=self.contract,
            motor_manifest=self.controller.spec.manifest(),
            updates=self.method.updates,
        )
        arguments.update(overrides)
        artifacts.save_artifact(path, **arguments)

    def mutate_file(self, source, name, mutate):
        with ZipFile(source) as archive:
            data = json.loads(archive.read("metadata.json"))
            data["state"] = torch.load(
                io.BytesIO(archive.read("payload")),
                map_location="cpu",
                weights_only=True,
            )
        mutate(data)
        buffer = io.BytesIO()
        torch.save(data.pop("state"), buffer)
        payload = buffer.getvalue()
        data["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        path = self.root / name
        with ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", json.dumps(data))
            archive.writestr("payload", payload)
        return path

    def test_checkpoint_export_load_retains_only_causal_weights_and_exact_actions(self):
        training = artifacts.load_artifact(self.checkpoint, kind="training")
        self.assertEqual(training["updates"], 1)
        self.assertEqual(training["state"]["updates"], 1)
        receipt = artifacts.export_actor(self.checkpoint, self.actor_path)
        self.assertFalse(receipt["qualified"])
        self.assertEqual(
            receipt["source_checkpoint_sha256"], artifacts.file_sha256(self.checkpoint)
        )
        self.assertEqual(receipt["sha256"], artifacts.file_sha256(self.actor_path))
        data = artifacts.load_artifact(self.actor_path, kind="actor")
        self.assertEqual(set(data["state"]), {"motor", "estimator"})
        self.assertNotIn("source_physical_reference", data)
        loaded = artifacts.load_actor(self.actor_path)
        self.assertEqual(loaded.updates, 1)
        self.assertEqual(loaded.config, self.config)
        self.assertEqual(loaded.motor_contract, self.contract)
        self.assertIsNone(loaded.controller.history_frames)
        session = ControllerSession(
            loaded.controller, joint_names=self.names, actuator_profile=self.profile
        )
        history = model.CausalHistory()
        previous = torch.zeros(3, 12)
        for step in range(12):
            reset = torch.tensor([step == 0, step in (0, 4), step in (0, 7)])
            frame = random_frames(3)
            frame[:, 33:45] = previous
            frame[reset, 33:] = 0
            frames = history.push(frame, reset)
            with torch.inference_mode():
                expected = self.policy.actor.history_action(frame, frames)
            actual = session.step(**packet(loaded.controller, frame, step, reset))
            self.assertTrue(torch.equal(actual.raw_action, expected))
            self.assertTrue(
                torch.equal(actual.position_rad, self.default + 0.25 * expected)
            )
            previous = actual.raw_action.clone()

    def test_envelope_delegates_options_and_opaque_payload_to_backend(self):
        # A serialization probe, not an implemented learning method.
        backend = SimpleNamespace(
            configure=Mock(side_effect=lambda options: dict(options)),
            dump=Mock(side_effect=lambda state, *args: json.dumps(state).encode()),
            load=Mock(side_effect=lambda payload, *args: json.loads(payload)),
        )
        with (
            installed_backends({"payload_probe": "payload_probe"}),
            patch.dict(sys.modules, {"payload_probe": backend}),
        ):
            config = ExperimentConfig(
                method=MethodConfig("payload_probe", {"custom_option": "value"})
            )
            state = {"method_owned_state": [1, 2, 3]}
            path = self.root / "opaque.plab"
            self.save_training(path, config=config, state=state)
            loaded = artifacts.load_artifact(path, kind="training")
            self.assertEqual(loaded["state"], state)
            self.assertEqual(ExperimentConfig.from_dict(loaded["config"]), config)
            backend.dump.assert_called_once_with(
                state, "training", config.method.options, 1
            )
            backend.load.assert_called_once_with(
                json.dumps(state).encode(), "training", config.method.options, 1
            )

    def test_explicit_physics_export_preserves_weights_and_inputs_not_runtime_identity(
        self,
    ):
        source_bytes = self.checkpoint.read_bytes()
        artifacts.export_actor(self.checkpoint, self.actor_path)
        target = self.root / "actor_400.plab"
        receipt = artifacts.export_actor(self.checkpoint, target, physics_hz=400)
        baseline = artifacts.load_actor(self.actor_path)
        candidate = artifacts.load_actor(target)
        self.assertEqual(self.checkpoint.read_bytes(), source_bytes)
        self.assertEqual(receipt["source_physics_hz"], 200)
        self.assertEqual(receipt["physics_hz"], 400)
        self.assertTrue(receipt["timing_changed"])
        self.assertTrue(receipt["timing_change_requires_validation"])
        self.assertFalse(receipt["qualified"])
        self.assertEqual(receipt["source_actuator_profile"], self.profile)
        self.assertEqual(
            receipt["actuator_profile"], candidate.controller.spec.actuator_profile
        )
        self.assertNotEqual(receipt["actuator_profile"], self.profile)
        self.assertEqual(
            candidate.config,
            replace(self.config, task=replace(self.config.task, physics_hz=400)),
        )
        expected_contract = copy.deepcopy(self.contract)
        expected_contract["binding"].update(physics_dt_s=0.0025, decimation=8)
        self.assertEqual(candidate.motor_contract, expected_contract)
        self.assertEqual(candidate.motor_contract["source_num_envs"], 8)
        self.assertEqual(candidate.updates, baseline.updates)
        self.assertEqual(
            candidate.controller.state_sha256(), baseline.controller.state_sha256()
        )
        baseline_data = artifacts.load_artifact(self.actor_path, kind="actor")
        candidate_data = artifacts.load_artifact(target, kind="actor")
        torch.testing.assert_close(
            baseline_data["state"], candidate_data["state"], rtol=0, atol=0
        )
        expected_manifest = copy.deepcopy(baseline_data["motor_manifest"])
        expected_manifest["actuator_profile"] = receipt["actuator_profile"]
        self.assertEqual(candidate_data["motor_manifest"], expected_manifest)
        self.assertEqual(
            candidate_data["motor_manifest"],
            json.loads(json.dumps(candidate.controller.spec.manifest())),
        )
        self.assertNotIn("200Hz", json.dumps(candidate_data["motor_manifest"]))
        sessions = [
            ControllerSession(
                actor.controller,
                joint_names=self.names,
                actuator_profile=actor.controller.spec.actuator_profile,
            )
            for actor in (baseline, candidate)
        ]
        for step in range(4):
            frame = random_frames(3)
            reset = torch.tensor([step == 0, step in (0, 2), step == 0])
            frame[reset, 33:] = 0
            actions = [
                session.step(**packet(session.controller, frame, step, reset))
                for session in sessions
            ]
            self.assertTrue(torch.equal(actions[0].raw_action, actions[1].raw_action))
            self.assertTrue(
                torch.equal(actions[0].position_rad, actions[1].position_rad)
            )
        target_bytes = target.read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.export_actor(self.checkpoint, target, physics_hz=400)
        self.assertEqual(target.read_bytes(), target_bytes)

    def test_physics_binding_must_agree_with_config_and_old_missing_field_means_200(
        self,
    ):
        old = self.mutate_file(
            self.checkpoint,
            "old.plab",
            lambda data: data["config"]["task"].pop("physics_hz"),
        )
        artifacts.export_actor(old, self.actor_path)
        self.assertEqual(
            artifacts.load_actor(self.actor_path).config.task.physics_hz, 200
        )
        candidate = self.root / "candidate.plab"
        artifacts.export_actor(old, candidate, physics_hz=400)
        for index, (source, rate) in enumerate(
            ((self.actor_path, 400), (candidate, 200))
        ):
            path = self.mutate_file(
                source,
                f"mismatch{index}.plab",
                lambda data: data["config"]["task"].update(physics_hz=rate),
            )
            with self.assertRaisesRegex(ValueError, "disagrees with motor binding"):
                artifacts.load_actor(path)
        with self.assertRaisesRegex(ValueError, "disagrees with motor binding"):
            self.save_training(
                self.root / "bad_save.plab",
                config=replace(
                    self.config, task=replace(self.config.task, physics_hz=400)
                ),
            )
        self.assertFalse((self.root / "bad_save.plab").exists())
        for rate in (100, 400.0, True):
            path = self.root / f"invalid{rate}.plab"
            with self.assertRaises(ValueError):
                artifacts.export_actor(self.checkpoint, path, physics_hz=rate)
            self.assertFalse(path.exists())

    def test_rejects_unsupported_formats_unknown_kinds_and_fields(self):
        changes = (
            lambda data: data.update(format="parkour_lab_method_v1"),
            lambda data: data.update(format="parkour_lab_method_v2"),
            lambda data: data.update(format="go2_operator_roa_history_bundle_v1"),
            lambda data: data.update(format="go2_operator_actor_bundle_v2"),
            lambda data: data.update(kind="legacy_teacher"),
            lambda data: data.update(source_physical_reference={}),
            lambda data: data.update(updates=True),
        )
        for index, mutate in enumerate(changes):
            path = self.mutate_file(self.checkpoint, f"legacy{index}.pt", mutate)
            with self.subTest(index=index), self.assertRaises(ValueError):
                artifacts.load_artifact(path, kind="training")
        with self.assertRaises(ValueError):
            artifacts.load_artifact(self.checkpoint, kind="actor")

    def test_invalid_configuration_is_rejected_without_constructing_a_simulator(self):
        changes = (
            lambda data: data["config"].update(legacy_recipe="old"),
            lambda data: data["config"]["method"]["options"].update(
                initial_action_std=0
            ),
            lambda data: data["config"]["task"].update(difficulty_range=[0.8, 0.2]),
            lambda data: data["config"]["method"]["options"].update(
                learning_rate=float("nan")
            ),
        )
        for index, mutate in enumerate(changes):
            path = self.mutate_file(self.checkpoint, f"config{index}.pt", mutate)
            with self.subTest(index=index), self.assertRaises(ValueError):
                artifacts.load_artifact(path, kind="training")

    def test_nonfinite_learning_weights_are_rejected_on_save_and_load(self):
        for value in (float("nan"), float("inf")):
            state = copy.deepcopy(self.method.state_dict())
            state["policy_state"]["actor.estimator.0.weight"][0, 0] = value
            output = self.root / "invalid_training.pt"
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.save_training(output, state=state)
            self.assertFalse(output.exists())
            path = self.mutate_file(
                self.checkpoint,
                "nonfinite_training.pt",
                lambda data: data.update(state=state),
            )
            with self.assertRaises(ValueError):
                artifacts.load_artifact(path, kind="training")

    def test_nonfinite_actor_weights_and_noncausal_state_are_rejected(self):
        artifacts.export_actor(self.checkpoint, self.actor_path)
        changes = (
            lambda data: data["state"]["estimator"]["0.weight"].fill_(float("nan")),
            lambda data: data["state"]["motor"]["stock_first.weight"].fill_(
                float("inf")
            ),
            lambda data: data["state"].update(encoder={}),
        )
        for index, mutate in enumerate(changes):
            path = self.mutate_file(self.actor_path, f"bad_actor{index}.pt", mutate)
            with self.subTest(index=index), self.assertRaises(ValueError):
                artifacts.load_actor(path)

    def test_optimizer_containers_and_nonfinite_moments_are_rejected_before_loading(
        self,
    ):
        for owner in ("ppo_optimizer", "adaptation_optimizer"):
            for defect in ("container", "state", "moments"):
                state = copy.deepcopy(self.method.state_dict())
                if defect == "container":
                    state[owner] = None
                elif defect == "state":
                    state[owner]["state"] = []
                else:
                    moments = next(iter(state[owner]["state"].values()))
                    moments["exp_avg"].fill_(float("nan"))
                path = self.mutate_file(
                    self.checkpoint,
                    f"{owner}_{defect}.pt",
                    lambda data: data.update(state=state),
                )
                with (
                    self.subTest(owner=owner, defect=defect),
                    self.assertRaises(ValueError),
                ):
                    artifacts.load_artifact(path, kind="training")

    def test_existing_checkpoints_and_exports_are_not_overwritten(self):
        checkpoint_bytes = self.checkpoint.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save_training(self.checkpoint)
        self.assertEqual(self.checkpoint.read_bytes(), checkpoint_bytes)
        artifacts.export_actor(self.checkpoint, self.actor_path)
        actor_bytes = self.actor_path.read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.export_actor(self.checkpoint, self.actor_path)
        self.assertEqual(self.actor_path.read_bytes(), actor_bytes)

    def test_envelope_rejects_extra_members_and_payload_corruption(self):
        for defect in ("member", "duplicate"):
            path = self.root / f"{defect}.plab"
            path.write_bytes(self.checkpoint.read_bytes())
            with ZipFile(path, "a") as archive:
                if defect == "member":
                    archive.writestr("unexpected", b"bad")
                else:
                    # Duplicate payload members are rejected too; no zip traversal
                    # or extraction is ever needed by an artifact reader.
                    with self.assertWarns(UserWarning):
                        archive.writestr("payload", b"corrupt")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                artifacts.load_artifact(path, kind="training")

        with ZipFile(self.checkpoint) as archive:
            metadata, payload = archive.read("metadata.json"), archive.read("payload")
        path = self.root / "bad_hash.plab"
        with ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", metadata)
            archive.writestr("payload", payload + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            artifacts.load_artifact(path, kind="training")

    def test_export_and_load_do_not_advance_training_rng(self):
        before = torch.get_rng_state().clone()
        artifacts.export_actor(self.checkpoint, self.actor_path)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        artifacts.load_actor(self.actor_path)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))


if __name__ == "__main__":
    unittest.main()

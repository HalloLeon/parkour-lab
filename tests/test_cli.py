"""CLI configuration and lifecycle contracts with native application doubles."""

import builtins
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from parkour_lab.__main__ import main, parse_args, resolve_config
from parkour_lab.config import ExperimentConfig, MethodConfig, TaskConfig

from tests.support import installed_backends
from tests.support.isaaclab import modules


class CLIConfigurationTests(unittest.TestCase):
    def test_terrain_identity_derives_world_and_stratum_and_round_trips(self):
        from parkour_lab.environments.randomization import terrain_development_attempt

        self.assertEqual(TaskConfig().dynamics, "randomized")
        for index in (0, 49, 50, 99):
            with self.subTest(index=index):
                assignment = terrain_development_attempt("ramp/down/10deg", index)
                task = TaskConfig(
                    terrain="connected",
                    num_envs=1,
                    episode_length_s=31,
                    terrain_group="ramp/down/10deg",
                    terrain_attempt_index=index,
                )
                self.assertEqual(task.world, assignment["world"])
                self.assertEqual(task.dynamics, assignment["stratum"])
                config = ExperimentConfig(task=task)
                self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
                with self.assertRaisesRegex(ValueError, "evaluation-only"):
                    config.validate_training()
                for change in (
                    dict(terrain="flat"),
                    dict(num_envs=2),
                    dict(num_rows=2),
                    dict(episode_length_s=30),
                    dict(bank_profile="stand"),
                    dict(terrain_group=None),
                    dict(terrain_attempt_index=None),
                    dict(terrain_attempt_index=True),
                    dict(terrain_attempt_index=100),
                    dict(world={**task.world, "world_yaw": 0}),
                    dict(dynamics="randomized" if index < 50 else "nominal"),
                ):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        TaskConfig(**{**config.to_dict()["task"], **change})

    def test_terrain_selector_is_explicit_and_rejects_competing_inputs(self):
        selector = ["--terrain-group", "ramp/down/10deg", "--attempt-index", "50"]
        for extra in (
            ["--seed", "0"],
            ["--num-envs", "1"],
            ["--steps", "1500"],
            ["--command", "0.35", "0", "0"],
            ["--tape", "tape.json"],
            ["--profile", "stand"],
            ["--bank-profile", "stand"],
        ):
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["evaluate", "actor.plab", *selector, *extra])
        for command in (
            ["evaluate", "actor.plab", "--terrain-group", "ramp/down/10deg"],
            ["evaluate", "actor.plab", "--attempt-index", "50"],
            [
                "evaluate",
                "actor.plab",
                "--terrain-group",
                "ramp/down/10deg",
                "--attempt-index",
                "-1",
            ],
            ["train", *selector],
            ["play", "actor.plab", *selector],
        ):
            with (
                self.subTest(command=command),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(command)
        original = self.training_config()
        with patch(
            "parkour_lab.artifacts.load_artifact",
            return_value={"config": original.to_dict()},
        ):
            config = resolve_config(parse_args(["evaluate", "actor.plab", *selector]))
        self.assertEqual(config.method, original.method)
        self.assertEqual(config.task.physics_hz, original.task.physics_hz)
        self.assertEqual(config.task.terrain_attempt_index, 50)
        self.assertEqual(config.task.seed, 0)
        self.assertEqual(config.task.dynamics, "randomized")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config.to_dict()))
            with patch(
                "parkour_lab.artifacts.load_artifact",
                return_value={"config": original.to_dict()},
            ):
                self.assertEqual(
                    resolve_config(
                        parse_args(
                            ["evaluate", "actor.plab", *selector, "--config", str(path)]
                        )
                    ),
                    config,
                )
                for operation in ("evaluate", "play", "train"):
                    args = [operation, "--config", str(path)]
                    if operation != "train":
                        args.insert(1, "actor.plab")
                    with (
                        self.subTest(operation=operation),
                        self.assertRaises(ValueError),
                    ):
                        resolve_config(parse_args(args))
            path.write_text(
                json.dumps({"task": {"terrain": "flat", "dynamics": "nominal"}})
            )
            with (
                patch(
                    "parkour_lab.artifacts.load_artifact",
                    return_value={"config": original.to_dict()},
                ),
                self.assertRaisesRegex(ValueError, "conflict"),
            ):
                resolve_config(
                    parse_args(
                        ["evaluate", "actor.plab", *selector, "--config", str(path)]
                    )
                )

    def test_terrain_analysis_is_offline_and_exclusive_with_flat_bank(self):
        result = {"diagnostic_passed": False, "qualified": False}
        with (
            patch(
                "parkour_lab.evaluation.runner.analyze_terrain_attempt",
                return_value=result,
            ) as analyze,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(["analyze", "saved-run", "--terrain"]), 0)
        analyze.assert_called_once_with(Path("saved-run"))
        self.assertEqual(json.loads(output.getvalue()), result)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["analyze", "saved-run", "--terrain", "--flat-bank"])

    def test_connected_world_is_explicit_single_environment_playback(self):
        world = dict(
            target_family="ramp", tier=10, coarse_seed=21, fine_seed=22, reverse=True
        )
        task = TaskConfig(
            terrain="connected", num_envs=1, world=world, dynamics="nominal"
        )
        self.assertEqual(task.difficulty_range, (1.0, 1.0))
        self.assertEqual(task.physics_hz, 200)
        config = ExperimentConfig(task=task)
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            config.validate_training()
        for change in (
            dict(num_envs=2),
            dict(num_rows=2),
            dict(difficulty_range=(0, 1)),
            dict(world=None),
            dict(world={}),
            dict(world={**world, "unknown": 0}),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TaskConfig(
                    **{**dict(terrain="connected", num_envs=1, world=world), **change}
                )
        with self.assertRaisesRegex(ValueError, "require connected"):
            TaskConfig(terrain="flat", world=world)

    def test_explicit_export_physics_option_is_portable_and_bounded(self):
        for extra, expected in (([], None), (["--physics-hz", "400"], 400)):
            with (
                patch("parkour_lab.artifacts.export_actor", return_value={}) as export,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(["export", "checkpoint.plab", "actor.plab", *extra]), 0
                )
            export.assert_called_once_with(
                Path("checkpoint.plab"), Path("actor.plab"), physics_hz=expected
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                ["export", "checkpoint.plab", "actor.plab", "--physics-hz", "100"]
            )

    def test_artifact_rate_is_preserved_and_cannot_be_overridden_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "timing.json"
            for original_rate, other_rate in ((200, 400), (400, 200)):
                original = ExperimentConfig(task=TaskConfig(physics_hz=original_rate))
                path.write_text(json.dumps({"task": {"physics_hz": other_rate}}))
                with patch(
                    "parkour_lab.artifacts.load_artifact",
                    return_value={"config": original.to_dict()},
                ):
                    candidate = resolve_config(
                        parse_args(
                            ["evaluate", "actor.plab", "--bank-profile", "stand"]
                        )
                    )
                    self.assertEqual(candidate.task.physics_hz, original_rate)
                    for command in (
                        ["evaluate", "actor.plab"],
                        ["train", "--checkpoint", "checkpoint.plab"],
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "physics rate cannot be overridden"
                        ):
                            resolve_config(
                                parse_args([*command, "--config", str(path)])
                            )

    def test_only_operational_commands_are_exposed(self):
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(
            "{train,evaluate,play,export,bound-actions,analyze}", output.getvalue()
        )

    def test_invalid_threads_and_nonfinite_commands_fail_before_launch(self):
        for arguments in (
            ["train", "--cpu-threads", "0"],
            ["evaluate", "actor.plab", "--command", "nan", "0", "0"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(arguments)

    def test_flat_bank_fixes_task_and_attempts_without_changing_method(self):
        original = self.training_config()
        with patch(
            "parkour_lab.artifacts.load_artifact",
            return_value={"config": original.to_dict()},
        ):
            config = resolve_config(
                parse_args(["evaluate", "actor.plab", "--bank-profile", "stand"])
            )
        self.assertEqual(config.method, original.method)
        self.assertEqual(
            (
                config.task.terrain,
                config.task.num_envs,
                config.task.seed,
                config.task.bank_profile,
            ),
            ("flat", 100, 0, "stand"),
        )
        for extra in (
            ["--seed", "7"],
            ["--num-envs", "100"],
            ["--steps", "550"],
            ["--tape", "tape.json"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(
                    ["evaluate", "actor.plab", "--bank-profile", "stand", *extra]
                )

    def test_flat_only_dynamics_and_bank_are_not_training_overrides(self):
        with self.assertRaises(ValueError):
            TaskConfig(dynamics="nominal")
        for task in (
            TaskConfig(terrain="flat", dynamics="nominal"),
            TaskConfig(terrain="flat", num_envs=100, bank_profile="stand"),
        ):
            with self.assertRaises(ValueError):
                ExperimentConfig(task=task).validate_training()

    def test_tracking_widths_require_finite_positive_numbers(self):
        for name in ("linear_tracking_std", "angular_tracking_std"):
            for value in (0, -0.1, True, "0.25", None, float("inf"), float("nan")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    TaskConfig(**{name: value})

    def test_task_reward_override_preserves_continuation_method_and_defaults(self):
        original = self.training_config()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rewards.json"
            path.write_text(json.dumps({"task": {"linear_tracking_std": 0.25}}))
            with patch(
                "parkour_lab.artifacts.load_artifact",
                return_value={"config": original.to_dict()},
            ):
                resolved = resolve_config(
                    parse_args(
                        [
                            "train",
                            "--checkpoint",
                            "checkpoint.plab",
                            "--config",
                            str(path),
                        ]
                    )
                )
        self.assertEqual(resolved.method, original.method)
        self.assertEqual(resolved.task.linear_tracking_std, 0.25)
        self.assertEqual(resolved.task.angular_tracking_std, 0.5)
        self.assertEqual(original.task.linear_tracking_std, 0.5)
        self.assertEqual(ExperimentConfig.from_dict(resolved.to_dict()), resolved)

    def test_flat_profile_is_exclusive_and_rejects_wrong_task_before_startup(self):
        for arguments in (
            ["--profile", "stand", "--steps", "550"],
            ["--profile", "stand", "--command", "0", "0", "0"],
            ["--profile", "stand", "--tape", "commands.json"],
            ["--profile", "not-a-profile"],
            ["--bank-profile", "not-a-profile"],
            ["--profile", ""],
            ["--bank-profile", ""],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["evaluate", "actor.plab", *arguments])
        for terrain, duration, allowed in (
            ("flat", 20, True),
            ("flat", 11, False),
            ("procedural", 20, False),
        ):
            with patch(
                "parkour_lab.artifacts.load_artifact",
                return_value={
                    "config": ExperimentConfig(
                        task=TaskConfig(terrain=terrain, episode_length_s=duration)
                    ).to_dict()
                },
            ):
                args = parse_args(["evaluate", "actor.plab", "--profile", "stand"])
                if allowed:
                    self.assertEqual(resolve_config(args).task.terrain, "flat")
                else:
                    with self.assertRaisesRegex(ValueError, "Flat profiles require"):
                        resolve_config(args)

    def test_continuation_reward_map_is_retained_when_omitted_and_replaced_when_explicit(
        self,
    ):
        original = ExperimentConfig(
            task=TaskConfig(rewards={"action_rate_l2": {"weight": -0.02}})
        )
        for task, expected in (
            ({"linear_tracking_std": 0.3}, original.task.rewards),
            ({"rewards": {}}, {}),
            (
                {"rewards": {"feet_air_time": {"weight": 0.03}}},
                {"feet_air_time": {"weight": 0.03}},
            ),
        ):
            with tempfile.TemporaryDirectory() as folder, self.subTest(task=task):
                path = Path(folder) / "task.json"
                path.write_text(json.dumps({"task": task}))
                with patch(
                    "parkour_lab.artifacts.load_artifact",
                    return_value={"config": original.to_dict()},
                ):
                    config = resolve_config(
                        parse_args(
                            [
                                "train",
                                "--checkpoint",
                                "checkpoint.plab",
                                "--config",
                                str(path),
                            ]
                        )
                    )
                self.assertEqual(config.task.rewards, expected)
                self.assertEqual(config.method, original.method)
                self.assertEqual(
                    original.task.rewards, {"action_rate_l2": {"weight": -0.02}}
                )

    def training_config(self):
        return ExperimentConfig(
            task=TaskConfig(num_envs=8, device="cpu"),
            method=MethodConfig(options={"rollout_steps": 2, "num_mini_batches": 8}),
        )

    def test_play_and_evaluate_can_use_one_environment_without_changing_method_settings(
        self,
    ):
        original = self.training_config()
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps(original.to_dict()))
            for command in (
                ["play", "actor.plab", "--config", str(config)],
                ["evaluate", "actor.plab", "--config", str(config), "--num-envs", "1"],
            ):
                with (
                    self.subTest(command=command),
                    patch(
                        "parkour_lab.artifacts.load_artifact",
                        return_value={"config": original.to_dict()},
                    ),
                ):
                    resolved = resolve_config(parse_args(command))
                self.assertEqual(resolved.task.num_envs, 1)
                self.assertEqual(resolved.method, original.method)
                self.assertEqual(original.task.num_envs, 8)

    def test_task_only_configuration_preserves_actor_method_and_rejects_conflicts(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            installed_backends({"external_roa": "parkour_lab.methods.roa"}),
            patch(
                "parkour_lab.artifacts.load_artifact",
            ) as load,
        ):
            original = ExperimentConfig(
                method=MethodConfig("external_roa", {"learning_rate": 0.002})
            )
            load.return_value = {"config": original.to_dict()}
            path = Path(folder) / "task.json"
            args = parse_args(["evaluate", "actor.plab", "--config", str(path)])
            path.write_text(json.dumps({"task": {"terrain": "flat", "num_envs": 1}}))
            resolved = resolve_config(args)
            self.assertEqual(resolved.method, original.method)
            self.assertEqual(resolved.task.terrain, "flat")
            self.assertEqual(resolved.task.num_envs, 1)
            self.assertEqual(resolved.task.seed, original.task.seed)
            for method in (
                {"name": "roa"},
                {"name": "external_roa", "options": {"learning_rate": 0.003}},
            ):
                path.write_text(json.dumps({"method": method}))
                with self.assertRaisesRegex(ValueError, "cannot be overridden"):
                    resolve_config(args)

    def test_actor_configuration_defaults_also_allow_inference_batch_override(self):
        original = self.training_config()
        with patch(
            "parkour_lab.artifacts.load_artifact",
            return_value={"config": original.to_dict()},
        ) as load:
            resolved = resolve_config(
                parse_args(["evaluate", "actor.pt", "--num-envs", "1"])
            )
        load.assert_called_once_with(Path("actor.pt"), kind="actor")
        self.assertEqual(resolved.task.num_envs, 1)
        self.assertEqual(resolved.method.options["num_mini_batches"], 8)

    def test_cross_field_batch_constraints_are_training_only(self):
        # These are meaningful inference settings, but impossible PPO minibatches.
        config = ExperimentConfig(
            task=TaskConfig(num_envs=1, device="cpu"),
            method=MethodConfig(options={"rollout_steps": 2, "num_mini_batches": 8}),
        )
        with self.assertRaises(ValueError):
            config.validate_training()
        self.training_config().validate_training()
        uneven = ExperimentConfig(
            task=TaskConfig(num_envs=3, device="cpu"),
            method=MethodConfig(options={"rollout_steps": 2, "num_mini_batches": 4}),
        )
        with self.assertRaises(ValueError):
            uneven.validate_training()

    def test_invalid_training_batches_fail_before_native_imports_or_output_creation(
        self,
    ):
        invalid = ExperimentConfig(
            task=TaskConfig(num_envs=3, device="cpu"),
            method=MethodConfig(options={"rollout_steps": 2, "num_mini_batches": 4}),
        )
        original_import = builtins.__import__
        native_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("isaaclab", "omni")):
                native_imports.append(name)
                raise AssertionError("Invalid training reached native imports")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "runs"
            settings = Path(folder) / "settings.json"
            settings.write_text(json.dumps(invalid.to_dict()))
            with (
                patch("builtins.__import__", side_effect=guarded_import),
                self.assertRaises(ValueError),
            ):
                main(
                    [
                        "train",
                        "--config",
                        str(settings),
                        "--output-parent",
                        str(output),
                    ]
                )
            self.assertEqual(native_imports, [])
            self.assertFalse(output.exists())


class CLILifecycleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def run_session(
        self,
        operation,
        *,
        failure=None,
        standalone=False,
        close_error=False,
        terrain_attempt=False,
    ):
        order = []

        def launch(*, headless, device, fast_shutdown=True):
            self.assertEqual(headless, operation != "play")
            self.assertEqual(device, "cpu")

            def close():
                order.append("application")
                if fast_shutdown:
                    raise SystemExit(0)
                if close_error:
                    raise RuntimeError("close failed")

            return NS(app=NS(close=close))

        def execute(*args, **kwargs):
            if failure is not None:
                raise failure
            if terrain_attempt:
                commands = args[3]
                self.assertEqual(len(commands), 1500)
                self.assertEqual(commands[:100], [[0.0, 0.0, 0.0]] * 100)
                self.assertEqual(commands[100:1350], [[0.35, 0.0, 0.0]] * 1250)
                self.assertEqual(commands[1350:], [[0.0, 0.0, 0.0]] * 150)
            args[-1]["status"] = status

        def terminate(code):
            raise SystemExit(code)

        status = {
            "train": "TRAINING_COMPLETE_NOT_QUALIFIED",
        }.get(operation, "EVALUATION_COMPLETE_NOT_QUALIFIED")
        cfg = NS(validate=lambda: None, to_dict=lambda: {})
        env = NS(
            close=lambda: order.append("environment"),
            transition_counts={"control_steps": 1},
        )
        replacements = {
            "isaaclab.app": NS(AppLauncher=launch),
            "parkour_lab.environments.runtime": NS(LocomotionEnv=lambda **kwargs: env),
            "parkour_lab.environments.configuration": NS(
                build_environment_config=lambda *args, **kwargs: cfg
            ),
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            modules(replacements),
            patch(
                "parkour_lab.artifacts.load_artifact",
                return_value={
                    "config": ExperimentConfig(task=TaskConfig(device="cpu")).to_dict()
                },
            ),
            patch("parkour_lab.provenance.write_run_provenance"),
            patch(
                "parkour_lab.environments.rewards.reward_recipe",
                return_value={"resolved": True},
            ) as recipe,
            patch("parkour_lab.experiment.train", side_effect=execute),
            patch("parkour_lab.evaluation.runner.evaluate", side_effect=execute),
            patch("parkour_lab.artifacts.load_actor"),
            patch("parkour_lab.artifacts.file_sha256", return_value="0" * 64),
            patch(
                "parkour_lab.runtime.session.exit_native_process", side_effect=terminate
            ) as exit_process,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            settings = Path(folder) / "settings.json"
            settings.write_text(json.dumps({"task": {"device": "cpu"}}))
            argv = [
                operation,
                *([] if operation == "train" else ["actor.pt"]),
                "--config",
                str(settings),
                "--output-parent",
                folder,
            ]
            if terrain_attempt:
                argv += ["--terrain-group", "ramp/down/10deg", "--attempt-index", "50"]
            code = 2 if failure is not None or close_error else 0
            if isinstance(failure, KeyboardInterrupt):
                code = 130
            elif (
                isinstance(failure, SystemExit)
                and type(failure.code) is int
                and 1 <= failure.code <= 255
            ):
                code = failure.code
            if standalone:
                with self.assertRaises(SystemExit) as raised:
                    main(argv, standalone=True)
                self.assertEqual(raised.exception.code, code)
                exit_process.assert_called_once_with(code)
                if failure is not None:
                    self.assertIn(type(failure).__name__, errors.getvalue())
            elif failure is not None:
                with self.assertRaises(type(failure)) as raised:
                    main(argv)
                self.assertIs(raised.exception, failure)
                exit_process.assert_not_called()
            else:
                self.assertEqual(main(argv), code)
                exit_process.assert_not_called()
            report = json.loads(next(Path(folder).glob("*/report.json")).read_text())
            recipe.assert_called_once_with(env)
            self.assertEqual(report["reward_recipe"], {"resolved": True})
        self.assertEqual(order, ["environment", "application"])
        self.assertEqual(
            report["cleanup"],
            {
                "environment": "complete",
                "application": "ERROR: close failed" if close_error else "complete",
            },
        )
        self.assertEqual(report["status"], "ERROR" if code else status)
        self.assertEqual(
            report["session_status"], "ERROR" if failure is not None else status
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["exit_allowed"])

    def test_all_native_commands_publish_final_receipts_after_orderly_shutdown(self):
        for operation in ("train", "evaluate", "play"):
            with self.subTest(operation=operation):
                self.run_session(operation)

    def test_terrain_selector_dispatches_full_prescribed_tape_and_cleans_up(self):
        self.run_session("evaluate", terrain_attempt=True)

    def test_shutdown_does_not_mask_operation_failure_with_successful_process_exit(
        self,
    ):
        self.run_session("train", failure=RuntimeError("operation failed"))

    def test_standalone_native_commands_exit_after_final_receipts(self):
        for operation in ("train", "evaluate", "play"):
            with self.subTest(operation=operation):
                self.run_session(operation, standalone=True)

    def test_standalone_native_failures_print_tracebacks_and_exit_nonzero(self):
        for failure in (
            RuntimeError("operation failed"),
            KeyboardInterrupt(),
            SystemExit(7),
            SystemExit(0),
            SystemExit(256),
            SystemExit(-256),
            SystemExit("operation failed"),
        ):
            with self.subTest(failure=repr(failure)):
                self.run_session("train", standalone=True, failure=failure)

    def test_standalone_cleanup_failure_does_not_become_a_successful_exit(self):
        self.run_session("evaluate", standalone=True, close_error=True)

    def test_portable_commands_return_without_terminating_the_process(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("parkour_lab.runtime.session.exit_native_process") as exit_process,
            patch("parkour_lab.artifacts.export_actor", return_value={}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = Path(folder) / "report.json"
            report.write_text("{}")
            self.assertEqual(main(["analyze", str(report)], standalone=True), 0)
            self.assertEqual(
                main(["export", "checkpoint.pt", "actor.pt"], standalone=True), 0
            )
            exit_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()

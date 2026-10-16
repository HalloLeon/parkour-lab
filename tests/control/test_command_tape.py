"""Portable command recording, validation and exact replay contracts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parkour_lab.control import command_tape as core

ZERO = (0.0, 0.0, 0.0)

MOTOR = "a" * 64

COMMANDS = (
    ZERO,
    (0.4, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (-0.3, 0.0, 0.2),
    (0.0, -0.2, -0.5),
    ZERO,
    ZERO,
)


def make_tape(commands=COMMANDS, *, completed=True, error=None):
    builder = core.TapeBuilder(
        {"portable_motor_sha256": MOTOR, "terrain": "plane", "seed": 42}
    )
    for step, command in enumerate(commands):
        builder.append(step, command)
    return builder.finish(completed=completed, error=error)


class CommandTapeTests(unittest.TestCase):
    def test_roundtrip_run_length_encoding_is_exact_and_hash_is_canonical(self):
        tape = make_tape()
        self.assertEqual(tape["steps"], len(COMMANDS))
        self.assertEqual(len(tape["segments"]), 5)
        self.assertEqual(tape["segments"][1]["start_step"], 1)
        self.assertEqual(tape["segments"][1]["end_step_exclusive"], 3)
        expanded = [
            tuple(segment["command"])
            for segment in tape["segments"]
            for _ in range(segment["start_step"], segment["end_step_exclusive"])
        ]
        self.assertEqual(expanded, list(COMMANDS))
        self.assertEqual(core.seal(dict(reversed(list(tape.items())))), tape)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.json"
            core.write_tape(path, tape)
            self.assertEqual(core.load_tape(path), tape)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_builder_seal_and_finish_do_not_alias_caller_data(self):
        metadata = {"nested": {"value": 1}}
        builder = core.TapeBuilder(metadata)
        metadata["nested"]["value"] = 2
        command = [0.4, 0.0, 0.0]
        builder.append(0, command)
        command[0] = 0.7
        builder.append(1, ZERO)
        tape = builder.finish(completed=True)
        tape["segments"][0]["command"][0] = -0.3
        tape["metadata"]["nested"]["value"] = 3
        fresh = builder.finish(completed=True)
        self.assertEqual(fresh["segments"][0]["command"][0], 0.4)
        self.assertEqual(fresh["metadata"]["nested"]["value"], 1)

    def test_payload_and_digest_corruption_fail_before_replay(self):
        for key in ("sha256", "metadata", "segments", "steps"):
            tape = make_tape()
            if key == "sha256":
                tape[key] = "0" * 64
            elif key == "metadata":
                tape[key]["seed"] = 43
            elif key == "segments":
                tape[key][1]["command"][0] = 0.7
            else:
                tape[key] -= 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                core.validate_tape(tape)

    def test_empty_moving_and_interrupted_tapes_remain_incomplete(self):
        for commands, completed, error in (
            ((), True, None),
            (((0.4, 0.0, 0.0),), True, None),
            ((ZERO,), False, None),
            ((ZERO,), True, "native call failed"),
        ):
            with self.subTest(commands=commands, completed=completed, error=error):
                tape = make_tape(commands, completed=completed, error=error)
                self.assertFalse(tape["complete"])
                core.validate_tape(tape, require_complete=False)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    core.validate_tape(tape)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "incomplete.json"
                    core.write_tape(path, tape)
                    with self.assertRaisesRegex(ValueError, "Incomplete"):
                        core.load_tape(path)

    def test_rehashed_false_completeness_still_rejects(self):
        for tape in (make_tape((), completed=False), make_tape(((0.4, 0, 0),))):
            tape.update(complete=True, error=None)
            with self.assertRaisesRegex(ValueError, "explicit zero"):
                core.validate_tape(core.seal(tape))
        tape = make_tape(error="interrupted")
        tape["complete"] = True
        with self.assertRaises(ValueError):
            core.validate_tape(core.seal(tape))

    def test_nonfinite_boolean_wrong_shape_and_non_numeric_commands_reject(self):
        for command in (
            (float("nan"), 0, 0),
            (0, float("inf"), 0),
            (0, 0, -float("inf")),
            (True, 0, 0),
            ("0.4", 0, 0),
            (0, 0),
            (0, 0, 0, 0),
            "000",
        ):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    core.TapeBuilder({}).append(0, command)
                tape = make_tape()
                tape["segments"][0]["command"] = command
                with self.assertRaises(ValueError):
                    core.validate_tape(tape)

    def test_steps_and_segments_cannot_skip_repeat_overlap_or_exceed_budget(self):
        for step in (1, -1, True, 0.0, core.MAX_STEPS):
            with self.subTest(step=step), self.assertRaises(ValueError):
                core.TapeBuilder({}).append(step, ZERO)
        builder = core.TapeBuilder({})
        builder.append(0, ZERO)
        with self.assertRaises(ValueError):
            builder.append(0, ZERO)
        for mutation in (
            lambda tape: tape["segments"][1].update(start_step=2),
            lambda tape: tape["segments"][1].update(start_step=0),
            lambda tape: tape["segments"][1].update(end_step_exclusive=1),
            lambda tape: tape["segments"][0].update(start_step=True),
            lambda tape: tape["segments"][-1].update(end_step_exclusive=8),
            lambda tape: tape.update(steps=core.MAX_STEPS + 1),
            lambda tape: tape.update(steps=True),
        ):
            tape = make_tape()
            mutation(tape)
            with self.assertRaises(ValueError):
                core.validate_tape(core.seal(tape))
        with patch.object(core, "MAX_STEPS", 2):
            builder = core.TapeBuilder({})
            builder.append(0, ZERO)
            builder.append(1, ZERO)
            with self.assertRaises(ValueError):
                builder.append(2, ZERO)

    def test_unknown_fields_duplicate_json_and_oversized_files_reject(self):
        for mutation in (
            lambda tape: tape.update(action=[1, 2, 3]),
            lambda tape: tape.update(period_s=0.01),
            lambda tape: tape.update(metadata=[]),
            lambda tape: tape.update(complete=1),
            lambda tape: tape["segments"][0].update(raw_action=[0]),
        ):
            tape = make_tape()
            mutation(tape)
            with self.assertRaises(ValueError):
                core.validate_tape(core.seal(tape))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"version":1,"version":2}', '{"outer":{"x":1,"x":2}}'):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    core.load_tape(path)
            path.write_text("x" * 65)
            with patch.object(core, "MAX_BYTES", 64):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    core.load_tape(path)
                with self.assertRaisesRegex(ValueError, "size limit"):
                    core.write_tape(Path(directory) / "new.json", make_tape())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_publication_never_overwrites_existing_file_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.json"
            path.write_text("valuable existing recording")
            with self.assertRaises(FileExistsError):
                core.write_tape(path, make_tape())
            self.assertEqual(path.read_text(), "valuable existing recording")
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(FileExistsError):
                core.write_tape(link, make_tape())
            self.assertTrue(link.is_symlink())
            self.assertEqual(path.read_text(), "valuable existing recording")
            self.assertEqual(set(Path(directory).iterdir()), {path, link})

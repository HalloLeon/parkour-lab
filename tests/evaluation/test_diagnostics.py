"""Two saved Flat controls expose movement, posture, contact and missing phases."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parkour_lab.evaluation import EvaluationCase, summarize_flat_entry
from parkour_lab.evaluation.capture import VERSION, reward_identity
from parkour_lab.provenance import file_sha256, write_json
from tests.support.evaluation import analytic_trace


def saved_controls(folder, *, steps=1000, stationary=False, omit_stand=False):
    cases = [
        EvaluationCase("validation", "Flat", None, "plane", profile)
        for profile in ("stand", "forward-035")
    ]
    identity = dict(
        policy_state_sha256="a" * 64,
        training_seed=42,
        checkpoint_id="caller",
        checkpoint_sha256="b" * 64,
    )
    recipe = {"terms": {"tracking": {"weight": 1.0}}}
    digest = reward_identity(recipe)
    write_json(
        folder / "manifest.json",
        dict(
            version=VERSION,
            status="COMPLETE",
            cases=[case.as_dict() for case in cases],
            case_version="v2",
            **identity,
            final_policy_state_sha256="a" * 64,
            core_reward_recipe_sha256=digest,
        ),
    )
    for case in cases:
        if omit_stand and case.profile == "stand":
            continue
        path = folder / case.case_id
        path.mkdir()
        trace = analytic_trace(case, steps=steps, heading_rad=np.pi / 2)
        if stationary:
            trace["root_position_w"][:] = trace["initial_root_position_w"]
            trace["root_position_w"][:, 2] = 0.1
            trace["root_com_linear_velocity_b"][:] = 0
            trace["foot_contact_forces_w"][:] = 0
        if steps:
            trace["terminated" if steps < 1000 else "truncated"][-1] = True
        np.savez_compressed(path / "trace.npz", **trace)
        write_json(
            path / "capture.json",
            dict(
                version=VERSION,
                case=case.as_dict(),
                status="CAPTURED",
                **identity,
                cleanup={"environment": "complete", "application": "not_created"},
                observed_steps=steps,
                interactions=dict(session=steps, cumulative=steps, uncertain=0),
                realization={"task": {"reward_recipe": recipe}},
                core_reward_recipe_sha256=digest,
                trace_sha256=file_sha256(path / "trace.npz"),
            ),
        )
    return cases


class FlatEntryTests(unittest.TestCase):
    def test_perfect_body_forward_control_has_heading_invariant_contrast(self):
        with tempfile.TemporaryDirectory() as temp:
            saved_controls(Path(temp))
            report = summarize_flat_entry(temp)
        self.assertTrue(report["capture_valid"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["valid_for_comparison"])
        self.assertFalse(report["calibration_established"])
        self.assertFalse(report["full_bank"])
        contrast = report["command_contrast"]
        self.assertAlmostEqual(contrast["forward_minus_stand_velocity_m_s"], 0.35)
        self.assertAlmostEqual(contrast["forward_minus_stand_travel_m"], 4.2)
        forward = report["cases"]["forward-035"]
        self.assertEqual(forward["settled_samples"], 550)
        self.assertAlmostEqual(forward["settled_mean_root_height_above_plane_m"], 0.35)
        self.assertEqual(forward["settled_foot_contact_fraction"], 1)
        self.assertEqual(forward["settled_supported_foot_contact_fraction"], 1)
        self.assertEqual(forward["stopping"]["residual_planar_speed_m_s"], 0)
        self.assertEqual(forward["core_return"], 1000)

    def test_stationary_low_posture_and_no_contact_are_not_hidden_by_return(self):
        with tempfile.TemporaryDirectory() as temp:
            saved_controls(Path(temp), stationary=True)
            report = summarize_flat_entry(temp)
        forward = report["cases"]["forward-035"]
        self.assertEqual(forward["core_return"], 1000)
        self.assertEqual(forward["actual_onset_forward_travel_m"], 0)
        self.assertEqual(forward["settled_foot_contact_fraction"], 0)
        self.assertEqual(forward["settled_supported_foot_contact_fraction"], 0)
        self.assertAlmostEqual(forward["settled_mean_root_height_above_plane_m"], 0.1)
        self.assertAlmostEqual(forward["settled_tracking"]["planar_rmse_m_s"], 0.35)
        self.assertEqual(
            report["command_contrast"]["forward_minus_stand_velocity_m_s"], 0
        )

    def test_early_falls_keep_the_return_and_unavailable_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            saved_controls(Path(temp), steps=25)
            report = summarize_flat_entry(temp)
        forward = report["cases"]["forward-035"]
        self.assertTrue(report["capture_valid"])
        self.assertTrue(forward["physical_failure"])
        self.assertEqual(forward["core_return"], 25)
        self.assertEqual(forward["settled_samples"], 0)
        self.assertIsNone(forward["actual_onset_forward_travel_m"])
        self.assertIsNone(forward["settled_mean_root_height_above_plane_m"])
        self.assertIsNone(forward["settled_foot_contact_fraction"])
        self.assertIsNone(forward["stopping"]["residual_planar_speed_m_s"])
        self.assertTrue(
            all(value is None for value in report["command_contrast"].values())
        )

    def test_missing_attempt_stays_unknown_and_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            saved_controls(Path(temp), omit_stand=True)
            report = summarize_flat_entry(temp)
        self.assertFalse(report["capture_valid"])
        self.assertIsNone(report["cases"]["stand"])
        self.assertTrue(
            all(value is None for value in report["command_contrast"].values())
        )

    def test_saved_trace_digest_is_checked_before_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            cases = saved_controls(Path(temp))
            path = Path(temp) / cases[0].case_id / "trace.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Trace digest"):
                summarize_flat_entry(temp)

    def test_summary_needs_no_learner_or_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            saved_controls(Path(temp))
            program = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3', 'isaaclab', 'omni', 'pxr'):
            raise AssertionError('Offline diagnostic imported ' + fullname)
sys.meta_path.insert(0, Guard())
from parkour_lab.evaluation import summarize_flat_entry
assert summarize_flat_entry(sys.argv[1])['diagnostic_only']
"""
            result = subprocess.run(
                [sys.executable, "-c", program, temp],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

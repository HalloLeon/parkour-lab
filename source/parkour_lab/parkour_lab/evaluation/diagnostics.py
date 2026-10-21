"""Saved Flat stand/forward controls, without success or calibration thresholds."""

import json
from pathlib import Path

import numpy as np

from parkour_lab.control.proprioception import CONTACT_THRESHOLD_N

from .capture import score_capture
from .cases import PERIOD_S, EvaluationCase, case_from_dict
from .scoring.exposure import MAXIMUM_FOOT_CENTER_CLEARANCE_M
from .scoring.kinematics import heading
from .scoring.trace import validate_trace


def _mean(values):
    return float(np.mean(values)) if values.size else None


def _measurements(trace):
    time = trace["decision_time_s"]
    settled = (time >= 3) & (time < 14)
    motion = (time >= 2) & (time < 14)
    position = trace["root_position_w"]
    anchor_index = round(2 / PERIOD_S) - 1
    travel = excursion = None
    if motion.any() and len(time) > anchor_index:
        offset = position[motion, :2] - position[anchor_index, :2]
        angle = heading(trace["root_quaternion_w"])[anchor_index]
        travel = float(offset[-1] @ np.array([np.cos(angle), np.sin(angle)]))
        excursion = float(np.linalg.norm(offset, axis=-1).max())
    contact = (
        np.linalg.norm(trace["foot_contact_forces_w"][settled], axis=-1)
        > CONTACT_THRESHOLD_N
    )
    valid = trace["support_valid"][settled]
    clearance = (
        trace["foot_position_w"][settled, :, 2] - trace["support_height_w"][settled]
    )
    supported = contact & valid & (np.abs(clearance) <= MAXIMUM_FOOT_CENTER_CLEARANCE_M)
    height = (
        position[settled, 2] - trace["env_origin_w"][2]
        if "env_origin_w" in trace
        else np.empty(0)
    )
    quaternion = trace["root_quaternion_w"][settled]
    upright = 1 - 2 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2)
    return dict(
        settled_samples=int(settled.sum()),
        prescribed_settled_samples=550,
        observed_motion_duration_s=int(motion.sum()) * PERIOD_S,
        prescribed_motion_duration_s=12.0,
        actual_onset_forward_travel_m=travel,
        maximum_onset_xy_excursion_m=excursion,
        settled_mean_forward_velocity_m_s=_mean(
            trace["root_com_linear_velocity_b"][settled, 0]
        ),
        settled_mean_root_height_above_plane_m=_mean(height),
        settled_minimum_root_height_above_plane_m=(
            float(height.min()) if height.size else None
        ),
        settled_mean_root_upright_z=_mean(upright),
        settled_foot_contact_fraction=_mean(contact),
        settled_any_foot_contact_fraction=_mean(contact.any(-1)),
        settled_supported_foot_contact_fraction=_mean(supported),
        settled_surface_sample_valid_fraction=_mean(valid),
    )


def summarize_flat_entry(folder):
    """Verify and describe the existing two validation cases from safe NumPy traces.

    Measurements condition on available original-attempt samples. A low posture,
    missing contact or unavailable phase remains visible; no pass gate is inferred.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    cases = tuple(case_from_dict(value) for value in manifest["cases"])
    expected = {
        EvaluationCase("validation", "Flat", None, "plane", profile).case_id
        for profile in ("stand", "forward-035")
    }
    if len(cases) != 2 or {case.case_id for case in cases} != expected:
        raise ValueError("Flat-entry requires the two current Flat validation controls")
    report = score_capture(folder)
    rows = {}
    for case, result in zip(cases, report["cases"]):
        if result is None:
            rows[case.profile] = None
            continue
        with np.load(
            folder / case.case_id / "trace.npz", allow_pickle=False
        ) as archive:
            trace = validate_trace(case, {key: archive[key] for key in archive.files})
        rows[case.profile] = dict(
            case_id=case.case_id,
            seed=case.seed,
            capture_valid=result["capture_valid"],
            core_return=result["core_return"],
            duration_s=result["duration_s"],
            physical_failure=result["physical_failure"],
            first_ending=result["first_ending"],
            settled_tracking=result["phases"]["settled_motion"],
            stopping=result["stopping"],
            requested_forward_travel_m=case.command_at(3.0)[0] * 12,
            **_measurements(trace),
        )

    def contrast(name):
        values = [
            rows[profile].get(name) if rows[profile] else None
            for profile in ("stand", "forward-035")
        ]
        return values[1] - values[0] if all(v is not None for v in values) else None

    return dict(
        version="go2_flat_entry_diagnostic_v1",
        case_version="v2",
        training_seed=report["training_seed"],
        checkpoint_id=report["checkpoint_id"],
        checkpoint_sha256=report["checkpoint_sha256"],
        policy_state_sha256=report["policy_state_sha256"],
        diagnostic_only=True,
        calibration_established=False,
        capture_valid=report["capture_valid"],
        full_bank=False,
        valid_for_comparison=False,
        conditional_on_available_phases=True,
        contact_force_threshold_N=CONTACT_THRESHOLD_N,
        maximum_foot_center_clearance_m=MAXIMUM_FOOT_CENTER_CLEARANCE_M,
        posture_reference="root link origin above the Flat plane; upright is root Z dot world Z",
        cases=rows,
        command_contrast=dict(
            forward_minus_stand_velocity_m_s=contrast(
                "settled_mean_forward_velocity_m_s"
            ),
            forward_minus_stand_travel_m=contrast("actual_onset_forward_travel_m"),
        ),
    )

"""Conservative contact-supported terrain exposure; absence of proof stays visible."""

import numpy as np

from parkour_lab.control.proprioception import CONTACT_THRESHOLD_N

HEIGHT_CHANGE_M = 0.0025
MAXIMUM_FOOT_CENTER_CLEARANCE_M = 0.08
PLATFORM_OUTER_HALF_WIDTH_M = 1.3


def surface_exposure(case, trace):
    height_change = HEIGHT_CHANGE_M if case.version == "v1" else 0.0005
    support, valid = trace["support_height_w"], trace["support_valid"]
    contact = (
        np.linalg.norm(trace["foot_contact_forces_w"], axis=-1) > CONTACT_THRESHOLD_N
    )
    clearance = trace["foot_position_w"][..., 2] - support
    supported = contact & valid & (np.abs(clearance) <= MAXIMUM_FOOT_CENTER_CLEARANCE_M)
    initial, initial_valid = (
        trace["initial_support_height_w"],
        trace["initial_support_valid"],
    )
    changed = (
        supported
        & initial_valid[None]
        & (np.abs(support - initial[None]) > height_change)
    )
    local_span = np.zeros(len(support), dtype=bool)
    for index, mask in enumerate(supported):
        if mask.sum() > 1:
            local_span[index] = np.ptp(support[index, mask]) > height_change
    exposed = changed.any(-1) | local_span
    contacting = supported.any(-1)
    outside = platform_only = None
    if "env_origin_w" in trace and supported.any():
        xy = trace["foot_position_w"][..., :2] - trace["env_origin_w"][:2]
        radius = np.abs(xy).max(-1)[supported]
        outside = float((radius > PLATFORM_OUTER_HALF_WIDTH_M).mean())
        platform_only = bool((radius <= 1.0).all())
    if case.family == "Flat":
        label = "flat"
    elif (
        case.version == "v2"
        and case.physical_level == 0
        and case.condition in ("id", "composition")
    ):
        label = "flat_entry"
    elif exposed.any():
        label = "nonflat_contact_demonstrated"
    elif contacting.any() and platform_only and case.family in ("Slopes", "Stairs"):
        label = "platform_only_supported_contacts"
    elif contacting.any():
        label = "no_nonflat_contact_demonstrated"
    else:
        label = "unavailable_no_supported_contacts"
    return dict(
        label=label,
        contact_force_threshold_N=CONTACT_THRESHOLD_N,
        support_height_change_threshold_m=height_change,
        maximum_foot_center_clearance_m=MAXIMUM_FOOT_CENTER_CLEARANCE_M,
        surface_sample_valid_fraction=float(valid.mean()) if valid.size else None,
        supported_contact_samples=int(supported.sum()),
        contact_control_samples=int(contacting.sum()),
        demonstrated_nonflat_control_samples=int(exposed.sum()),
        demonstrated_nonflat_fraction_of_contact_samples=(
            float(exposed[contacting].mean()) if contacting.any() else None
        ),
        outside_platform_fraction_of_supported_feet=outside,
        central_platform_half_width_m=(
            1.0
            if case.family in ("Slopes", "Stairs") and label != "flat_entry"
            else None
        ),
        outside_platform_buffer_m=PLATFORM_OUTER_HALF_WIDTH_M - 1.0,
        interpretation=(
            "Version 2 level zero is a true-plane entry control; composition adds no roughness. It is not evidence of nonflat traversal."
            if label == "flat_entry"
            else "Contact plus changed/local supporting heights is conservative evidence; a family label or missing evidence is not traversability."
        ),
    )

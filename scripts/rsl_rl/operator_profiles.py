"""Small, versioned refinements of the stock operator learning objective.

No simulator imports or YAML callable construction. Recognizing a reward width
does NOT validate an environment: reference_config still compares the complete
physical, observation, action, event, termination and reward contract.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RefinementProfile:
    name: str
    yaw_tracking_std: float
    entropy_coef: float


PROFILES = {
    profile.name: profile
    for profile in (
        RefinementProfile("stock", yaw_tracking_std=0.5, entropy_coef=0.01),
        RefinementProfile("yaw_precision_v1", yaw_tracking_std=0.2, entropy_coef=0.001),
        RefinementProfile("low_entropy_v1", yaw_tracking_std=0.5, entropy_coef=0.001),
    )
}


def environment_profile(saved):
    """Return a representative of a known yaw kernel, NOT a full learning profile.

    Stock and low-entropy training share the same environment reward contract.
    Only source_profile, with the agent config, can distinguish those two.
    """
    try:
        value = saved["rewards"]["track_ang_vel_z_exp"]["params"]["std"]
        width = float(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Missing or invalid source yaw-tracking reward width"
        ) from error
    for profile in PROFILES.values():
        if width == profile.yaw_tracking_std:
            return profile
    raise ValueError(f"Unsupported source yaw-tracking reward width: {value}")


def source_profile(saved, agent):
    """Accept only a known reward/entropy pair before starting a training worker."""
    width = environment_profile(saved).yaw_tracking_std
    try:
        entropy = float(agent["algorithm"]["entropy_coef"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid source entropy coefficient") from error
    for profile in PROFILES.values():
        if (width, entropy) == (profile.yaw_tracking_std, profile.entropy_coef):
            return profile
    raise ValueError(
        f"Unsupported source yaw-reward/entropy pair: std={width}, entropy={entropy}"
    )


def select_profile(saved, agent, requested="source"):
    source = source_profile(saved, agent)
    if requested == "source":
        return source
    if requested not in PROFILES:
        raise ValueError(f"Unsupported refinement profile: {requested}")
    return PROFILES[requested]


def apply_reward_profile(cfg, profile):
    """The only permitted reward change: width, not weight, function or gating."""
    cfg.rewards.track_ang_vel_z_exp.params["std"] = profile.yaw_tracking_std


def profile_manifest(saved, agent, requested="source"):
    source = source_profile(saved, agent)
    selected = select_profile(saved, agent, requested)
    parameters = {
        "rewards.track_ang_vel_z_exp.params.std": (
            source.yaw_tracking_std,
            selected.yaw_tracking_std,
        ),
        "algorithm.entropy_coef": (source.entropy_coef, selected.entropy_coef),
    }
    changes = {
        name: {"source": before, "selected": after}
        for name, (before, after) in parameters.items()
        if before != after
    }
    if len(changes) == 2:
        interpretation = "Joint yaw-objective/entropy refinement, not a single-factor causal ablation. "
    elif "algorithm.entropy_coef" in changes:
        interpretation = "Entropy-only profile change; the source reward is unchanged. "
    elif changes:
        interpretation = "Yaw-kernel-only profile change; the source entropy coefficient is unchanged. "
    else:
        interpretation = "Source learning profile preserved. "
    return {
        "source": asdict(source),
        "selected": asdict(selected),
        "changed": source != selected,
        "changed_parameters": changes,
        "reward_parameters_changed": source.yaw_tracking_std
        != selected.yaw_tracking_std,
        "initial_policy_and_noise": "exact source tensors; no action-noise reset or clamp",
        "reward_weights_changed": False,
        "benchmark_thresholds_changed": False,
        "interpretation": (
            interpretation
            + "Additional training and a fresh optimizer prevent attributing changes to a "
            "profile alone without a matched continuation. "
            "Lower entropy pressure does not guarantee that learned action noise decreases. "
            "Compare physical screening metrics; reward integrals use different kernels."
        ),
    }

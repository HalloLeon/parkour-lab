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
    )
}


def environment_profile(saved):
    """Recognize only the two supported yaw kernels; reject unknown/missing data."""
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
    profile = environment_profile(saved)
    try:
        entropy = float(agent["algorithm"]["entropy_coef"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid source entropy coefficient") from error
    if entropy != profile.entropy_coef:
        raise ValueError(f"Source entropy coefficient does not match {profile.name}")
    return profile


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
    return {
        "source": asdict(source),
        "selected": asdict(selected),
        "changed": source != selected,
        "initial_policy_and_noise": "exact source tensors; no action-noise reset or clamp",
        "reward_weights_changed": False,
        "benchmark_thresholds_changed": False,
        "interpretation": (
            "Joint yaw-objective/entropy refinement, not a single-factor causal ablation. "
            "Lower entropy pressure does not guarantee that learned action noise decreases. "
            "Compare physical screening metrics; reward integrals use different kernels."
        ),
    }

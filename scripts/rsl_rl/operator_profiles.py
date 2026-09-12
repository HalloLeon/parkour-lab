"""Small, versioned refinements of the stock operator learning objective.

No simulator imports or YAML callable construction. Recognizing a reward signature
does NOT validate an environment: reference_config still compares the complete
physical, observation, action, event, termination and reward contract.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RefinementProfile:
    name: str
    yaw_tracking_std: float
    entropy_coef: float
    stationary_precision: bool = False


STOCK_FUNCTIONS = {
    "track_lin_vel_xy_exp": "isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp",
    "track_ang_vel_z_exp": "isaaclab.envs.mdp.rewards:track_ang_vel_z_exp",
}
STATIONARY_FUNCTIONS = {
    "track_lin_vel_xy_exp": "track_lin_vel_xy_stationary",
    "track_ang_vel_z_exp": "track_ang_vel_z_stopped",
}
STATIONARY_STDS = {"track_lin_vel_xy_exp": 0.05, "track_ang_vel_z_exp": 0.1}
PRECISION_FRACTION = 1.0 / 3.0


PROFILES = {
    profile.name: profile
    for profile in (
        RefinementProfile("stock", yaw_tracking_std=0.5, entropy_coef=0.01),
        RefinementProfile("yaw_precision_v1", yaw_tracking_std=0.2, entropy_coef=0.001),
        RefinementProfile("low_entropy_v1", yaw_tracking_std=0.5, entropy_coef=0.001),
        RefinementProfile(
            "stationary_twist_v1",
            yaw_tracking_std=0.5,
            entropy_coef=0.001,
            stationary_precision=True,
        ),
    )
}


def environment_profile(saved):
    """Recognize a known reward signature, NOT a complete environment or agent.

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
    try:
        terms = saved["rewards"]
        stationary = any(
            "stationary_std" in terms.get(name, {}).get("params", {})
            or "precision_fraction" in terms.get(name, {}).get("params", {})
            or str(terms.get(name, {}).get("func", "")).endswith(":" + function)
            for name, function in STATIONARY_FUNCTIONS.items()
        )
        if stationary:
            for name, function in STATIONARY_FUNCTIONS.items():
                term = terms[name]
                if (
                    term["func"]
                    not in (
                        f"operator_rewards:{function}",
                        f"scripts.rsl_rl.operator_rewards:{function}",
                    )
                    or float(term["params"]["std"]) != 0.5
                    or float(term["params"]["stationary_std"]) != STATIONARY_STDS[name]
                    or float(term["params"]["precision_fraction"]) != PRECISION_FRACTION
                ):
                    raise ValueError("Unsupported stationary tracking reward contract")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("Unsupported stationary tracking reward contract") from error
    for profile in PROFILES.values():
        if (
            width == profile.yaw_tracking_std
            and stationary == profile.stationary_precision
        ):
            return profile
    raise ValueError(f"Unsupported source yaw-tracking reward width: {value}")


def source_profile(saved, agent):
    """Accept only a known reward-signature/entropy pair before starting a worker."""
    environment = environment_profile(saved)
    width = environment.yaw_tracking_std
    try:
        entropy = float(agent["algorithm"]["entropy_coef"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid source entropy coefficient") from error
    for profile in PROFILES.values():
        if (width, entropy, environment.stationary_precision) == (
            profile.yaw_tracking_std,
            profile.entropy_coef,
            profile.stationary_precision,
        ):
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
    """Reconstruct a known objective; never load a saved function or parameter."""
    cfg.rewards.track_ang_vel_z_exp.params["std"] = profile.yaw_tracking_std
    if profile.stationary_precision:
        try:
            from . import operator_rewards
        except ImportError:
            import operator_rewards
        for name, function in STATIONARY_FUNCTIONS.items():
            term = getattr(cfg.rewards, name)
            term.func = getattr(operator_rewards, function)
            term.params.update(
                stationary_std=STATIONARY_STDS[name],
                precision_fraction=PRECISION_FRACTION,
            )
    else:
        # Explicitly switching back from a known stationary source is allowed.
        for name in STATIONARY_FUNCTIONS:
            term = getattr(cfg.rewards, name, None)
            if term is not None and "stationary_std" in term.params:
                term.func = STOCK_FUNCTIONS[name]
                term.params.pop("stationary_std")
                term.params.pop("precision_fraction")


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
    for name, function in STATIONARY_FUNCTIONS.items():
        for key, value in {
            "func": f"operator_rewards:{function}",
            "params.stationary_std": STATIONARY_STDS[name],
            "params.precision_fraction": PRECISION_FRACTION,
        }.items():
            original = STOCK_FUNCTIONS[name] if key == "func" else None
            parameters[f"rewards.{name}.{key}"] = (
                value if source.stationary_precision else original,
                value if selected.stationary_precision else original,
            )
    changes = {
        name: {"source": before, "selected": after}
        for name, (before, after) in parameters.items()
        if before != after
    }
    if source.stationary_precision != selected.stationary_precision:
        interpretation = (
            "Command-gated stationary objective change; no position/heading reference "
            "or inference assistance. "
        )
    elif len(changes) == 2:
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
        "reward_parameters_changed": any(
            name.startswith("rewards.") for name in changes
        ),
        "reward_functions_changed": source.stationary_precision
        != selected.stationary_precision,
        "stationary_precision": {
            "enabled": selected.stationary_precision,
            "planar_gate": "exact zero commanded vx and vy, both yaw signs",
            "yaw_gate": "exact zero commanded vx, vy and wz",
            "broad_fraction": 1 - PRECISION_FRACTION,
            "fine_fraction": PRECISION_FRACTION,
            "planar_fine_std_m_s": STATIONARY_STDS["track_lin_vel_xy_exp"],
            "yaw_fine_std_rad_s": STATIONARY_STDS["track_ang_vel_z_exp"],
            "reward_peaks_changed": False,
            "absolute_pose_hold": False,
        },
        "initial_policy_and_noise": "exact source tensors; no action-noise reset or clamp",
        "reward_weights_changed": False,
        "benchmark_thresholds_changed": False,
        "interpretation": (
            interpretation
            + "Additional training and a fresh optimizer prevent attributing changes to a "
            "profile alone without a matched continuation. "
            "Lower entropy pressure does not guarantee that learned action noise decreases. "
            "Compare physical screening metrics; reward integrals are not comparable "
            "when the reward kernels change."
        ),
    }

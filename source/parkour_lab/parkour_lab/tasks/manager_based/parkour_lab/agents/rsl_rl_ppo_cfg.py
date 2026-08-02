from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from parkour_lab.learning.distillation.architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
)


@configclass
class PrivilegedTeacherActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the modular Phase-1 teacher actor."""

    class_name: str = "PrivilegedTeacherActorCritic"

    # Compress the concatenated privileged heights and validity mask to the
    # fixed representation consumed by the transferable motor actor.
    terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM

    # Hidden-layer widths used before projecting that scan to the latent.
    scan_encoder_hidden_dims: list[int] = [128, 64]

    # Privileged simulator dynamics are compressed to the motor's fixed
    # adaptation latent during teacher training.
    dynamics_encoder_hidden_dims: list[int] = [128, 64]

    # The deployable encoder learns the same latent from manager-owned
    # proprioception/action history.
    history_encoder_hidden_dims: list[int] = [256, 128]


@configclass
class RegularizedPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO with bidirectional regularized online adaptation."""

    class_name: str = "RegularizedPPO"

    # Scale applied to the history encoder's Smooth L1 regression objective.
    adaptation_loss_coef: float = 1.0

    # Every twentieth rollout executes actions from the history encoder. This
    # exposes that deployable path to the states induced by its own predictions.
    history_rollout_interval: int = 20

    # Increase the reverse, privileged-encoder regularizer from zero to 0.1.
    # Delaying it lets PPO first discover a useful privileged representation;
    # the later ramp makes that representation reproducible from robot history.
    privileged_regularization_coef_start: float = 0.0
    privileged_regularization_coef_end: float = 0.1
    privileged_regularization_warmup_iterations: int = 200
    privileged_regularization_ramp_iterations: int = 300


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the privileged terrain-aware teacher."""

    # Number of environment steps collected from each parallel environment
    # before one PPO update is performed. The total rollout size per iteration
    # is ``num_envs * num_steps_per_env``.
    num_steps_per_env = 24

    # Maximum number of PPO iterations. Each iteration collects a rollout,
    # computes returns and advantages, and updates the policy and value model.
    max_iterations = 150

    # Number of PPO iterations between checkpoints.
    save_interval = 50

    # Experiment directory name used beneath ``logs/rsl_rl``.
    experiment_name = "parkour_lab"

    # Logging backend used for training metrics.
    logger = "tensorboard"

    # Dictionary keys name RSL-RL network inputs; list entries name Isaac Lab
    # observation groups declared on ObservationsCfg in parkour_lab_env_cfg.py:
    # policy -> DeployablePolicyCfg, heading_target -> OracleHeadingTargetCfg,
    # terrain -> PrivilegedTerrainCfg, dynamics -> PrivilegedDynamicsCfg, and
    # critic_privileged -> CriticPrivilegedCfg.
    obs_groups = {
        # RSL-RL calls the action-producing actor input "policy". Its counterpart
        # below is the actor_* configuration on RslRlPpoActorCriticCfg. The
        # identically named list entry is instead ObservationsCfg.policy.
        "policy": ["policy", "heading_target", "terrain", "dynamics"],
        # The "critic" input feeds the value-estimating network configured by
        # critic_* below. It sees every actor group plus simulator-only state.
        "critic": [
            "policy",
            "heading_target",
            "terrain",
            "dynamics",
            "critic_privileged",
        ],
    }

    # RSL-RL 3 configures the custom actor and standard critic through one
    # actor-critic policy object.
    policy = PrivilegedTeacherActorCriticCfg(
        # Initial standard deviation of the Gaussian action distribution. This
        # controls exploration before the standard deviation is learned.
        init_noise_std=1.0,
        # Keep learned running statistics disabled: the deployable student and
        # exported motor actor do not carry an RSL-RL normalizer. Every actor
        # observation must instead use a fixed, deployment-reproducible scale at
        # its observation source. Apply the same convention to critic inputs so
        # checkpoint behavior does not depend on hidden normalization state.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Hidden-layer widths of the shared, directly transferable motor actor.
        actor_hidden_dims=[512, 256, 128],
        # Hidden-layer widths of the value-estimating critic network.
        critic_hidden_dims=[512, 256, 128],
        # Nonlinear activation used after each hidden layer.
        activation="elu",
    )

    # PPO optimization settings shared by the teacher and both ablations.
    algorithm = RegularizedPPOCfg(
        # Weight of the critic loss relative to the policy objective.
        value_loss_coef=1.0,
        # Clip value-function changes to avoid excessively large updates.
        use_clipped_value_loss=True,
        # Maximum probability-ratio deviation allowed by the PPO surrogate
        # objective during one update.
        clip_param=0.2,
        # Weight of the entropy bonus that encourages action exploration.
        entropy_coef=0.005,
        # Number of passes over each collected rollout.
        num_learning_epochs=5,
        # Number of minibatches used for every learning epoch.
        num_mini_batches=4,
        # Initial optimizer learning rate.
        learning_rate=1.0e-3,
        # Adapt the learning rate using the measured KL divergence.
        schedule="adaptive",
        # Discount factor applied to future rewards.
        gamma=0.99,
        # Generalized Advantage Estimation bias-variance parameter.
        lam=0.95,
        # Target KL divergence used by the adaptive learning-rate schedule.
        desired_kl=0.01,
        # Maximum gradient norm used for gradient clipping.
        max_grad_norm=1.0,
    )


@configclass
class PPOPrivilegedCriticRunnerCfg(PPORunnerCfg):
    """Ablation with terrain available only to the critic."""

    # Identify this routing variant in its run-directory name.
    run_name = "privileged_critic"

    # Keep terrain privileged to the value function. The actor still receives
    # the oracle heading required by every Phase-1 teacher variant.
    obs_groups = {
        "policy": ["policy", "heading_target", "dynamics"],
        "critic": [
            "policy",
            "heading_target",
            "terrain",
            "dynamics",
            "critic_privileged",
        ],
    }


@configclass
class PPOBaselineRunnerCfg(PPORunnerCfg):
    """Ablation without terrain observations."""

    # Identify this routing variant in its run-directory name.
    run_name = "baseline_no_terrain"

    # Remove terrain from both networks while retaining the critic-only state.
    obs_groups = {
        "policy": ["policy", "heading_target", "dynamics"],
        "critic": [
            "policy",
            "heading_target",
            "dynamics",
            "critic_privileged",
        ],
    }

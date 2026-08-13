from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from parkour_lab.learning.distillation.architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
)
from parkour_lab.learning.distillation.teacher.symmetry import (
    compute_symmetric_states,
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

    # Keep exploration broad enough to discover motion while allowing the
    # learned distribution to settle around a useful deterministic policy.
    min_noise_std: float = 0.10
    max_noise_std: float = 0.80


@configclass
class RegularizedPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO with left-right augmentation and regularized online adaptation."""

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

    # Do not let the adaptive schedule raise the shared optimizer above its
    # initial rate during long runs, including for the adaptation encoders.
    max_learning_rate: float = 3.0e-4


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the privileged terrain-aware teacher."""

    # Number of environment steps collected from each parallel environment
    # before one PPO update is performed. The total rollout size per iteration
    # is ``num_envs * num_steps_per_env``. Forty-eight steps provide a longer
    # temporal window for obstacle approach, traversal, and landing credit.
    num_steps_per_env = 48

    # Nominal Phase-1 training budget. This is long enough for environments to
    # revisit the curriculum after bootstrap; 150 iterations is only a smoke
    # test and must not be treated as an obstacle-mastery budget.
    max_iterations = 1000

    # Number of PPO iterations between checkpoints. This retains ten evenly
    # spaced snapshots over the nominal training run.
    save_interval = 100

    # Experiment directory name used beneath ``logs/rsl_rl``.
    experiment_name = "parkour_lab"

    # Logging backend used for training metrics.
    logger = "tensorboard"

    # Bound the action that reaches Isaac Lab's ActionManager. This also bounds
    # last_action observations and the squared action-rate reward, preventing a
    # rare policy outlier from feeding back through the actor, critic, and
    # adaptation history.
    clip_actions = 5.0

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
        init_noise_std=0.8,
        # Keep the direct parameter used by existing checkpoints and their Adam
        # state; the custom actor projects it into positive safe bounds.
        noise_std_type="scalar",
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
        # Retain enough exploration to acquire locomotion before the terrain
        # curriculum exposes obstacle-specific behavior.
        entropy_coef=0.01,
        # Number of passes over each collected rollout.
        num_learning_epochs=5,
        # Number of minibatches used for every learning epoch.
        num_mini_batches=4,
        # Initial optimizer learning rate.
        learning_rate=3.0e-4,
        # Adapt the learning rate using the measured KL divergence.
        schedule="adaptive",
        # Discount factor applied to future rewards. The longer horizon retains
        # useful credit from obstacle approach through landing and completion.
        gamma=0.995,
        # Generalized Advantage Estimation bias-variance parameter.
        lam=0.95,
        # Target KL divergence used by the adaptive learning-rate schedule.
        desired_kl=0.01,
        # Maximum gradient norm used for gradient clipping.
        max_grad_norm=1.0,
        # Reflect complete left-right transitions during PPO updates. Data
        # augmentation imposes no gait phase and leaves mirror loss disabled.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
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

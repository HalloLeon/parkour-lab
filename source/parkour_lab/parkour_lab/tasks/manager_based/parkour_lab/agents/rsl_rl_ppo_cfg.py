from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


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

    # The policy receives deployable state and terrain geometry. The critic
    # receives the same input followed by its simulator-only state.
    obs_groups = {
        "policy": ["policy", "terrain"],
        "critic": ["policy", "terrain", "critic_privileged"],
    }

    # RSL-RL 3 configures both networks through one actor-critic policy object.
    policy = RslRlPpoActorCriticCfg(
        # Initial standard deviation of the Gaussian action distribution. This
        # controls exploration before the standard deviation is learned.
        init_noise_std=1.0,
        # Observations already use deliberate physical scaling, so additional
        # running normalization remains disabled for both networks.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Hidden-layer widths of the action-producing actor network.
        actor_hidden_dims=[512, 256, 128],
        # Hidden-layer widths of the value-estimating critic network.
        critic_hidden_dims=[512, 256, 128],
        # Nonlinear activation used after each hidden layer.
        activation="elu",
    )

    # PPO optimization settings shared by the teacher and both ablations.
    algorithm = RslRlPpoAlgorithmCfg(
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

    # Keep terrain privileged to the value function so the actor must act from
    # the deployable observation core alone.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "terrain", "critic_privileged"],
    }


@configclass
class PPOBaselineRunnerCfg(PPORunnerCfg):
    """Ablation without terrain observations."""

    # Identify this routing variant in its run-directory name.
    run_name = "baseline_no_terrain"

    # Remove terrain from both networks while retaining the critic-only state.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic_privileged"],
    }

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Number of environment steps collected from each parallel environment
    # before one PPO update is performed.
    #
    # Total rollout size per PPO iteration is:
    #     num_envs * num_steps_per_env
    #
    # Larger values give PPO more data per update, but require more memory
    # and may make each update slower.
    num_steps_per_env = 24

    # Maximum number of PPO training iterations.
    #
    # One iteration usually means:
    #   1. collect rollouts from the environment
    #   2. compute returns and advantages
    #   3. update the policy and value function
    #
    # Total environment steps are approximately:
    #     num_envs * num_steps_per_env * max_iterations
    max_iterations = 150

    # Save a policy checkpoint every N PPO iterations.
    #
    # With save_interval=50 and max_iterations=150, checkpoints will be saved
    # around iterations 50, 100, and 150.
    save_interval = 50

    # Name of the experiment used for organizing logs and checkpoints.
    #
    # Isaac Lab/RSL-RL will group training outputs under this experiment name,
    # making it easier to separate this run from other tasks or agents.
    experiment_name = "parkour_lab"

    # Logging backend used during training.
    #
    # "tensorboard" writes logs that can be viewed with TensorBoard.
    # Other supported backends may include "wandb" or "neptune", depending on
    # your Isaac Lab/RSL-RL setup.
    logger = "tensorboard"

    # Enables runtime checks for NaN values coming from the environment.
    #
    # This is useful during development because NaNs in observations, rewards,
    # actions, or critic targets can quickly destroy PPO training.
    check_for_nan = True

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    # Configuration for the actor-critic policy network used by PPO.
    #
    # The actor predicts actions.
    # The critic estimates the value function.
    policy = RslRlPpoActorCriticCfg(
        # Initial standard deviation of the action distribution.
        #
        # PPO samples actions from a stochastic policy during training.
        # A higher value encourages broader initial exploration.
        # A lower value makes actions less random at the beginning.
        init_noise_std=1.0,
        # Hidden layer sizes for the actor network.
        #
        # The actor maps observations to actions.
        # This creates an MLP with layers:
        #     input -> 512 -> 256 -> 128 -> action output
        #
        # Larger networks can represent more complex policies but require more
        # computation and may overfit or train less stably if oversized.
        actor_hidden_dims=[512, 256, 128],
        # Hidden layer sizes for the critic network.
        #
        # The critic maps observations, or privileged critic observations if
        # provided by the environment, to a scalar value estimate.
        #
        # This creates an MLP with layers:
        #     input -> 512 -> 256 -> 128 -> value output
        critic_hidden_dims=[512, 256, 128],
        # Activation function used between hidden layers in both actor and critic.
        #
        # "elu" is commonly used in locomotion-style PPO because it is smooth
        # and handles negative activations better than plain ReLU.
        activation="elu",
    )

    # Configuration for the PPO optimization algorithm.
    algorithm = RslRlPpoAlgorithmCfg(
        # Weight applied to the critic/value-function loss.
        #
        # Higher values make the optimizer focus more on accurate value
        # prediction. Lower values reduce critic influence relative to the
        # actor policy loss.
        value_loss_coef=1.0,
        # Whether to clip the value-function update, similar to PPO policy
        # clipping.
        #
        # This prevents the critic value estimate from changing too much in a
        # single update, which can improve training stability.
        use_clipped_value_loss=True,
        # PPO clipping range for policy updates.
        #
        # PPO limits how much the new policy is allowed to differ from the old
        # policy during an update. A value of 0.2 is a common default.
        #
        # Smaller values make updates more conservative.
        # Larger values allow more aggressive policy changes.
        clip_param=0.2,
        # Weight applied to the entropy bonus.
        #
        # Entropy encourages exploration by rewarding less-certain action
        # distributions. Higher values keep the policy more exploratory.
        # Lower values allow the policy to become deterministic sooner.
        entropy_coef=0.005,
        # Number of times PPO reuses the collected rollout data for learning
        # during each update.
        #
        # More epochs can improve sample efficiency, but too many may overfit
        # to stale rollout data and destabilize PPO.
        num_learning_epochs=5,
        # Number of mini-batches the rollout data is split into for each
        # learning epoch.
        #
        # With num_mini_batches=4, each epoch performs 4 gradient updates using
        # different chunks of the collected rollout batch.
        num_mini_batches=4,
        # Optimizer learning rate.
        #
        # This controls the step size for updating the actor and critic neural
        # network weights.
        learning_rate=1.0e-3,
        # Learning-rate schedule.
        #
        # "adaptive" typically adjusts the learning rate based on the observed
        # KL divergence between the old and updated policies.
        schedule="adaptive",
        # Discount factor for future rewards.
        #
        # gamma=0.99 means rewards far into the future still matter, but are
        # slightly discounted at each timestep.
        #
        # Higher gamma values encourage long-horizon behavior.
        # Lower gamma values focus more on immediate rewards.
        gamma=0.99,
        # Lambda parameter for Generalized Advantage Estimation, or GAE.
        #
        # Controls the bias-variance tradeoff in advantage estimation.
        #
        # lam close to 1.0 gives lower bias but higher variance.
        # lam closer to 0.0 gives higher bias but lower variance.
        lam=0.95,
        # Target KL divergence used by the adaptive learning-rate schedule.
        #
        # KL divergence measures how much the policy changed after an update.
        # If the policy changes too much, the learning rate may be reduced.
        # If it changes too little, the learning rate may be increased.
        desired_kl=0.01,
        # Maximum gradient norm used for gradient clipping.
        #
        # If gradients become too large, they are scaled down so their norm does
        # not exceed this value. This helps prevent unstable updates.
        max_grad_norm=1.0,
    )

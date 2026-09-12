"""Isaac Lab binding for v3 training-only sequence coverage; import after Kit."""

try:
    from .operator_command import OperatorTransitionCommand
    from .operator_sequences import ReversalSequencePlan
except ImportError:
    from operator_command import OperatorTransitionCommand
    from operator_sequences import ReversalSequencePlan


class OperatorReversalSequenceCommand(OperatorTransitionCommand):
    """Keep v2 sampling for background episodes and after a sequence finishes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sequence_plan = ReversalSequencePlan(self.num_envs, self.device)

    def _resample_command(self, env_ids):
        if not len(env_ids):
            return
        new_episode = self.command_counter[env_ids] == 0
        super()._resample_command(env_ids)
        selected, commands, categories, seconds = self.sequence_plan.resample(
            env_ids, new_episode
        )
        self.category[selected] = categories
        self.vel_command_b[selected] = commands
        self.is_standing_env[selected] = categories == 0
        self.time_left[selected] = seconds

    def __str__(self):
        return "OperatorReversalSequenceCommand (v3): randomized training-only chains plus v2"

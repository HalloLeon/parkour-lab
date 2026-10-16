"""Real native collection accounting, independent of any learner or checkpoint."""


def nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class InteractionCounter:
    """Charge all advancing replicas, including discarded and uncertain collection.

    The SDK's control counter advances after a full physics batch, before reward,
    reset and diagnostic reads. A failed partial batch is conservatively charged
    as one full vector batch. This counter is not a crash-proof event journal.
    """

    def __init__(self, num_envs):
        if nonnegative_integer(num_envs, "num_envs") == 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.session = self.cumulative = self.uncertain = 0
        self._native_steps = self._native_ticks = 0

    def sync(self, native_control_steps, native_physics_ticks, *, failed=False):
        steps = nonnegative_integer(native_control_steps, "native control steps")
        ticks = nonnegative_integer(native_physics_ticks, "native physics ticks")
        if type(failed) is not bool:
            raise ValueError("failed must be boolean")
        if steps < self._native_steps or ticks < self._native_ticks:
            raise ValueError("Native counters cannot decrease within a session")
        completed = steps - self._native_steps
        partial = failed and not completed and ticks > self._native_ticks
        charge = (completed + int(partial)) * self.num_envs
        self.session += charge
        self.cumulative += charge
        self.uncertain += int(partial) * self.num_envs
        self._native_steps, self._native_ticks = steps, ticks

    def restore(self, cumulative, *, uncertain=0):
        """Restore prior-run spend before this recreated environment collects data."""
        total = nonnegative_integer(cumulative, "cumulative interactions")
        unknown = nonnegative_integer(uncertain, "uncertain interactions")
        if self.session or total < self.cumulative or unknown < self.uncertain:
            raise ValueError("Restore before collection without reducing prior spend")
        if unknown > total:
            raise ValueError("Uncertain interactions cannot exceed total spend")
        self.cumulative, self.uncertain = total, unknown

    def state_dict(self):
        return dict(
            session=self.session, cumulative=self.cumulative, uncertain=self.uncertain
        )

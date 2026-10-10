"""Shared affine action delivery; PPO samples remain upstream of this transform."""

import torch

BOUNDED_MODE = "joint_limits_v1"
BOUNDED_RAW_ACTION_MEANING = (
    "unscaled policy request; q_target = default_q + 0.25 * delivered_action; "
    "delivered_action bounded by archived joint target limits; "
    "previous action and action-rate reward use delivered_action"
)


class JointActionTransform:
    """Bound encoded actions so even rounded affine targets stay inside the limits."""

    def __init__(self, default, target_limits_rad=None):
        self.default = default
        self.limits = self.lower = self.upper = None
        if target_limits_rad is not None:
            self.limits = default.new_tensor(target_limits_rad)
            self.lower = (self.limits[:, 0] - default) / 0.25
            self.upper = (self.limits[:, 1] - default) / 0.25
            # Division by 0.25 is exact; subtraction/addition may round outward.
            self.lower = torch.where(
                default + 0.25 * self.lower < self.limits[:, 0],
                torch.nextafter(self.lower, torch.full_like(self.lower, torch.inf)),
                self.lower,
            )
            self.upper = torch.where(
                default + 0.25 * self.upper > self.limits[:, 1],
                torch.nextafter(self.upper, torch.full_like(self.upper, -torch.inf)),
                self.upper,
            )

    def __call__(self, raw):
        delivered = raw if self.limits is None else raw.clamp(self.lower, self.upper)
        return delivered, self.default + 0.25 * delivered

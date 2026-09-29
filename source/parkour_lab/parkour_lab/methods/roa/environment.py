"""Native observation/motor delivery for the current ROA method.

History and privileged labels are method-owned. The shared controller runtime
remains independent of this training observation schema.
"""


def read_dynamics(env):
    """Actual startup properties, not sampled requests or effective contact friction."""
    import torch

    robot = env.scene["robot"]
    base = robot.body_names.index("base")
    mass = robot.root_physx_view.get_masses()[:, base : base + 1].to(env.device)
    nominal = robot.data.default_mass[:, base : base + 1].to(env.device)
    com = robot.root_physx_view.get_coms()[:, base, :3].to(env.device)
    material = (
        robot.root_physx_view.get_material_properties().to(env.device).mean(dim=1)
    )
    if (nominal <= 0).any() or (mass <= 0).any():
        raise ValueError("Invalid native mass")
    result = torch.cat((mass / nominal - 1.0, com, material), dim=1).float()
    if result.shape != (env.num_envs, 7) or not torch.isfinite(result).all():
        raise ValueError("Invalid measured dynamics vector")
    return result.detach().clone()


class ROAEnvironment:
    """ROA history and privileged features over shared native control ticks."""

    def __init__(self, host, *, contact_conditioned=False):
        from parkour_lab.methods.roa.model import CausalHistory

        self.host, self.env = host, host.env
        self.history = CausalHistory()
        self.dynamics = read_dynamics(self.env)
        self.contacts = None
        if contact_conditioned:
            from parkour_lab.methods.roa.contacts import ContactFeatures

            self.contacts = ContactFeatures(self.env)

    def observations(self, native, reset):
        from tensordict import TensorDict

        frame = native["proprio"]
        context = {"contacts": self.contacts.sample(reset)} if self.contacts else {}
        return TensorDict(
            {
                "policy": frame.clone(),
                "history": self.history.push(frame, reset).flatten(1),
                "critic_state": native["policy"].clone(),
                "terrain": native["terrain"].clone(),
                "dynamics": self.dynamics.clone(),
                **context,
            },
            batch_size=[self.env.num_envs],
        )

    def reset(self, *, seed=None, command=None):
        import torch

        native, first = self.host.reset(seed=seed, command=command)
        if not torch.equal(read_dynamics(self.env), self.dynamics):
            raise ValueError("Persistent dynamics changed on reset")
        return self.observations(native, first), first

    def step(self, raw, phase, *, next_command=None):
        from tensordict import TensorDict

        native, reward, done, info = self.host.step(
            raw, phase, next_command=next_command
        )
        final = info["final_observation"]
        if final is not None:
            # Critic-only bootstrap: no history push or second contact read.
            final = TensorDict(
                {"critic_state": final["policy"], "terrain": final["terrain"]},
                batch_size=[len(info["final_env_ids"])],
            )
        return (
            self.observations(native, done),
            reward,
            done,
            {**info, "final_observation": final},
        )

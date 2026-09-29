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
    """One native observation delivery, one verified motor delivery per control tick."""

    def __init__(self, env, app, *, contact_conditioned=False):
        import torch
        from parkour_lab.methods.roa.model import CausalHistory
        from parkour_lab.control.motor_contract import make_motor_contract
        from parkour_lab.runtime.motor import (
            NativeJointTargetBridge,
            _runtime_motor_binding,
            NATIVE_RAW_ACTION_MEANING,
        )

        self.env, self.app = env, app
        self.history = CausalHistory()
        binding, digest = _runtime_motor_binding(env)
        self.manifest = {
            "joint_names": binding["joint_names"],
            "period_s": 0.02,
            "configuration": {"default_position_rad": binding["default_position_rad"]},
            "actuator_profile": "native_motor_sha256:" + digest,
            "raw_action_meaning": NATIVE_RAW_ACTION_MEANING,
        }
        self.motor_contract = make_motor_contract(
            binding, self.manifest["actuator_profile"]
        )
        self.bridge = NativeJointTargetBridge(
            env, self.motor_contract, self.manifest, preserve_native_raw=True
        )
        self.dynamics = read_dynamics(env)
        self.contacts = None
        if contact_conditioned:
            from parkour_lab.methods.roa.contacts import ContactFeatures

            self.contacts = ContactFeatures(env)
        self.previous = torch.zeros((env.num_envs, 12), device=env.device)
        self.resets = self.partial_reset_steps = self.steps = 0
        self.stage_counts = {}
        if env.step_dt != 0.02 or env.physics_dt != 0.005 or env.cfg.decimation != 4:
            raise ValueError("ROA requires native 50Hz control / 200Hz physics")

    def observations(self, native, reset, command=None):
        import torch
        from tensordict import TensorDict

        frame, clean = native["proprio"], native["policy"]
        if command is not None:
            # An explicit command owns just the command component. Keep the one
            # native noisy sensor draw and push history exactly once per tick.
            term = self.env.command_manager.get_term("base_velocity")
            desired = frame.new_tensor(command)
            if (
                desired.shape != (3,)
                or not torch.isfinite(desired).all()
                or not torch.equal(frame[:, 6:9], term.vel_command_b)
                or not torch.equal(clean[:, 9:12], term.vel_command_b)
            ):
                raise ValueError(
                    "Invalid command or native command observation binding"
                )
            desired = desired.expand(self.env.num_envs, 3)
            term.time_left.fill_(float("inf"))
            term.is_standing_env.fill_(False)
            term.is_heading_env.fill_(False)
            term.vel_command_b.copy_(desired)
            frame, clean = frame.clone(), clean.clone()
            frame[:, 6:9], clean[:, 9:12] = desired, desired
        expected = self.previous.clone()
        expected[reset] = 0
        command = self.env.command_manager.get_command("base_velocity")
        if (
            frame.shape != (self.env.num_envs, 45)
            or clean.shape != (self.env.num_envs, 48)
            or not torch.equal(frame[:, -12:], expected)
            or not torch.equal(frame[:, 6:9], command)
            or not torch.equal(
                clean[:, :3], self.env.scene["robot"].data.root_lin_vel_b
            )
        ):
            raise ValueError(
                "Pre-action sensor, command, COM-velocity or previous-action alignment failed"
            )
        # Teacher and student share noisy proprioception; only the declared
        # teacher extension sees contacts. The scan remains critic-only.
        context = {"contacts": self.contacts.sample(reset)} if self.contacts else {}
        return TensorDict(
            {
                "policy": frame.clone(),
                "history": self.history.push(frame, reset).flatten(1),
                "critic_state": clean.clone(),
                "terrain": native["terrain"].clone(),
                "dynamics": self.dynamics.clone(),
                **context,
            },
            batch_size=[self.env.num_envs],
        )

    def reset(self, *, seed=None, command=None):
        import torch

        native, _ = self.env.reset(**({"seed": seed} if seed is not None else {}))
        if not torch.equal(read_dynamics(self.env), self.dynamics):
            raise ValueError("Persistent dynamics changed on reset")
        self.previous.zero_()
        reset = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return self.observations(native, reset, command), reset

    def step(self, raw, stage, *, next_command=None):
        import torch
        from tensordict import TensorDict
        from parkour_lab.control.controller import JointTargets

        if not self.app.is_running():
            raise RuntimeError("Simulation application stopped during training")
        raw = raw.detach().clone()
        targets = self.bridge.default + 0.25 * raw
        delivered = self.bridge.encode(
            JointTargets(self.bridge.joint_names, targets, raw)
        )
        with torch.no_grad():
            native, reward, terminated, timed_out, extras = self.env.step(delivered)
        self.bridge.verify_delivery(terminated, timed_out)
        done = terminated | timed_out
        ids, final = extras["final_env_ids"], extras["final_observation"]
        if not torch.equal(ids, done.nonzero(as_tuple=False).flatten()):
            raise ValueError("Final observation rows do not match episode endings")
        if len(ids):
            if final is None or not torch.equal(
                final["proprio"][:, -12:], delivered[ids]
            ):
                raise ValueError(
                    "Final observations must precede the action-buffer reset"
                )
            # Timeout values need only the asymmetric critic's inputs. Do not
            # push history or read teacher contacts again for an ended episode.
            final = TensorDict(
                {"critic_state": final["policy"], "terrain": final["terrain"]},
                batch_size=[len(ids)],
            )
        self.previous = delivered.clone()
        self.steps += 1
        self.resets += int(done.sum())
        self.partial_reset_steps += int(done.any() and not done.all())
        counts = self.stage_counts.setdefault(
            stage, {"control_steps": 0, "terminated_rows": 0, "timeout_rows": 0}
        )
        counts["control_steps"] += 1
        counts["terminated_rows"] += int(terminated.sum())
        counts["timeout_rows"] += int(timed_out.sum())
        return (
            self.observations(native, done, next_command),
            reward,
            done,
            {
                **extras,
                "terminated": terminated,
                "truncated": timed_out,
                "time_outs": timed_out & ~terminated,
                "final_observation": final,
            },
        )

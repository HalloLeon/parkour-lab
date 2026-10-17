"""CPU doubles for the shared native interface, independent of any learner."""

import copy
from types import SimpleNamespace as NS

import torch

from parkour_lab.control.motor_contract import PARAMETERS
from parkour_lab.runtime.native import foot_contacts, motion_state


class Scene(dict):
    """Small mapping with the native environment-origin attribute."""

    def __init__(self, count, **assets):
        super().__init__(assets)
        self.env_origins = torch.zeros(count, 3)


def motor_binding(names, default, count=4):
    return {
        "joint_names": list(names),
        "default_position_rad": list(default),
        "action": {"class_type": "JointPositionAction", "scale": 0.25, "clip": None},
        "step_dt_s": 0.02,
        "physics_dt_s": 0.005,
        "decimation": 4,
        "actuators": {
            "legs": {
                "joint_names": list(names),
                "configuration": {"class_type": "DCMotor"},
                "resolved_parameters": {
                    name: [[1.0] * 12 for _ in range(count)] for name in PARAMETERS
                },
            }
        },
    }


def rebatch(binding, count):
    result = copy.deepcopy(binding)
    for actuator in result["actuators"].values():
        for name, rows in actuator["resolved_parameters"].items():
            actuator["resolved_parameters"][name] = [
                copy.deepcopy(rows[0]) for _ in range(count)
            ]
    return result


def make_native_env(count=8):
    """Resolved motor and fresh sensors; step emulates delivery, not physics."""

    def zeros(width):
        return torch.zeros(count, width)

    names = [f"joint{i}" for i in range(12)]
    default = torch.linspace(-0.3, 0.3, 12).repeat(count, 1)
    data = NS(
        default_joint_pos=default,
        default_joint_vel=zeros(12),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(count, 12, 1),
        joint_pos=default + 0.1,
        joint_vel=zeros(12),
        joint_pos_target=zeros(12),
        root_lin_vel_b=zeros(3),
        root_ang_vel_b=zeros(3),
        root_ang_vel_w=zeros(3),
        root_pos_w=zeros(3),
        body_pos_w=torch.zeros(count, 5, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
    )
    action_cfg = NS(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=False,
        use_default_offset=True,
        scale=0.25,
        clip=None,
        class_type=type(
            "JointPositionAction",
            (),
            {
                "__module__": "isaaclab.envs.mdp.actions.joint_actions",
            },
        ),
        to_dict=lambda: {
            "class_type": "JointPositionAction",
            "scale": 0.25,
            "clip": None,
        },
    )
    term = NS(
        cfg=action_cfg,
        processed_actions=zeros(12),
        IO_descriptor=NS(
            joint_names=names, scale=0.25, clip=None, offset=default[0].tolist()
        ),
    )
    motor = NS(joint_names=names, cfg=NS(to_dict=lambda: {"class_type": "DCMotor"}))
    for parameter, value in zip(
        PARAMETERS, (25.0, 0.5, 23.5, 30.0, 1e9, 1000.0, 0.0, 0.0), strict=True
    ):
        setattr(motor, parameter, torch.full((count, 12), value))
    command = NS(
        cfg=NS(heading_command=False, rel_heading_envs=0.0, rel_standing_envs=0.0),
        vel_command_b=zeros(3),
        time_left=torch.ones(count),
        is_standing_env=torch.zeros(count, dtype=torch.bool),
        is_heading_env=torch.zeros(count, dtype=torch.bool),
    )
    env = NS(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        physics_dt=0.005,
        index=0,
        cfg=NS(decimation=4, observations=NS(policy=NS(enable_corruption=False))),
        scene=Scene(
            count, robot=NS(data=data, joint_names=names, actuators={"legs": motor})
        ),
        action_manager=NS(
            action=zeros(12),
            active_terms=["joint_pos"],
            total_action_dim=12,
            get_term=lambda name: term,
        ),
        command_manager=NS(
            get_term=lambda name: command,
            get_command=lambda name: command.vel_command_b,
        ),
        observation_manager=NS(
            compute=lambda: {}, compute_group=lambda *args, **kwargs: None
        ),
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        max_episode_length=1000,
        capture_motion=False,
        reward_manager=NS(active_terms=["tracking", "effort"], _step_reward=zeros(2)),
    )

    def step(action):
        env.action_manager.action.copy_(action)
        term.processed_actions.copy_(0.25 * action + default)
        data.joint_pos_target.copy_(term.processed_actions)
        env.episode_length_buf += 1
        terminated = torch.zeros(count, dtype=torch.bool)
        timed_out = env.episode_length_buf >= env.max_episode_length
        env.action_manager.action[timed_out] = 0
        env.episode_length_buf[timed_out] = 0
        env.index += 1
        return (
            {},
            torch.zeros(count),
            terminated,
            timed_out,
            {
                "motion_state": motion_state(env) if env.capture_motion else None,
            },
        )

    env.step = step
    env.scene["contact_forces"] = NS(
        body_names=["base", "RL_foot", "FR_foot", "RR_foot", "FL_foot"],
        data=NS(net_forces_w=torch.zeros(count, 5, 3)),
    )
    return env


class NativeEnvironmentFixture:
    """Resolved stock motor plus deterministic, auto-resetting native API doubles."""

    def __init__(self, count=8):
        self.env = make_native_env(count)
        self.robot = self.env.scene["robot"]
        self.data = self.robot.data
        self.term = self.env.action_manager.get_term("joint_pos")
        self.command_term = self.env.command_manager.get_term("base_velocity")
        self.command = self.command_term.vel_command_b
        self.command_term.time_left = torch.ones(count)
        self.command_term.is_standing_env = torch.ones(count, dtype=torch.bool)
        self.command_term.is_heading_env = torch.ones(count, dtype=torch.bool)
        self.nominal = torch.full((count, 1), 10.0)
        self.mass = self.nominal * torch.linspace(0.8, 1.2, count)[:, None]
        self.com = torch.zeros(count, 1, 7)
        self.com[:, 0, 0] = torch.linspace(-0.01, 0.01, count)
        self.com[:, 0, 6] = 1.0
        self.material = torch.tensor([0.7, 0.6, 0.01]).repeat(count, 2, 1)
        self.data.default_mass = self.nominal.clone()
        self.robot.body_names = ["base"]
        self.robot.root_physx_view = NS(
            get_masses=lambda: self.mass,
            get_coms=lambda: self.com,
            get_material_properties=lambda: self.material,
        )
        self.app = NS(is_running=lambda: self.running)
        self.running = True
        self.resets = 0
        self.reset_seeds = []
        self.records = []
        self.deliveries = []
        self.on_reset = lambda: None
        self.on_step = lambda: None
        self.next_terminated = self.next_timed_out = None
        self.rows = torch.linspace(-1, 1, count)
        self.env.reset = self.reset
        self.env.step = self.step

    def observations(self):
        clean_frame = torch.cat(
            (
                self.data.root_ang_vel_b,
                self.data.projected_gravity_b,
                self.command,
                self.data.joint_pos - self.data.default_joint_pos,
                self.data.joint_vel,
                self.env.action_manager.action,
                foot_contacts(self.env),
            ),
            dim=1,
        )
        noisy_frame = clean_frame.clone()
        noisy_frame[:, :3] += 0.001 * (self.env.index + 1)
        terrain = torch.zeros(self.env.num_envs, 264)
        terrain[:, :132] = self.rows[:, None] * 0.02
        terrain[:, 132:] = 1.0
        return {
            "proprio": noisy_frame,
            "policy": torch.cat((self.data.root_lin_vel_b, clean_frame), dim=1),
            "terrain": terrain,
        }

    def reset(self, *, seed=None):
        self.resets += 1
        self.reset_seeds.append(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.env.action_manager.action.zero_()
        self.env.episode_length_buf.zero_()
        self.data.joint_pos.copy_(
            self.data.default_joint_pos + self.rows[:, None] * 0.02
        )
        self.data.joint_vel.zero_()
        self.data.root_lin_vel_b.zero_()
        self.data.root_ang_vel_b[:, 0] = self.rows * 0.03
        self.command[:, 0] = 0.25
        self.command[:, 1] = self.rows * 0.05
        self.command[:, 2] = self.rows * 0.1
        self.on_reset()
        return self.observations(), {}

    def step(self, action):
        self.env.episode_length_buf += 1
        previous = self.env.action_manager.action.clone()
        self.deliveries.append(action.clone())
        self.env.action_manager.action.copy_(action)
        self.term.processed_actions.copy_(self.data.default_joint_pos + 0.25 * action)
        self.data.joint_pos_target.copy_(self.term.processed_actions)
        self.data.joint_vel.copy_(
            (self.term.processed_actions - self.data.joint_pos) * 0.1
        )
        self.data.joint_pos.add_(
            0.05 * (self.term.processed_actions - self.data.joint_pos)
        )
        self.data.root_lin_vel_b.mul_(0.8).add_(
            0.2 * (self.command + 0.02 * action[:, :3]) / (self.mass / self.nominal)
        )
        self.data.root_ang_vel_b[:, 2] = self.command[:, 2] + 0.01 * action[:, 0]
        reward = (
            1.0
            - (self.data.root_lin_vel_b[:, :2] - self.command[:, :2]).square().sum(1)
            - 0.005 * action.square().mean(1)
        )
        rates = self.env.reward_manager._step_reward
        rates[:, 0] = (
            1.0
            - (self.data.root_lin_vel_b[:, :2] - self.command[:, :2]).square().sum(1)
        ) / self.env.step_dt
        rates[:, 1] = -0.005 * action.square().mean(1) / self.env.step_dt
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool)
        timed_out = torch.zeros_like(terminated)
        index = self.env.index
        terminated[0] = index % 17 == 3
        timed_out[1] = index % 19 == 4
        terminated[2] = timed_out[2] = index % 23 == 5
        if self.next_terminated is not None:
            terminated = self.next_terminated.clone()
        if self.next_timed_out is not None:
            timed_out = self.next_timed_out.clone()
        done = terminated | timed_out
        ids = done.nonzero(as_tuple=False).flatten()
        final = (
            {key: value[ids].clone() for key, value in self.observations().items()}
            if len(ids)
            else None
        )
        motion = motion_state(self.env) if self.env.capture_motion else None
        self.env.action_manager.action[done] = 0
        self.env.episode_length_buf[done] = 0
        self.data.joint_pos[done] = self.data.default_joint_pos[done]
        self.data.joint_vel[done] = 0
        self.data.root_lin_vel_b[done] = 0
        self.env.index += 1
        self.on_step()
        native = self.observations()
        self.records.append(
            {
                "previous": previous,
                "action": action.clone(),
                "terminated": terminated.clone(),
                "timed_out": timed_out.clone(),
                "next_proprio": native["proprio"].clone(),
            }
        )
        return (
            native,
            reward,
            terminated,
            timed_out,
            {
                "time_outs": torch.ones_like(timed_out),
                "native_extra": "retained",
                "final_env_ids": ids,
                "final_observation": final,
                "motion_state": motion,
            },
        )

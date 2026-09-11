"""Small per-episode record for the opt-in scene-placement feedback comparison."""

from __future__ import annotations

import json


class SceneFeedbackCapture:
    """Observe normal policy/reset state without warming up or stepping physics."""

    def __init__(self):
        self.episodes = []
        self.pending = None

    @staticmethod
    def _values(tensor):
        return tensor.detach().cpu().tolist()

    def begin(self, env, observations):
        if env.num_envs != 1 or self.pending is not None or len(self.episodes) >= 3:
            raise ValueError(
                "Scene feedback capture requires three single-env episodes"
            )
        asset = env.scene["robot"]
        data = asset.data
        initial = {
            "environment_origin_w_m": self._values(env.scene.env_origins[0]),
            "root_position_env_m": self._values(
                (data.root_pos_w - env.scene.env_origins)[0]
            ),
            "root_orientation_wxyz": self._values(data.root_quat_w[0]),
            "linear_velocity_body_m_s": self._values(data.root_lin_vel_b[0]),
            "angular_velocity_body_rad_s": self._values(data.root_ang_vel_b[0]),
            "joint_names": list(asset.joint_names),
            "joint_position_rad": self._values(data.joint_pos[0]),
            "joint_velocity_rad_s": self._values(data.joint_vel[0]),
        }
        record = {
            "episode_index": len(self.episodes),
            "initial_state": initial,
            "initial_observations": {
                name: self._values(value[0]) for name, value in observations.items()
            },
            "first_policy_action": None,
        }
        # Invalid input is evidence of a broken run, not something to sanitize.
        json.dumps(record, allow_nan=False)
        self.pending = record

    def record_action(self, actions):
        if self.pending is None or self.pending["first_policy_action"] is not None:
            raise ValueError(
                "First policy action requires an unmatched initial snapshot"
            )
        value = self._values(actions[0])
        if len(value) != 12:
            raise ValueError("Expected twelve Go2 actions")
        json.dumps(value, allow_nan=False)
        self.pending["first_policy_action"] = value

    def finish(self, env, steps, outcomes, progress, waypoints):
        if (
            self.pending is None
            or self.pending["first_policy_action"] is None
            or type(steps) is not int
            or steps <= 0
        ):
            raise ValueError(
                "Scene feedback episode lacks its initial state or duration"
            )
        record = dict(self.pending)
        manager = env.termination_manager
        record.update(
            duration_steps=steps,
            duration_s=steps * env.step_dt,
            termination_reasons=[
                name
                for name in manager.active_terms
                if bool(manager.get_term(name)[0].item())
            ],
            outcomes={name: bool(value[0].item()) for name, value in outcomes.items()},
            max_course_progress_m=float(progress[0].item()),
            max_waypoints_reached=int(waypoints[0].item()),
        )
        json.dumps(record, allow_nan=False)
        if not record["termination_reasons"]:
            raise ValueError(
                "Completed scene feedback episode has no termination reason"
            )
        self.episodes.append(record)
        self.pending = None

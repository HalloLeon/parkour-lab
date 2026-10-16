"""Bounded retained-task integration check; requires the configured Isaac Lab interpreter."""

import argparse
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from parkour_lab.provenance import (
    dependency_identity,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.session import exit_native_process, finish_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain",
        choices=("flat", "rough", "procedural", "steps", "traversal", "connected"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A caller selects a new directory; no previous evidence is overwritten.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "terrain": args.terrain,
        "behavior_validated": False,
        "package_sources": package_source_identity(),
        "interpreter": {"executable": sys.executable, "version": sys.version},
    }

    def publish():
        write_json(output / "report.json", report)

    app = env = None
    code = 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        import torch

        from parkour_lab.config import TaskConfig
        from parkour_lab.environments.configuration import build_environment_config
        from parkour_lab.environments.runtime import LocomotionEnv

        torch.set_num_threads(2)
        task = TaskConfig(
            terrain=args.terrain,
            device=args.device,
            num_envs=1 if args.terrain == "connected" else 4,
            seed=1077,
            world=(
                dict(
                    target_family="stairs", tier=0.04, coarse_seed=1077, fine_seed=1078
                )
                if args.terrain == "connected"
                else None
            ),
        )
        write_json(output / "task.json", asdict(task))
        for module, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
            ("yaml", "PyYAML"),
            ("trimesh", "trimesh"),
        ):
            report.setdefault("dependencies", {})[module] = dependency_identity(
                module, distribution
            )
        import isaaclab

        # Preserve the SDK's actual diff, not just its version or a claimed tag.
        if "commit" in report["dependencies"]["isaaclab"]:
            try:
                write_run_provenance(output / "sdk", isaaclab.__file__)
            except RuntimeError as error:
                report["sdk_git_unavailable"] = str(error)
        else:
            report["sdk_git_unavailable"] = (
                "Imported SDK source is not tracked by a Git checkout"
            )
        cfg = build_environment_config(task, evaluation=True)
        cfg.validate()
        import yaml

        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = LocomotionEnv(cfg=cfg)

        def check_observations(observations):
            assert observations["proprio"].shape == (env.num_envs, 49)
            assert all(torch.isfinite(value).all() for value in observations.values())

        observations, _ = env.reset(seed=task.seed)
        check_observations(observations)
        action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        saved_final = saved_copy = None
        with torch.inference_mode():
            for step in range(32):
                assert app.is_running(), "Application stopped before smoke completed"
                if step == 2:
                    # Exercise native same-step autoreset without waiting a full horizon.
                    env.episode_length_buf[0] = env.max_episode_length - 1
                observations, reward, terminated, truncated, extras = env.step(action)
                check_observations(observations)
                assert torch.isfinite(reward).all()
                ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                assert torch.equal(ids, extras["final_env_ids"])
                final = extras["final_observation"]
                if len(ids):
                    assert final is not None
                    assert all(value.shape[0] == len(ids) for value in final.values())
                    assert all(torch.isfinite(value).all() for value in final.values())
                else:
                    assert final is None
                if saved_final is not None:
                    assert all(
                        torch.equal(saved_final[name], saved_copy[name])
                        for name in saved_final
                    )
                if step == 2:
                    assert truncated[0] and not truncated[1:].any()
                    assert env.episode_length_buf[0] == 0
                    saved_final = final
                    saved_copy = {name: value.clone() for name, value in final.items()}
                    report["timeout_final_rows"] = ids.tolist()
                if step == 16:
                    survivors = env.episode_length_buf[1:].clone()
                    observations, _ = env.reset(
                        env_ids=torch.tensor([0], device=env.device)
                    )
                    check_observations(observations)
                    assert env.episode_length_buf[0] == 0
                    assert torch.equal(env.episode_length_buf[1:], survivors)
                    report["explicit_reset_rows"] = [0]
                    report["explicit_reset_scope"] = (
                        "partial" if env.num_envs > 1 else "full_single_replica"
                    )
        report.update(
            status="SMOKE_PASSED",
            control_steps=32,
            num_envs=env.num_envs,
            transition_counts=dict(env.transition_counts),
        )
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        code = finish_session(env, app, report, publish, code)
        if app is not None:
            # Keep native locals alive until process exit after Kit unloads.
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Sequential native cases own fresh stages and cannot replace a live context."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from parkour_lab.evaluation import environment
from parkour_lab.evaluation.cases import EvaluationCase
from tests.support.isaaclab import modules


class CaseEnvironmentTests(unittest.TestCase):
    def test_sequential_stages_do_not_reuse_same_path_terrain_cache(self):
        events = []
        raycaster = NS(meshes={"/World/ground": "old terrain"})

        def new_stage():
            events.append("new_stage")

        sim = NS(
            SimulationContext=NS(instance=lambda: None),
            create_new_stage=new_stage,
        )
        with (
            modules(
                {"isaaclab": NS(sim=sim), "isaaclab.sensors": NS(RayCaster=raycaster)}
            ),
            patch.object(
                environment.gc, "collect", side_effect=lambda: events.append("release")
            ),
        ):
            environment._prepare_case_stage()
            self.assertEqual(raycaster.meshes, {})
            raycaster.meshes["/World/ground"] = "first case mesh"
            environment._prepare_case_stage()
            self.assertEqual(raycaster.meshes, {})
        self.assertEqual(events, ["release", "new_stage", "release", "new_stage"])

    def test_live_context_rejects_without_replacing_stage_or_clearing_meshes(self):
        create_stage = Mock()
        mesh = object()
        raycaster = NS(meshes={"/World/ground": mesh})
        sim = NS(
            SimulationContext=NS(instance=lambda: object()),
            create_new_stage=create_stage,
        )
        with (
            modules(
                {"isaaclab": NS(sim=sim), "isaaclab.sensors": NS(RayCaster=raycaster)}
            ),
            self.assertRaisesRegex(RuntimeError, "active environment"),
        ):
            environment._prepare_case_stage()
        create_stage.assert_not_called()
        self.assertIs(raycaster.meshes["/World/ground"], mesh)

    def test_stage_precedes_construction_and_failed_case_binding_closes_task(self):
        events = []
        case = EvaluationCase("validation", "Flat", None, "plane", "stand")
        task = NS(
            training=True,
            _env=NS(cfg=NS(curriculum=NS(terrain_levels=None))),
            close=Mock(side_effect=lambda: events.append("close")),
        )
        with (
            patch.object(
                environment,
                "_prepare_case_stage",
                side_effect=lambda: events.append("stage"),
            ),
            patch.object(
                environment,
                "build_config",
                side_effect=lambda *args, **kwargs: events.append("config"),
            ),
            patch.object(
                environment,
                "make",
                side_effect=lambda cfg: (events.append("make"), task)[1],
            ),
            self.assertRaisesRegex(ValueError, "disable training"),
        ):
            environment.make_case_task(case, "cpu")
        self.assertEqual(events, ["stage", "config", "make", "close"])
        task.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

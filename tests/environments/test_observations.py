"""Terrain scan configuration, sensor adapter and pure tensor preprocessing."""

from __future__ import annotations

import importlib.util
import math
import unittest
from dataclasses import dataclass
from types import SimpleNamespace as NS

import torch
from parkour_lab.environments.terrain import _terrain_height_components

from tests.support import SOURCE_ROOT
from tests.support.isaaclab import modules


def observation_module():
    """Load the whole production module; double only native types and configclass."""

    class RayCaster:
        pass

    name = "parkour_lab.environments._observations_test"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE_ROOT / "parkour_lab/environments/observations.py"
    )
    module = importlib.util.module_from_spec(spec)
    with modules(
        {
            name: module,
            "isaaclab.assets": NS(Articulation=object),
            "isaaclab.envs": NS(ManagerBasedRLEnv=object),
            "isaaclab.managers": NS(SceneEntityCfg=lambda name: NS(name=name)),
            "isaaclab.sensors": NS(RayCaster=RayCaster),
            "isaaclab.utils": NS(configclass=dataclass),
        }
    ):
        spec.loader.exec_module(module)
    return module


class HeightScanConfigTests(unittest.TestCase):
    def test_rejects_non_finite_or_ambiguous_geometry(self) -> None:
        config = observation_module().HeightScanObservationCfg
        config(num_rays=132, vertical_offset=0.3, clip=0.5)
        invalid = (
            ({"num_rays": True}, "positive integer"),
            ({"num_rays": 1.5}, "positive integer"),
            ({"num_rays": 0}, "positive integer"),
            ({"vertical_offset": math.nan}, "vertical_offset must be finite"),
            ({"clip": math.inf}, "clip must be finite and positive"),
            ({"clip": 0.0}, "clip must be finite and positive"),
        )
        for overrides, message in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                config(**overrides)


class TerrainHeightScanTests(unittest.TestCase):
    def test_sensor_adapter_concatenates_heights_then_validity(self):
        observation = observation_module()
        sensor = observation.RayCaster()
        sensor.data = NS(
            ray_hits_w=torch.tensor([[[0.0, 0.0, 0.7], [0.0, 0.0, math.inf]]])
        )
        root = torch.tensor([[0.0, 0.0, 1.0]])
        env = NS(scene={"scanner": sensor, "body": NS(data=NS(root_pos_w=root))})
        config = observation.HeightScanObservationCfg(num_rays=2, clip=0.5)
        result = observation.terrain_height_scan(
            env, config, NS(name="scanner"), NS(name="body")
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        self.assertEqual(result.shape, (1, 4))

    def test_missing_or_wrong_sensor_is_not_silently_replaced(self):
        observation = observation_module()
        env = NS(scene={"robot": NS()})
        with self.assertRaises(KeyError):
            observation.terrain_height_scan(env)
        env.scene["height_scanner"] = object()
        with self.assertRaisesRegex(TypeError, "RayCaster"):
            observation.terrain_height_scan(env)


class TerrainHeightComponentsTests(unittest.TestCase):
    def test_fixed_metric_normalization_and_validity(self) -> None:
        root_z = torch.tensor([1.0])
        ray_hits_w = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.7],
                    [0.0, 0.0, 0.9],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.0, 1.5],
                    [0.0, 0.0, float("inf")],
                    [float("nan"), 0.0, 0.7],
                ]
            ]
        )

        heights, validity = _terrain_height_components(
            root_z,
            ray_hits_w,
            num_rays=6,
            vertical_offset=0.3,
            clip=0.5,
        )

        torch.testing.assert_close(
            heights,
            torch.tensor([[0.0, -0.4, 1.0, -1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(
            validity, torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        )
        self.assertTrue(torch.isfinite(heights).all())
        self.assertTrue(torch.isfinite(validity).all())

    def test_non_finite_root_invalidates_every_ray(self) -> None:
        heights, validity = _terrain_height_components(
            torch.tensor([float("nan")]),
            torch.zeros(1, 2, 3),
            num_rays=2,
            vertical_offset=0.3,
            clip=0.5,
        )
        torch.testing.assert_close(heights, torch.ones(1, 2))
        torch.testing.assert_close(validity, torch.zeros(1, 2))

    def test_rejects_mismatched_shapes(self) -> None:
        root_z = torch.zeros(1)
        ray_hits_w = torch.zeros(1, 2, 3)

        with self.assertRaisesRegex(ValueError, "Expected root_z"):
            _terrain_height_components(
                root_z[:, None], ray_hits_w, num_rays=2, vertical_offset=0.3, clip=0.5
            )
        with self.assertRaisesRegex(ValueError, "Expected root_z"):
            _terrain_height_components(
                root_z, ray_hits_w[..., 0], num_rays=2, vertical_offset=0.3, clip=0.5
            )
        with self.assertRaisesRegex(ValueError, r"\[E, 3, 3\]"):
            _terrain_height_components(
                root_z, ray_hits_w, num_rays=3, vertical_offset=0.3, clip=0.5
            )


if __name__ == "__main__":
    unittest.main()

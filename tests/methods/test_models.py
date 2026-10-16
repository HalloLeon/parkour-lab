"""Neural component arithmetic, initialization and state round trips."""

import copy
import unittest

import torch
from torch import nn

from parkour_lab.methods.models import (
    PrivilegedScanEncoder,
    TerrainConditionedInput,
)


class ModelComponentsTests(unittest.TestCase):
    def setUp(self):
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)

    def assert_state_equal(self, actual, expected):
        self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, expected.state_dict()[name]), name)

    def test_scan_initialization_and_rng_match_explicit_layers(self):
        torch.manual_seed(67)
        expected = nn.Module()
        expected.network = nn.Sequential(
            nn.Linear(264, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
        )
        rng = torch.random.get_rng_state()
        torch.manual_seed(67)
        actual = PrivilegedScanEncoder(264)
        self.assert_state_equal(actual, expected)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        inputs = torch.randn(7, 264)
        self.assertTrue(torch.equal(actual(inputs), expected.network(inputs)))

    def test_terrain_conditioning_initial_arithmetic_and_round_trip(self):
        reference = nn.Linear(52, 128).double()
        model = TerrainConditionedInput(reference)
        inputs = torch.randn(7, 316, dtype=torch.float64)
        self.assertTrue(
            torch.equal(model(inputs), reference(inputs[:, :52].contiguous()))
        )
        with torch.no_grad():
            model.projection.weight.normal_()
        clone = TerrainConditionedInput(copy.deepcopy(reference))
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assert_state_equal(clone, model)
        self.assertTrue(torch.equal(clone(inputs), model(inputs)))
        expected = reference(inputs[:, :52].contiguous()) + model.projection(
            model.encoder(inputs[:, 52:])
        )
        self.assertTrue(torch.equal(model(inputs), expected))
        with self.assertRaises(ValueError):
            model(inputs[:, :-1])


if __name__ == "__main__":
    unittest.main()

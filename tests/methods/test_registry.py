"""Named built-ins and real installed entry-point loading without import strings."""

import json
import unittest

from parkour_lab.config import MethodConfig
from parkour_lab.methods import get_backend, roa

from tests.support import installed_backends


class RegistryTests(unittest.TestCase):
    def test_named_backends_and_resolved_options(self):
        self.assertIs(get_backend("roa"), roa)
        config = MethodConfig()
        self.assertEqual(
            config, MethodConfig("roa", json.loads(json.dumps(config.options)))
        )
        with self.assertRaises(ValueError):
            MethodConfig("roa", {"typo": 1})
        self.assertEqual(config.options["learning_rate"], 0.0002)
        self.assertEqual(config.options["regularization_start_update"], 3000)
        self.assertEqual(config.options["regularization_end_update"], 10000)
        for options in (
            {"regularization_start_update": -1},
            {"regularization_end_update": False},
            {"regularization_start_update": 5, "regularization_end_update": 4},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MethodConfig("roa", options)
        for name in ("missing_method", "os.system", "package:factory", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_backend(name)

    def test_installed_external_entry_point_and_conflicts(self):
        # An ROA alias verifies the loader contract, not another learner.
        with installed_backends(
            {
                "external_roa": "parkour_lab.methods.roa",
                "roa": "parkour_lab.methods.roa",
            }
        ):
            self.assertIs(get_backend("external_roa"), roa)
            self.assertEqual(MethodConfig("external_roa").options, roa.configure({}))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                get_backend("roa")
            with installed_backends({"external_roa": "parkour_lab.methods.roa"}):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    get_backend("external_roa")


if __name__ == "__main__":
    unittest.main()

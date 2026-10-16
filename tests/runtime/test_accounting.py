"""Real collection is charged once, independently of learner updates or diagnostics."""

import unittest

from parkour_lab.runtime.accounting import InteractionCounter


class InteractionCounterTests(unittest.TestCase):
    def test_vector_steps_count_all_replicas_once_at_both_physics_rates(self):
        for decimation in (4, 8):
            with self.subTest(decimation=decimation):
                counter = InteractionCounter(160)
                counter.sync(1, decimation)
                counter.sync(1, decimation, failed=True)
                self.assertEqual(
                    counter.state_dict(), dict(session=160, cumulative=160, uncertain=0)
                )
                counter.sync(2, 2 * decimation)
                self.assertEqual(counter.cumulative, 320)

    def test_failed_partial_physics_is_conservatively_charged_without_double_count(
        self,
    ):
        counter = InteractionCounter(160)
        counter.sync(0, 2, failed=True)
        counter.sync(0, 2, failed=True)
        self.assertEqual(
            counter.state_dict(), dict(session=160, cumulative=160, uncertain=160)
        )
        counter.sync(1, 6)
        self.assertEqual(
            counter.state_dict(), dict(session=320, cumulative=320, uncertain=160)
        )

    def test_restore_preserves_spend_and_rejects_collected_warmup(self):
        counter = InteractionCounter(4)
        counter.restore(60, uncertain=4)
        self.assertEqual(
            counter.state_dict(), dict(session=0, cumulative=60, uncertain=4)
        )
        counter.sync(1, 4)
        self.assertEqual(counter.cumulative, 64)
        with self.assertRaisesRegex(ValueError, "before collection"):
            counter.restore(100)
        self.assertEqual(counter.cumulative, 64)

    def test_malformed_or_decreasing_native_and_restore_values_are_atomic(self):
        counter = InteractionCounter(4)
        counter.restore(12, uncertain=4)
        before = counter.state_dict()
        for total, unknown in ((-1, 0), (True, 0), (11, 4), (12, 3), (12, 13)):
            with (
                self.subTest(total=total, unknown=unknown),
                self.assertRaises(ValueError),
            ):
                counter.restore(total, uncertain=unknown)
            self.assertEqual(counter.state_dict(), before)
        counter.sync(1, 4)
        before = counter.state_dict()
        for steps, ticks in ((0, 4), (1, 3), (True, 4), (1, -1)):
            with self.subTest(steps=steps, ticks=ticks), self.assertRaises(ValueError):
                counter.sync(steps, ticks)
            self.assertEqual(counter.state_dict(), before)


if __name__ == "__main__":
    unittest.main()

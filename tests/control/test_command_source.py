"""Pure command ingress tests; no Torch, simulator or physical stopping claim."""

import math
import unittest
from dataclasses import FrozenInstanceError

from parkour_lab.control.command_source import BodyTwistLease, SequenceControl


class BodyTwistLeaseTests(unittest.TestCase):
    def live(self):
        lease = BodyTwistLease()
        generation = lease.arm(1.0)
        self.assertTrue(
            lease.receive(
                (0.55, -0.2, 0.5), time_s=1.0, sequence=0, generation=generation
            )
        )
        return lease, generation

    def assert_zero(self, decision, status):
        self.assertEqual(decision.command, (0.0, 0.0, 0.0))
        self.assertEqual(decision.status, status)

    def test_initial_arm_waiting_and_explicit_epoch_changes(self):
        lease = BodyTwistLease()
        initial = lease.resolve(0.0)
        self.assert_zero(initial, "disarmed")
        self.assertEqual(initial.generation, 0)
        self.assertIsNone(initial.source_time_s)
        self.assertIsNone(initial.sequence)
        first = lease.arm(0.0)
        self.assertEqual(first, 1)
        self.assert_zero(lease.resolve(100.0), "armed_waiting")
        self.assertTrue(
            lease.receive((0, 0, -0.5), time_s=100.0, sequence=42, generation=first)
        )
        second = lease.arm(100.01)
        self.assertEqual(second, 2)
        waiting = lease.resolve(100.01)
        self.assert_zero(waiting, "armed_waiting")
        self.assertIsNone(waiting.source_time_s)
        self.assertIsNone(waiting.sequence)
        self.assertFalse(
            lease.receive((1, 2, 3), time_s=100.02, sequence=43, generation=first)
        )
        self.assertTrue(
            lease.receive((1, 2, 3), time_s=100.02, sequence=0, generation=second)
        )

    def test_boundary_expiry_preserves_source_evidence_and_decisions_do_not_renew(self):
        lease, generation = self.live()
        for now in (1.0, 1.05, 1.1, 1.2, 1.25):
            decision = lease.resolve(now)
            self.assertEqual(decision.status, "active")
            self.assertEqual(decision.command, (0.55, -0.2, 0.5))
            self.assertEqual(decision.source_time_s, 1.0)
            self.assertEqual(decision.decision_time_s, now)
        decision = lease.resolve(math.nextafter(1.25, math.inf))
        self.assert_zero(decision, "expired")
        self.assertEqual(
            (decision.source_time_s, decision.sequence, decision.generation),
            (1.0, 0, generation),
        )
        self.assertFalse(
            lease.receive((1, 0, 0), time_s=1.3, sequence=1, generation=generation)
        )
        self.assert_zero(lease.resolve(1.3), "expired")

    def test_new_packet_renews_receipt_but_replays_and_foreign_epochs_do_not(self):
        lease, generation = self.live()
        self.assertFalse(
            lease.receive(None, time_s=1.1, sequence=0, generation=generation)
        )
        self.assertFalse(
            lease.receive(None, time_s=1.15, sequence=None, generation=generation + 1)
        )
        self.assertEqual(lease.resolve(1.2).command, (0.55, -0.2, 0.5))
        self.assertEqual(lease.resolve(1.2).source_time_s, 1.0)
        self.assertTrue(
            lease.receive(
                (0.1, 0.2, 0.3), time_s=1.2, sequence=1, generation=generation
            )
        )
        self.assertEqual(lease.resolve(1.45).status, "active")
        self.assertFalse(
            lease.receive((1, 0, 0), time_s=1.451, sequence=1, generation=generation)
        )
        self.assert_zero(lease.resolve(1.451), "expired")

    def test_expiry_checked_before_any_late_packet_can_resurrect(self):
        for sequence, epoch_delta in ((1, 0), (0, 0), (99, 1)):
            with self.subTest(sequence=sequence, epoch_delta=epoch_delta):
                lease, generation = self.live()
                self.assertFalse(
                    lease.receive(
                        (1, 0, 0),
                        time_s=1.26,
                        sequence=sequence,
                        generation=generation + epoch_delta,
                    )
                )
                self.assert_zero(lease.resolve(1.26), "expired")

    def test_release_disconnect_latch_until_explicit_arm_and_fresh_packet(self):
        for method, status in (("release", "released"), ("disconnect", "disconnected")):
            with self.subTest(method=method):
                lease, generation = self.live()
                getattr(lease, method)(1.1)
                decision = lease.resolve(1.1)
                self.assert_zero(decision, status)
                self.assertEqual((decision.source_time_s, decision.sequence), (1.0, 0))
                self.assertFalse(
                    lease.receive(
                        (1, 0, 0), time_s=1.12, sequence=999, generation=generation
                    )
                )
                self.assert_zero(lease.resolve(1.12), status)
                next_generation = lease.arm(1.14)
                self.assert_zero(lease.resolve(1.14), "armed_waiting")
                self.assertFalse(
                    lease.receive(
                        (1, 0, 0), time_s=1.16, sequence=1000, generation=generation
                    )
                )
                self.assertTrue(
                    lease.receive(
                        (0, 0, 0), time_s=1.16, sequence=0, generation=next_generation
                    )
                )
                self.assert_zero(lease.resolve(1.16), "active")

    def test_immutable_copied_command_and_no_clamping_or_privileged_interlock(self):
        lease = BodyTwistLease()
        generation = lease.arm(0.0)
        command = [100.0, -100.0, 999.0]
        self.assertTrue(
            lease.receive(command, time_s=0.0, sequence=0, generation=generation)
        )
        decision = lease.resolve(0.0)
        command[:] = [0, 0, 0]
        self.assertEqual(decision.command, (100.0, -100.0, 999.0))
        with self.assertRaises(FrozenInstanceError):
            decision.command = (0, 0, 0)
        with self.assertRaises(TypeError):
            decision.command[0] = 0
        lease.release(0.02)
        self.assertEqual(decision.status, "active")

    def test_malformed_fresh_content_fails_zero_without_claiming_acceptance(self):
        bad_commands = (
            None,
            {1, 2, 3},
            {1: 0, 2: 0, 3: 0},
            (),
            (1, 2),
            (1, 2, 3, 4),
            "123",
            b"123",
            (1, None, 3),
            (1, True, 3),
            (1, "2", 3),
            (1, math.nan, 3),
            (math.inf, 2, 3),
            (1, 2, -math.inf),
            ((1,), 2, 3),
            (10**1000, 0, 0),
        )
        for command in bad_commands:
            with self.subTest(command=command):
                lease, generation = self.live()
                self.assertFalse(
                    lease.receive(
                        command, time_s=1.1, sequence=1, generation=generation
                    )
                )
                decision = lease.resolve(1.1)
                self.assert_zero(decision, "invalid_packet")
                self.assertEqual((decision.source_time_s, decision.sequence), (1.0, 0))
                self.assertFalse(
                    lease.receive(
                        (1, 0, 0), time_s=1.12, sequence=2, generation=generation
                    )
                )

    def test_strict_metadata_types_fail_closed(self):
        for field, values in (
            ("generation", (True, 1.0, None, 0, -1)),
            ("sequence", (True, 0.0, None, -1)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    lease, generation = self.live()
                    packet = dict(time_s=1.1, sequence=1, generation=generation)
                    packet[field] = value
                    with self.assertRaises(ValueError):
                        lease.receive((1, 0, 0), **packet)
                    self.assert_zero(lease.resolve(1.1), "invalid_packet")

    def test_invalid_or_backwards_clock_latches_every_entry_point(self):
        for method in ("arm", "receive", "release", "disconnect", "resolve"):
            for now in (
                -1.0,
                0.99,
                math.nan,
                math.inf,
                -math.inf,
                True,
                "1.1",
                10**1000,
            ):
                with self.subTest(method=method, now=now):
                    lease, generation = self.live()
                    with self.assertRaises(ValueError):
                        if method == "receive":
                            lease.receive(
                                (1, 0, 0), time_s=now, sequence=1, generation=generation
                            )
                        else:
                            getattr(lease, method)(now)
                    self.assert_zero(lease.resolve(1.0), "invalid_time")
                    with self.assertRaises(ValueError):
                        lease.arm(0.99)
                    self.assertEqual(lease.arm(1.01), generation + 1)
                    self.assert_zero(lease.resolve(1.01), "armed_waiting")

    def test_receipt_cannot_predate_prior_decision_or_be_future_of_next_decision(self):
        lease, generation = self.live()
        lease.resolve(1.1)
        with self.assertRaises(ValueError):
            lease.receive((1, 0, 0), time_s=1.09, sequence=1, generation=generation)
        self.assert_zero(lease.resolve(1.1), "invalid_time")
        generation = lease.arm(1.2)
        self.assertTrue(
            lease.receive((1, 0, 0), time_s=2.0, sequence=0, generation=generation)
        )
        with self.assertRaises(ValueError):
            lease.resolve(1.9)
        self.assert_zero(lease.resolve(2.0), "invalid_time")

    def test_timeout_validation(self):
        for value in (0, -0.1, math.nan, math.inf, True, "0.25", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BodyTwistLease(value)
        lease = BodyTwistLease(0.1)
        self.assertEqual(lease.timeout_s, 0.1)
        with self.assertRaises(AttributeError):
            lease.timeout_s = 5


class SequenceControlTests(unittest.TestCase):
    def test_sequence_source_preserves_command_and_pause_status_without_a_lease(self):
        source = SequenceControl()
        self.assertFalse(source.reset_requested)
        self.assertFalse(source.quit_requested)
        self.assertEqual(source.resolve(0.0).status, "scripted:initial_stand")
        source.command = (0.4, -0.2, 0.5)
        decision = source.poll(10.0, available=True)
        self.assertEqual(decision.command, source.command)
        self.assertIsNone(decision.source_time_s)
        self.assertIsNone(decision.sequence)
        self.assertEqual(decision.generation, 0)
        paused = source.poll(10.02, available=False)
        self.assertEqual(paused.command, (0.0, 0.0, 0.0))
        self.assertEqual(paused.status, "scripted:paused")
        source.stop(10.04)
        self.assertEqual(source.resolve(10.04).status, "scripted:finished")


if __name__ == "__main__":
    unittest.main()

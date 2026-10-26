"""Lab 2 autograder tests: nanoos/sched_sim.py. Expected times are worked out by hand in the comments."""

from __future__ import annotations

import unittest

from nanoos.sched_sim import (Program, Request, Simulator, fcfs_per_call, jain, least_attained, make_programs, p99,
                              percentile, program_fcfs)


def jct(programs, policy, **sim):
    sim.setdefault("gpu_slots", 1)
    sim.setdefault("reprefill_ms", lambda p: 0.0)
    return Simulator(**sim).run(programs, policy).jct_ms


class TestMetrics(unittest.TestCase):
    def test_jain(self):
        self.assertAlmostEqual(jain([3, 3, 3]), 1.0)
        self.assertAlmostEqual(jain([1, 0, 0]), 1 / 3)
        self.assertAlmostEqual(jain([1, 2]), 9 / 10)

    def test_percentile_nearest_rank(self):
        xs = list(range(1, 101))
        self.assertEqual(p99(xs), 99)
        self.assertEqual(percentile(xs, 50), 50)
        self.assertEqual(p99([5.0]), 5.0)
        self.assertEqual(percentile([3, 1, 2], 100), 3)


class TestPolicyKeys(unittest.TestCase):
    def test_keys_order_requests(self):
        old = Program("a", 0.0, [(10, 0)] * 3)
        new = Program("b", 50.0, [(10, 0)])
        r_old = Request(old, step=2, ready_ms=100.0, attained_ms=20.0)
        r_new = Request(new, step=0, ready_ms=60.0, attained_ms=0.0)
        self.assertLess(fcfs_per_call(r_new), fcfs_per_call(r_old))     # earlier request first
        self.assertLess(program_fcfs(r_old), program_fcfs(r_new))       # older program first
        self.assertLess(least_attained(r_new), least_attained(r_old))   # less GPU time so far first


class TestSimulator(unittest.TestCase):
    def test_single_gpu_fcfs_ties_by_id(self):
        # A 0-10, B 10-20
        ps = [Program("B", 0, [(10, 0)]), Program("A", 0, [(10, 0)])]
        self.assertEqual(jct(ps, fcfs_per_call), {"A": 10, "B": 20})

    def programs(self):
        return [Program("A", 0, [(10, 0)] * 3), Program("B", 5, [(10, 0)] * 2), Program("C", 15, [(10, 0)])]

    def test_per_call_fcfs(self):
        # 0-10 A0 | 10-20 B0 (ready 5) | 20-30 A1 (ready 10) | 30-40 C (ready 15) | 40-50 B1 | 50-60 A2
        self.assertEqual(jct(self.programs(), fcfs_per_call), {"A": 60, "B": 45, "C": 25})

    def test_program_fcfs(self):
        # A runs to completion first (0-30), then B (30-50), then C (50-60)
        self.assertEqual(jct(self.programs(), program_fcfs), {"A": 30, "B": 45, "C": 45})

    def test_least_attained(self):
        # 0-10 A0 | 10-20 B0 | 20-30 C (0 ms attained) | 30-40 A1 (ready 10) | 40-50 B1 | 50-60 A2
        self.assertEqual(jct(self.programs(), least_attained), {"A": 60, "B": 45, "C": 15})

    def test_parallel_gpus(self):
        ps = [Program("A", 0, [(10, 0)]), Program("B", 0, [(10, 0)]), Program("C", 0, [(10, 0)])]
        self.assertEqual(jct(ps, fcfs_per_call, gpu_slots=2), {"A": 10, "B": 10, "C": 20})

    def test_tool_time_and_last_tool(self):
        # arrives at 7: 7-17 llm, 17-47 tool, 47-57 llm, 57-62 tool -> JCT 62 - 7 = 55
        self.assertEqual(jct([Program("A", 7, [(10, 30), (10, 5)])], fcfs_per_call), {"A": 55})


class TestKV(unittest.TestCase):
    def prog(self):
        return [Program("A", 0, [(10, 50), (10, 0)])]

    def test_kept_kv_avoids_reprefill(self):
        r = Simulator(gpu_slots=1, kv_slots=1, reprefill_ms=lambda p: 100).run(self.prog(), fcfs_per_call)
        self.assertEqual((r.jct_ms, r.reprefills, r.kv_kept), ({"A": 70}, 0, 1))

    def test_ttl_shorter_than_tool_drops_kv(self):
        r = Simulator(gpu_slots=1, kv_slots=1, reprefill_ms=lambda p: 100, kv_ttl_ms=20).run(self.prog(), fcfs_per_call)
        self.assertEqual((r.jct_ms, r.reprefills, r.kv_kept), ({"A": 170}, 1, 0))

    def test_ttl_longer_than_tool_keeps_kv(self):
        r = Simulator(gpu_slots=1, kv_slots=1, reprefill_ms=lambda p: 100, kv_ttl_ms=60).run(self.prog(), fcfs_per_call)
        self.assertEqual(r.jct_ms, {"A": 70})

    def test_no_kv_slots(self):
        r = Simulator(gpu_slots=1, kv_slots=0, reprefill_ms=lambda p: 100).run(self.prog(), fcfs_per_call)
        self.assertEqual(r.jct_ms, {"A": 170})

    def test_kv_slot_contention_in_dispatch_order(self):
        # Both finish their first call at 10; A was dispatched first, so A keeps the only slot.
        ps = [Program("A", 0, [(10, 50), (10, 0)]), Program("B", 0, [(10, 50), (10, 0)])]
        r = Simulator(gpu_slots=2, kv_slots=1, reprefill_ms=lambda p: 100).run(ps, fcfs_per_call)
        self.assertEqual(r.jct_ms, {"A": 70, "B": 170})

    def test_reprefill_counts_as_attained_gpu_time(self):
        r = Simulator(gpu_slots=1, kv_slots=0, reprefill_ms=lambda p: 100).run(self.prog(), fcfs_per_call)
        self.assertEqual(r.gpu_busy_ms, 120)


class TestWorkload(unittest.TestCase):
    def test_every_program_finishes_under_every_policy(self):
        ps = make_programs(n=40, seed=3)
        for pol in (fcfs_per_call, program_fcfs, least_attained):
            with self.subTest(policy=pol.__name__):
                r = Simulator(gpu_slots=4, kv_slots=8).run(ps, pol)
                self.assertEqual(set(r.jct_ms), {p.id for p in ps})
                for p in ps:
                    self.assertGreaterEqual(r.jct_ms[p.id], sum(a + b for a, b in p.steps) - 1e-9)

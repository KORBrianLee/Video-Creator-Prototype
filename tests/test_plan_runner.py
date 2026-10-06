import json
import os
from pathlib import Path
import tempfile
import unittest

from local_video import digest, plan_runner


class FakeControl:
    """Jobs finish after a fixed number of status checks with a preset digest."""
    def __init__(self, outcomes, reject=()):
        self.outcomes, self.reject, self.submitted, self.checks = list(outcomes), set(reject), [], {}

    def submit(self, request):
        if request.get("id") in self.reject:
            raise ValueError("RAM 부족으로 거절")
        job = f"job{len(self.submitted)}"
        self.submitted.append(job)
        self.checks[job] = 0
        return {"job_id": job}

    def status(self, job):
        self.checks[job] += 1
        return {"state": "completed" if self.checks[job] >= 3 else "running"}


class PlanRunnerTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "plan-runner-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.report = Path(self.temporary.name)

    def run_plan(self, runs, digests, reject=(), **plan):
        control = FakeControl(digests, reject)
        queue = list(digests)
        def digest_job(job):
            return dict(queue.pop(0))
        sleeps = []
        summary = plan_runner.run_plan({"name": "t", "runs": runs, **plan}, control, digest_job, self.report,
                                       sleep=sleeps.append, clock=lambda: 0.0)
        return summary, control, sleeps

    def runs(self, *ids):
        return [{"id": i, "request": {"id": i}} for i in ids]

    def test_waits_locally_and_writes_one_small_summary(self):
        ok = {"state": "completed", "seconds": 300, "warnings": [], "stages_s_gib": {"video_infer": [120, 2.7]}}
        summary, control, sleeps = self.run_plan(self.runs("a", "b"), [ok, ok])
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(len(control.submitted), 2)
        self.assertEqual(len(sleeps), 4, "the runner, not the agent, waits between status checks")
        saved = (self.report / "summary.json").read_text(encoding="utf-8")
        self.assertLessEqual(len(saved), plan_runner.SUMMARY_LIMIT)
        self.assertTrue((self.report / "a.json").is_file())

    def test_collapse_skips_longer_runs(self):
        dark = {"state": "completed", "warnings": ["darkening: luma 90->30"]}
        summary, control, _ = self.run_plan(self.runs("2s", "8s", "15s"), [{"state": "completed", "warnings": []}, dark, dark])
        self.assertEqual(len(control.submitted), 2)
        self.assertIn("8s shows collapse", summary["stop_reason"])

    def test_same_failure_twice_stops_and_attempt_limit_is_six(self):
        failed = {"state": "failed", "stage": "stopped", "error": "UR error. details"}
        summary, control, _ = self.run_plan(self.runs("a", "b", "c"), [failed, failed, failed])
        self.assertEqual(len(control.submitted), 2)
        self.assertEqual(summary["stop_reason"], "same failure twice")
        ok = {"state": "completed", "warnings": []}
        summary, control, _ = self.run_plan(self.runs(*"abcdefgh"), [ok] * 8, max_attempts=10)
        self.assertEqual(summary["attempts"], 6)
        self.assertIn("attempt limit", summary["stop_reason"])

    def test_rejected_submissions_count_and_repeat_rejection_stops(self):
        summary, control, _ = self.run_plan(self.runs("a", "b", "c"), [], reject={"a", "b", "c"})
        self.assertEqual(summary["stop_reason"], "same rejection twice")
        self.assertEqual([r["state"] for r in summary["runs"]], ["rejected", "rejected"])


class QualityWarningTests(unittest.TestCase):
    def test_signals_from_real_failures(self):
        # Neo 15s measured on this PC: luma 89 -> 64 -> 48 -> 65 -> 52 -> 33 -> 39, saturation rising to 96.
        warnings = digest.quality_warnings([89, 64, 48, 65, 52, 33, 39], [42, 32, 16, 15, 36, 59, 96])
        self.assertTrue(any(w.startswith("darkening") for w in warnings))
        self.assertTrue(any(w.startswith("colour_blowout") for w in warnings))
        # Neo 8s on this PC: darkened to 49 mid-clip and partly recovered; the end-only comparison missed it.
        neo_8s = [92.9, 87.7, 81.9, 76.0, 68.4, 60.1, 53.7, 48.7, 49.0, 51.9, 63.7, 74.4, 51.4]
        self.assertTrue(any(w.startswith("darkening") for w in digest.quality_warnings(neo_8s, [40] * 13)))
        # Neo 2s on XPU: gentle drift 105 -> 90 is not a collapse.
        neo_2s = [105.0, 103.8, 103.5, 98.9, 97.3, 94.4, 93.5, 91.4, 91.5, 90.6, 90.5, 89.7, 90.3]
        self.assertEqual(digest.quality_warnings(neo_2s, [40] * 13), [])
        self.assertIn("static_suspected", digest.quality_warnings([80.2] * 6, [40.1] * 6))
        self.assertIn("near_black_frames", digest.quality_warnings([90, 80, 5, 70], [40, 40, 40, 40]))
        self.assertEqual(digest.quality_warnings([90, 88, 86, 87, 85, 84], [40, 41, 42, 40, 43, 41]), [])


if __name__ == "__main__":
    unittest.main()

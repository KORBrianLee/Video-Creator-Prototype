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


class MotionSignalTests(unittest.TestCase):
    def background(self):
        # Real footage is textured; on a flat background a zoom or shift would also "explain" a moving subject.
        from PIL import Image, ImageFilter
        image = Image.new("L", (120, 80))
        image.putdata([(x * 37 + y * 91 + (x * y) % 23 * 11) % 256 for y in range(80) for x in range(120)])
        return image.filter(ImageFilter.GaussianBlur(1))

    def frames(self, moving):
        from PIL import ImageDraw
        frames = []
        for index in range(6):
            image = self.background().crop((10, 10, 106, 64))
            for x in range(0, 96, 7):  # a little per-frame grain
                point = (x, (x * 3 + index) % 54)
                image.putpixel(point, min(255, image.getpixel(point) + index % 3))
            if moving:
                ImageDraw.Draw(image).rectangle((10 + index * 12, 20, 19 + index * 12, 34), fill=255)
            else:
                ImageDraw.Draw(image).rectangle((10, 20, 19, 34), fill=255)
            frames.append(image)
        return frames

    def test_a_still_view_warns_and_a_subject_crossing_it_does_not(self):
        still = digest.motion_signals(self.frames(moving=False))
        crossing = digest.motion_signals(self.frames(moving=True))
        self.assertLess(still["local_p99"], digest.LITTLE_MOTION_P99)
        self.assertGreater(crossing["local_p99"], digest.LITTLE_MOTION_P99)
        self.assertEqual(digest.motion_warnings(still), [f"little_motion: p99 {still['local_p99']}"])
        self.assertEqual(digest.motion_warnings(crossing), [])
        self.assertEqual(digest.motion_signals(self.frames(moving=False)[:1]), {})

    def test_a_tilting_camera_is_told_apart_from_a_subject_crossing_a_locked_view(self):
        background = self.background()
        crossing = digest.motion_signals(self.frames(moving=True))
        tilting = digest.motion_signals([background.crop((10, 10 + 2 * i, 106, 64 + 2 * i)) for i in range(6)])
        self.assertLess(crossing["camera_fit"], digest.CAMERA_MOVED_FIT)
        self.assertGreaterEqual(tilting["camera_fit"], digest.CAMERA_MOVED_FIT)
        self.assertEqual(digest.motion_warnings(tilting), [f"camera_moved: fit {tilting['camera_fit']}"])


class LongVideoNotesTests(unittest.TestCase):
    def test_step_timing_and_a_cut_prompt_are_reported(self):
        phases = [{"name": "first_frame"},
                  {"name": "ltx_generate", "gpu": {"timing": {"text_seconds": 40.2, "load_seconds": 12.0,
                                                              "denoise_step_seconds": [233.7, 146.0, 150.0],
                                                              "decode_seconds": 300.5},
                                                   "prompt": {"tokens": 142, "limit": 128, "truncated": True}}}]
        timing, warnings = digest.long_video_notes(phases)
        self.assertEqual(timing, {"text": 40.2, "load": 12.0, "first_step": 233.7, "per_step": 148.0, "steps": 3,
                                  "decode": 300.5})
        self.assertEqual(warnings, ["prompt_truncated: 142>128 tokens"])

    def test_older_records_without_timing_add_nothing(self):
        self.assertEqual(digest.long_video_notes([{"name": "ltx_generate", "gpu": {"backend": "torch_xpu"}}]), (None, []))


if __name__ == "__main__":
    unittest.main()

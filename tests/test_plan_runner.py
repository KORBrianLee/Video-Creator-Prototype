import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

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


class ReviewStripTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "strip-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)

    def video(self, width=64, height=36, name="clip", colour=(230, 200, 120), box=(0, 10, 8, 26)):
        from PIL import Image, ImageDraw
        from local_video import backend
        folder = self.work / name
        folder.mkdir()
        for index in range(48):
            frame = Image.new("RGB", (width, height), (20, 40, 80))
            ImageDraw.Draw(frame).rectangle((box[0] + index, box[1], box[2] + index, box[3]), fill=colour)
            frame.save(folder / f"frame_{index:03d}.png")
        video = folder / f"{name}.mp4"
        backend._encode_frames(folder, video, {"width": width, "height": height, "frames": 48, "fps": 24, "steps": 4},
                               lambda: False)
        return video

    def test_clips_of_different_sizes_line_up_row_by_row(self):
        from PIL import Image
        small, large = self.video(), self.video(128, 72, "large")
        result = digest.compare_strips([small, large], self.work / "compare.jpg", seconds=(0.5, 1.5))
        self.assertEqual(len(result["rows"]), 2)
        with Image.open(result["path"]) as sheet:
            self.assertEqual(sheet.size, (2 * 320, 2 * (180 + 14)))

    def test_frames_across_the_span_are_tiled_with_their_seconds(self):
        from PIL import Image
        result = digest.review_strip(self.video(), self.work / "strip.jpg", count=4)
        self.assertEqual(result["seconds"], [0.0, 0.7, 1.3, 2.0])
        with Image.open(result["path"]) as sheet:
            self.assertEqual(sheet.size, (4 * 320, 180 + 16))

    def test_the_box_is_magnified(self):
        from PIL import Image
        result = digest.review_strip(self.video(), self.work / "half.jpg", box=(0.5, 0.0, 1.0, 1.0), start=0.5, end=1.5,
                                     count=2, columns=2)
        self.assertEqual(result["seconds"], [0.5, 1.5])
        with Image.open(result["path"]) as sheet:
            self.assertEqual(sheet.size, (2 * 320, 360 + 16))

    def test_workers_are_found_by_their_vests_but_machines_are_not(self):
        from PIL import Image, ImageDraw
        frame = Image.new("RGB", (512, 288), (120, 110, 95))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((100, 120, 111, 137), fill=(200, 255, 30))
        draw.rectangle((100, 129, 111, 130), fill=(210, 210, 210))
        draw.rectangle((300, 100, 420, 200), fill=(255, 110, 0))
        draw.rectangle((200, 60, 260, 120), fill=(220, 165, 30))
        boxes = digest.vest_boxes(frame)
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = boxes[0]
        self.assertTrue(x0 * 512 < 100 < 111 < x1 * 512 and y0 * 288 < 120 < 137 < y1 * 288)
        self.assertGreaterEqual(round(x1 - x0, 6), 0.08)
        self.assertGreaterEqual(round(y1 - y0, 6), 0.28)

    def test_the_people_sheet_has_a_row_of_worker_crops_per_moment(self):
        from PIL import Image
        video = self.video(512, 288, "worker", colour=(200, 255, 30), box=(100, 120, 111, 137))
        result = digest.people_sheet(video, self.work / "people.jpg", count=3)
        self.assertEqual(result["found"], {"0.0": 1, "1.0": 1, "2.0": 1})
        with Image.open(result["path"]) as sheet:
            self.assertEqual(sheet.height, 3 * (160 + 14))


class FreshDigestTests(unittest.TestCase):
    def test_the_digest_comes_from_a_new_process_running_the_deployed_code(self):
        calls = []
        def run(command, **options):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=b'{"job": "abc", "state": "failed", "error": "x"}\n')
        self.assertEqual(plan_runner.fresh_digest("abc", run=run), {"job": "abc", "state": "failed", "error": "x"})
        self.assertEqual(calls[0][-2:], ["digest", "abc"])
        self.assertTrue(calls[0][1].endswith("video_agent.py"))

    def test_a_tool_error_or_broken_output_falls_back_to_this_process(self):
        for stdout in [b'{"error": "not found"}\n', b"", b"not json"]:
            run = lambda command, **options: subprocess.CompletedProcess(command, 1, stdout=stdout)
            with mock.patch.object(digest, "job_digest", return_value={"job": "abc", "state": "completed"}) as local:
                self.assertEqual(plan_runner.fresh_digest("abc", run=run)["state"], "completed")
                local.assert_called_once_with("abc")
        def hung(command, **options):
            raise subprocess.TimeoutExpired(command, 600)
        with mock.patch.object(digest, "job_digest", return_value={"state": "completed"}):
            self.assertEqual(plan_runner.fresh_digest("abc", run=hung), {"state": "completed"})


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

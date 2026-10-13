import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from local_video import cleanup

RECORDED, OLD, RECENT, FAILED, JUST_FAILED, RUNNING = (c * 12 for c in "abcdef")


class CleanupTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "cleanup-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.now = time.time()
        self.write("reports/STATE.json", {"best": {"job": RECORDED}})
        for job, state, hours in [(RECORDED, "completed", 200), (OLD, "completed", 100), (RECENT, "completed", 2),
                                  (FAILED, "failed", 2), (JUST_FAILED, "failed", 0.2), (RUNNING, "running", 0.5)]:
            self.write(f"jobs/{job}/status.json", {"job_id": job, "state": state})
            self.write(f"jobs/{job}/output/video.mp4", "x")
            self.age(f"jobs/{job}", hours)
        for name, job in [("gone", "9" * 12), ("kept", RECORDED), ("removed", OLD)]:
            self.write(f"cache/request-index/{name}.json", {"job_id": job})
        for name, hours in [("old", 100), ("new", 2)]:
            self.write(f"cache/shots/{name}/clip.mp4", "x")
            self.age(f"cache/shots/{name}", hours)
            self.write(f"cache/conditioning/stage/{name}/frame.png", "x")
            self.age(f"cache/conditioning/stage/{name}", hours)
        for name, hours in [("scene-recent", 2), ("scene-stale", 30)]:
            self.write(f"tmp/{name}/frame.png", "x" * 2**21)
            self.age(f"tmp/{name}", hours)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")

    def age(self, relative, hours):
        stamp = self.now - hours * 3600
        os.utime(self.root / relative, (stamp, stamp))

    def listed(self, days=3, refresh=None):
        found, _, _ = cleanup.candidates(self.root, days, self.now, refresh)
        return {path.relative_to(self.root).as_posix(): reason for path, reason in found}

    def test_garbage_is_listed_while_recorded_active_and_recent_work_stays(self):
        self.assertEqual(self.listed(), {
            f"jobs/{OLD}": "not_recorded", f"jobs/{FAILED}": "failed",
            "cache/request-index/gone.json": "job_gone", "cache/request-index/removed.json": "job_gone",
            "tmp/scene-stale": "scratch"})

    def test_preview_summarises_without_deleting(self):
        result = cleanup.clean(self.root, now=self.now)
        self.assertFalse(result["applied"])
        self.assertEqual(result["count"], 5)
        self.assertEqual({item["job"]: item["reason"] for item in result["jobs"]}, {OLD: "not_recorded", FAILED: "failed"})
        self.assertEqual(result["by_reason"]["job_gone"]["count"], 2)
        self.assertEqual([item["path"] for item in result["largest_other"]], ["tmp/scene-stale"])
        self.assertEqual(result["active_jobs"], [RUNNING])
        self.assertEqual(result["kept_recorded_jobs"], 1)
        self.assertTrue((self.root / "jobs" / OLD).is_dir())
        self.assertLess(len(json.dumps(result)), 2048)

    def test_apply_deletes_only_the_listed_paths(self):
        result = cleanup.clean(self.root, apply=True, now=self.now)
        self.assertTrue(result["applied"])
        self.assertEqual(result["count"], 5)
        for job in [RECORDED, RECENT, JUST_FAILED, RUNNING]:
            self.assertTrue((self.root / "jobs" / job / "output" / "video.mp4").is_file(), job)
        self.assertFalse((self.root / "jobs" / OLD).exists())
        self.assertFalse((self.root / "jobs" / FAILED).exists())
        self.assertTrue((self.root / "cache" / "request-index" / "kept.json").is_file())
        self.assertTrue((self.root / "tmp" / "scene-recent").is_dir())
        self.assertTrue((self.root / "cache" / "shots" / "old").is_dir())

    def test_idle_engine_also_clears_recent_scratch_and_old_clip_caches(self):
        self.write(f"jobs/{RUNNING}/status.json", {"job_id": RUNNING, "state": "completed"})
        self.age(f"jobs/{RUNNING}", 0.5)
        listed = self.listed()
        self.assertEqual(listed["tmp/scene-recent"], "scratch")
        self.assertEqual(listed["cache/shots/old"], "old_cache")
        self.assertEqual(listed["cache/conditioning/stage/old"], "old_cache")
        self.assertNotIn("cache/shots/new", listed)
        self.assertNotIn("cache/conditioning/stage/new", listed)

    def test_without_state_file_finished_jobs_stay(self):
        (self.root / "reports" / "STATE.json").unlink()
        listed = self.listed()
        self.assertNotIn(f"jobs/{OLD}", listed)
        self.assertNotIn(f"jobs/{RECORDED}", listed)
        self.assertEqual(listed[f"jobs/{FAILED}"], "failed")
        result = cleanup.clean(self.root, now=self.now)
        self.assertIsNone(result["kept_recorded_jobs"])
        self.assertIn("note", result)

    def test_a_running_job_whose_worker_died_is_cleaned_as_failed(self):
        self.age(f"jobs/{RUNNING}", 2)
        dead = lambda job: {"state": "failed"}
        self.assertEqual(self.listed(refresh=dead)[f"jobs/{RUNNING}"], "failed")
        self.assertEqual(cleanup.clean(self.root, now=self.now, refresh=dead)["active_jobs"], [])

    def test_days_widens_what_counts_as_unrecorded_old(self):
        self.assertNotIn(f"jobs/{OLD}", self.listed(days=5))


if __name__ == "__main__":
    unittest.main()

"""Resource isolation, image integrity and video-cache safety for the Neo backend."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_video import neodragon as neo, control


class NeoTests(unittest.TestCase):
    def setUp(self):
        parent = Path(tempfile.gettempdir()) / "neo-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "tmp" / "attempt"
        self.work.mkdir(parents=True)
        self.request = {"threads": 4, "minimum_free_ram_gib": 3.0,
                        "maximum_working_set_gib": 5.0, "reserve_ram_gib": 1.0}

    def child(self):
        class Child:
            pid = 123
            returncode = None
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -1

            def wait(self, timeout=None):
                return self.returncode
        return Child()

    def finished_child(self):
        process = self.child()
        process.returncode = 0
        return process

    def test_little_free_ram_still_starts_in_low_memory_mode(self):
        with patch.object(neo.media, "_memory", return_value=(0, int(0.8*2**30))), \
                patch("local_video.resources.commit_bytes", return_value=(34*2**30, 14*2**30)), \
                patch.object(neo.subprocess, "Popen", return_value=self.finished_child()) as launch:
            record = neo.run_stage(self.root, self.work, "video_infer", self.request, lambda update: None, lambda: False)
        launch.assert_called_once()
        self.assertEqual(record["state"], "completed")

    def test_exhausted_commit_never_starts_a_child(self):
        with patch.object(neo.media, "_memory", return_value=(0, 6*2**30)), \
                patch("local_video.resources.commit_bytes", return_value=(34*2**30, int(0.5*2**30))), \
                patch("local_video.resources.wait_for_commit", side_effect=MemoryError("commit 여유 없음")), \
                patch.object(neo.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(MemoryError, "commit"):
                neo.run_stage(self.root, self.work, "video_infer", self.request, lambda update: None, lambda: False)
        launch.assert_not_called()

    def test_conversion_needs_commit_only_when_not_prepared(self):
        needs = []
        def wait(needed, cancelled, progress):
            needs.append(needed)
            raise MemoryError("commit 여유 없음")
        with patch.object(neo.media, "_memory", return_value=(0, 4*2**30)), \
                patch("local_video.resources.wait_for_commit", side_effect=wait), \
                patch.object(neo.subprocess, "Popen") as launch:
            with self.assertRaises(MemoryError):
                neo.run_stage(self.root, self.work, "video_pack", self.request, lambda update: None, lambda: False)
        launch.assert_not_called()
        self.assertEqual(needs, [int(7.5 * 2**30)])

    def test_cancel_terminates_only_the_owned_phase_and_records_failure(self):
        process = self.child()
        cancelled = iter([False, True])
        with patch.object(neo.media, "_memory", return_value=(0, 8*2**30)), \
                patch.object(neo.subprocess, "Popen", return_value=process):
            with self.assertRaises(InterruptedError):
                neo.run_stage(self.root, self.work, "video_infer", self.request,
                              lambda update: None, lambda: next(cancelled))
        self.assertTrue(process.terminated)
        record = json.loads((self.work / "video_infer.metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "cancelled")
        self.assertEqual(record["process_id"], process.pid)

    def test_memory_cap_terminates_phase_and_keeps_existing_cache(self):
        process = self.child()
        untouched = self.root / "old-cache.mp4"
        untouched.write_bytes(b"previous output")
        with patch.object(neo.media, "_memory", side_effect=[(0, 8*2**30), (6*2**30, 8*2**30)]), \
                patch.object(neo.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(MemoryError, "작업 메모리가 사용자 지정 상한"):
                neo.run_stage(self.root, self.work, "video_infer", self.request, lambda update: None, lambda: False)
        self.assertTrue(process.terminated)
        self.assertEqual(untouched.read_bytes(), b"previous output")
        self.assertEqual(json.loads((self.work / "video_infer.metrics.json").read_text(encoding="utf-8"))["state"], "failed")

    def test_cache_requires_passed_safety_and_full_video_decode(self):
        clip = self.work / "clip.mp4"
        clip.write_bytes(b"test only")
        record = {"status": "complete", "cache_key": "test", "safety": {"unsafe": True},
                  "video_sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}
        metadata = self.work / "metadata.json"
        metadata.write_text(json.dumps(record))
        with patch.object(neo.media, "verify_video") as verify:
            self.assertIsNone(neo.cached(self.work, "test", {}, lambda: False))
        verify.assert_not_called()
        record["safety"]["unsafe"] = False
        metadata.write_text(json.dumps(record))
        with patch.object(neo.media, "verify_video", side_effect=RuntimeError("damaged")):
            self.assertIsNone(neo.cached(self.work, "test", {}, lambda: False))

    def test_changed_input_image_is_rejected_before_any_engine(self):
        image = self.root / "reference.png"
        image.write_bytes(b"changed")
        scene = {"image_path": str(image), "image_sha256": "0"*64}
        with patch.object(neo.media, "_engine") as engine:
            with self.assertRaisesRegex(RuntimeError, "변경"):
                neo.first_frame(self.root, self.work, scene, neo.PRESETS["preview"], self.request,
                                lambda update: None, lambda: False)
        engine.assert_not_called()

    def test_quality_gate_detects_same_size_video_tampering(self):
        app = self.root / "app"
        folder = app / "docs" / "quality"
        folder.mkdir(parents=True)
        video = self.root / "baseline.mp4"
        video.write_bytes(b"validated")
        validation = {"state": "passed_cpu_visual_review", "pipeline_revision": "verified",
                      "video_path": str(video), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                      "model_revision": hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()}
        (folder / "neodragon-cpu.json").write_text(json.dumps(validation))
        with patch.object(control, "APP", app), patch.object(neo, "pipeline_revision", return_value="verified"), \
                patch.object(control, "manifest", return_value={}), \
                patch.object(control, "load_config", return_value=({"backend": "cpu"}, self.root)):
            self.assertTrue(control.quality_validation("neodragon", self.root)[1])
            video.write_bytes(b"corrupted")
            self.assertEqual(control.quality_validation("neodragon", self.root), ("validation_artifact_changed", False))

    def test_no_model_imports_in_controller_or_mcp(self):
        import sys
        self.assertNotIn("torch", sys.modules)
        env = neo.environment(self.root, 4)
        for key in ("TEMP", "TMP", "HF_HOME", "TORCH_HOME"):
            self.assertTrue(Path(env[key]).resolve().is_relative_to(self.root))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_same_size_modified_model_is_rejected_before_inference(self):
        from local_video.integrity import verify
        models = self.root / "models"
        models.mkdir()
        model = models / "test-model.bin"
        model.write_bytes(b"pinned")
        item = {"filename": model.name, "size": 6, "sha256": hashlib.sha256(b"pinned").hexdigest()}
        request = {"_model_spec": {"files": [item]}}
        verify(request, self.root, lambda update: None, lambda: False)
        model.write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            verify(request, self.root, lambda update: None, lambda: False)


class FailureTailTests(unittest.TestCase):
    def test_keeps_the_last_error_line_without_caret_markers(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEMP")) as folder:
            log = Path(folder) / "ltx_generate.log"
            log.write_text("step 1\r\nTraceback\n  File x\n    noise = torch.randn(...)\n    ^^^^^^\n"
                           "RuntimeError: Expected a 'xpu' device type\n\n", encoding="utf-8")
            self.assertEqual(neo.failure_tail(log), "RuntimeError: Expected a 'xpu' device type")
            self.assertEqual(neo.failure_tail(Path(folder) / "missing.log"), "기록 없음")


if __name__ == "__main__":
    unittest.main()

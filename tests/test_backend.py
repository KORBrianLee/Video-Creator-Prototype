"""Encode/decode, cache integrity, cancellation and safe engine arguments."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from local_video import backend


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cvl-backend-")
        self.work = Path(self.temporary.name)
        self.settings = {"width": 64, "height": 64, "frames": 8, "fps": 8, "steps": 4}

    def tearDown(self):
        self.temporary.cleanup()

    def frames(self, static=False):
        from PIL import Image, ImageDraw
        for index in range(8):
            frame = Image.new("RGB", (64, 64), (20, 40, 80))
            drawing = ImageDraw.Draw(frame)
            offset = 0 if static else index * 5
            drawing.rectangle((offset, 20, offset+15, 40), fill=(230, 200, 120))
            frame.save(self.work / f"frame_{index:03d}.png")
        video = self.work / "clip.mp4"
        backend._encode_frames(self.work, video, self.settings, lambda: False)
        return video

    def test_generated_frame_encoding_full_decode_and_movement(self):
        result = backend.verify_video(self.frames(), self.settings)
        self.assertTrue(result["decoded"])
        self.assertTrue(result["nonstatic"])
        self.assertEqual(result["frame_count"], 8)
        self.assertEqual(result["duration_seconds"], 1.0)
        self.assertGreater(result["mean_adjacent_frame_luma_difference"], 0.02)

    def test_static_clip_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "프레임 변화"):
            backend.verify_video(self.frames(static=True), self.settings)

    def test_wrong_frame_count_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "프레임 수"):
            backend.verify_video(self.frames(), self.settings, expected_frames=9)

    def test_corrupt_clip_never_reuses_cache(self):
        video = self.frames()
        metadata = {"status": "complete", "cache_key": "key", "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
        backend._write_json(self.work / "metadata.json", metadata)
        self.assertIsNotNone(backend._cached(self.work, "key", self.settings, lambda: False))
        video.write_bytes(video.read_bytes()[:128])
        self.assertIsNone(backend._cached(self.work, "key", self.settings, lambda: False))

    def test_corrupt_clip_with_matching_hash_still_requires_full_decode(self):
        video = self.frames()
        video.write_bytes(video.read_bytes()[:128])
        backend._write_json(self.work / "metadata.json", {"status": "complete", "cache_key": "key", "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
        self.assertIsNone(backend._cached(self.work, "key", self.settings, lambda: False))

    def test_cache_verification_propagates_cancel(self):
        video = self.frames()
        backend._write_json(self.work / "metadata.json", {"status": "complete", "cache_key": "key", "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
        with self.assertRaises(InterruptedError):
            backend._cached(self.work, "key", self.settings, lambda: True)

    def test_concat_uses_exact_original_frames_without_interpolation(self):
        video = self.frames()
        shots = [{"id": "s01", "video_path": str(video), "seed": 42}, {"id": "s02", "video_path": str(video), "seed": 43}]
        output, sheet = self.work / "combined.mp4", self.work / "contact.jpg"
        backend._assemble(shots, output, sheet, self.settings, lambda: False)
        result = backend.verify_video(output, self.settings, expected_frames=16)
        self.assertEqual(result["duration_seconds"], 2.0)
        self.assertTrue(sheet.is_file())
        from PIL import Image
        with Image.open(sheet) as contact:
            self.assertEqual(contact.size, (1024, 560))

    def test_single_shot_preserves_encoded_bytes_without_another_lossy_pass(self):
        video = self.frames()
        output, sheet = self.work / "single.mp4", self.work / "single.jpg"
        shot = {"id": "s01", "video_path": str(video), "seed": 42}
        with mock.patch.object(backend, "_writer", side_effect=AssertionError("already encoded")):
            backend._assemble([shot], output, sheet, self.settings, lambda: False)
        self.assertEqual(video.read_bytes(), output.read_bytes())
        self.assertEqual(backend.verify_video(output, self.settings)["frame_count"], 8)
        self.assertTrue(sheet.is_file())

    def test_cache_key_tracks_model_prompt_seed_and_settings(self):
        request = {"model_revision": "a", "model_profile": "lightning", "preset": "preview", "_model_spec": {"sampling": {"steps": 4}}}
        scene = {"id": "s01", "prompt": "a running dog", "negative_prompt": "static", "seed": 42}
        key = backend._cache_key(request, scene, self.settings, "engine", "cpu")
        for changed in [{"seed": 43}, {"prompt": "a jumping dog"}, {"negative_prompt": "blurry"}]:
            self.assertNotEqual(key, backend._cache_key(request, {**scene, **changed}, self.settings, "engine", "cpu"))
        self.assertNotEqual(key, backend._cache_key({**request, "model_revision": "b"}, scene, self.settings, "engine", "cpu"))
        self.assertNotEqual(key, backend._cache_key(request, scene, self.settings, "different engine", "cpu"))
        self.assertEqual(key, backend._cache_key(request, {**scene, "id": "different"}, self.settings, "engine", "cpu"))

    def test_model_paths_reject_traversal(self):
        with self.assertRaises(ValueError):
            backend._model_paths({"files": [{"role": "base", "filename": "../base.gguf"}]}, self.work, "lightning")

    def model_spec(self):
        files = []
        for role, filename in [("base", "base.gguf"), ("motion", "motion.safetensors")]:
            data = (role + " good model").encode()
            (self.work / filename).write_bytes(data)
            files.append({"role": role, "filename": filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return {"files": files}

    def test_model_hash_verified_and_unchanged_fingerprint_reused(self):
        spec = self.model_spec()
        paths = backend._model_paths(spec, self.work, "lightning")
        self.assertEqual(set(paths), {"base", "motion"})
        self.assertTrue((self.work / "base.gguf.verified.json").is_file())
        with mock.patch.object(backend.hashlib, "sha256", side_effect=AssertionError("unchanged model must not rehash")):
            self.assertEqual(paths, backend._model_paths(spec, self.work, "lightning"))

    def test_same_size_modified_model_forces_hash_and_is_rejected(self):
        spec = self.model_spec()
        backend._model_paths(spec, self.work, "lightning")
        file = self.work / "base.gguf"
        info = file.stat()
        file.write_bytes(b"X"*info.st_size)
        os.utime(file, ns=(info.st_atime_ns, info.st_mtime_ns+1_000_000_000))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            backend._model_paths(spec, self.work, "lightning")

    def test_model_verification_propagates_cancellation(self):
        spec = self.model_spec()
        with self.assertRaises(InterruptedError):
            backend._model_paths(spec, self.work, "lightning", is_cancelled=lambda: True)

    def test_prepared_base_selected_and_portable_without_original(self):
        spec = self.model_spec()
        filename = "base-lightning-linear.gguf"
        data = b"prepared model fixture"
        file = self.work / filename
        file.write_bytes(data)
        spec["prepared_base"] = {"filename": filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                 "input_sha256": spec["files"][0]["sha256"], "preparation_version": "lightning-linear-alphas-gguf-v1"}
        paths = backend._model_paths(spec, self.work, "lightning")
        self.assertEqual(paths["base"], file.resolve())
        (self.work / "base.gguf").unlink()
        self.assertEqual(backend._model_paths(spec, self.work, "lightning")["base"], file.resolve())
        spec["prepared_base"]["input_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "원본"):
            backend._model_paths(spec, self.work, "lightning")

    def test_resolved_paths_cannot_leave_runtime(self):
        runtime, outside = self.work / "runtime", self.work / "outside"
        runtime.mkdir()
        outside.mkdir()
        with self.assertRaises(ValueError):
            backend._path_in_runtime(outside / "video.mp4", runtime.resolve())
        link = runtime / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            # Directory junctions do not need the symbolic-link privilege.
            creation = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(outside)],
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(creation.returncode, 0, creation.stdout+creation.stderr)
        with self.assertRaises(ValueError):
            backend._path_in_runtime(link / "video.mp4", runtime.resolve())

    def test_cancel_stops_only_owned_inference_child(self):
        progress = []
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 3
        request = {"_runtime_dir": str(self.work), "maximum_working_set_gib": 5, "reserve_ram_gib": 1}
        before = time.monotonic()
        with mock.patch.object(backend, "_memory", return_value=(20*1024**2, 8*1024**3)):
            with self.assertRaises(InterruptedError):
                backend._infer([sys.executable, "-c", "import time; time.sleep(60)"], self.work, request, progress.append, cancel, 1, 1, 4)
        self.assertLess(time.monotonic()-before, 5)
        self.assertTrue(progress)
        self.assertIn("engine_pid", progress[0])

    def test_memory_cap_terminates_owned_inference_child(self):
        request = {"_runtime_dir": str(self.work), "maximum_working_set_gib": 0.1, "reserve_ram_gib": 0.05}
        before = time.monotonic()
        with mock.patch.object(backend, "_memory", return_value=(1024**3, 8*1024**3)), \
             mock.patch.object(backend, "_reclaimable", return_value=0):
            with self.assertRaisesRegex(MemoryError, "사용자 지정 상한"):
                backend._infer([sys.executable, "-c", "import time; time.sleep(60)"], self.work, request, lambda update: None, lambda: False, 1, 1, 4)
        self.assertLess(time.monotonic()-before, 5)

    def test_mapped_model_pages_do_not_trip_memory_guards(self):
        request = {"_runtime_dir": str(self.work), "maximum_working_set_gib": 5, "reserve_ram_gib": 1.4}
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 12
        readings = iter([(20*1024**2, 6*1024**3)])
        with mock.patch.object(backend, "_memory", side_effect=lambda pid: next(readings, (6*1024**3, int(0.9*1024**3)))), \
             mock.patch.object(backend, "_reclaimable", return_value=4*1024**3):
            with self.assertRaises(InterruptedError):
                backend._infer([sys.executable, "-c", "import time; time.sleep(60)"], self.work, request, lambda update: None, cancel, 1, 1, 4)
        from local_video.resources import learned_drops
        self.assertAlmostEqual(learned_drops(self.work)["inference"], 1.1, delta=0.05)

    def test_tight_ram_keeps_running_and_trims_the_engine(self):
        request = {"_runtime_dir": str(self.work), "maximum_working_set_gib": "auto", "reserve_ram_gib": 1}
        checks, updates, trims = 0, [], []
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 20
        with mock.patch.object(backend, "_memory", return_value=(20*1024**2, 100*1024**2)), \
             mock.patch("local_video.resources.commit_bytes", return_value=(34*1024**3, 14*1024**3)), \
             mock.patch("local_video.resources.trim_working_set", side_effect=lambda pid: trims.append(pid) or True):
            with self.assertRaises(InterruptedError):
                backend._infer([sys.executable, "-c", "import time; time.sleep(60)"], self.work, request, updates.append, cancel, 1, 1, 4)
        self.assertTrue(trims, "tight RAM must trim the engine instead of stopping it")
        self.assertTrue(any(update.get("memory_pressure") for update in updates))

    def test_exhausted_commit_terminates_owned_inference_child(self):
        request = {"_runtime_dir": str(self.work), "maximum_working_set_gib": "auto", "reserve_ram_gib": 1}
        readings = iter([(34*1024**3, 14*1024**3), (34*1024**3, 14*1024**3)])
        with mock.patch.object(backend, "_memory", return_value=(20*1024**2, 8*1024**3)), \
             mock.patch("local_video.resources.commit_bytes", side_effect=lambda: next(readings, (34*1024**3, 100*1024**2))):
            with self.assertRaisesRegex(MemoryError, "commit 여유"):
                backend._infer([sys.executable, "-c", "import time; time.sleep(60)"], self.work, request, lambda update: None, lambda: False, 1, 1, 4)

    def test_safe_argument_list_and_fixed_lightning_schedule(self):
        sigma = [25.146, 6.557, 2.276, 0.867, 0]
        command = backend.build_command(Path("sd-cli.exe"), "cpu", "lightning", {"base": Path("base.gguf"), "motion": Path("motion.safetensors")}, backend.PRESETS["lightning"]["preview"], {"seed": 52}, self.work, 4, {"sigmas": sigma})
        self.assertEqual(command[command.index("--cfg-scale")+1], "1")
        self.assertEqual(command[command.index("--steps")+1], "4")
        self.assertEqual(command[command.index("--seed")+1], "52")
        self.assertIn("--motion-module", command)
        self.assertEqual(command[command.index("--params-backend")+1], "disk")
        self.assertIn("--mmap", command)
        self.assertNotIn("--temporal-tiling", command)
        self.assertEqual(command[command.index("--vae-tile-size")+1], "256x256")

    def test_unsupported_engine_option_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "필수 옵션"):
            backend._validate_flags(["engine", "--mmap", "--made-up"], "--mmap --backend")


if __name__ == "__main__":
    unittest.main()

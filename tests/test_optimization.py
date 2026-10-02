"""Cache invalidation, prepared-weight integrity and lossless clip assembly."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from local_video import conditioning, neodragon, prepared


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        parent = Path(tempfile.gettempdir()) / "optimization-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "tmp" / "attempt"
        self.work.mkdir(parents=True)
        self.app = Path(neodragon.__file__).resolve().parents[1]
        self.request = {"threads": 4, "model_revision": "pinned-models"}
        self.scene = {"prompt": "a beach wave", "seed": 42, "width": 512, "height": 320}
        self.write_scene()
        (self.work / "first-frame.png").write_bytes(b"first image")

    def write_scene(self):
        (self.work / "request.json").write_text(json.dumps(self.scene), encoding="utf-8")

    def runner(self):
        def create(root, work, stage, request, progress, is_cancelled):
            (work / conditioning.ARTIFACTS[stage]).write_bytes(b"checked tensor fixture")
            return {"name": stage, "state": "completed", "elapsed_seconds": 12,
                    "peak_working_set_bytes": 200 * 2**20}
        return Mock(side_effect=create)

    def cached(self, stage, runner, cancelled=lambda: False):
        return conditioning.run_cached(self.root, self.work, stage, self.request,
                                       lambda update: None, cancelled, runner)

    def test_text_cache_reuses_without_loading_any_model(self):
        runner = self.runner()
        self.assertFalse(self.cached("video_text", runner)["cache_hit"])
        self.assertTrue(self.cached("video_text", runner)["cache_hit"])
        self.assertEqual(runner.call_count, 1)
        self.assertEqual((self.work / "video-text.pt").read_bytes(), b"checked tensor fixture")

    def test_text_key_tracks_prompt_model_and_implementation(self):
        key = conditioning.cache_key(self.app, self.work, "video_text", self.request)
        self.scene["seed"] = 43
        self.write_scene()
        self.assertEqual(key, conditioning.cache_key(self.app, self.work, "video_text", self.request))
        self.scene["prompt"] = "a different wave"
        self.write_scene()
        self.assertNotEqual(key, conditioning.cache_key(self.app, self.work, "video_text", self.request))
        self.assertNotEqual(key, conditioning.cache_key(self.app, self.work, "video_text", {**self.request, "model_revision": "new model"}))

    def test_image_latent_key_tracks_seed_and_image_bytes(self):
        key = conditioning.cache_key(self.app, self.work, "video_encode", self.request)
        self.scene["prompt"] = "different motion for the same input"
        self.write_scene()
        self.assertEqual(key, conditioning.cache_key(self.app, self.work, "video_encode", self.request))
        self.scene["seed"] += 1
        self.write_scene()
        self.assertNotEqual(key, conditioning.cache_key(self.app, self.work, "video_encode", self.request))
        self.scene["seed"] -= 1
        self.write_scene()
        (self.work / "first-frame.png").write_bytes(b"other image")
        self.assertNotEqual(key, conditioning.cache_key(self.app, self.work, "video_encode", self.request))

    def test_text_cache_tracks_effective_prompt_modifier(self):
        key = conditioning.cache_key(self.app, self.work, "video_text", self.request)
        self.scene["prompt_modifier"] = ""
        self.write_scene()
        self.assertNotEqual(key, conditioning.cache_key(self.app, self.work, "video_text", self.request))

    def test_same_size_corrupt_conditioning_is_recomputed(self):
        runner = self.runner()
        record = self.cached("video_text", runner)
        cache = self.root / "cache" / "conditioning" / "video_text" / record["cache_key"]
        artifact = cache / "video-text.pt"
        artifact.write_bytes(b"X" * artifact.stat().st_size)
        self.assertFalse(self.cached("video_text", runner)["cache_hit"])
        self.assertEqual(runner.call_count, 2)
        self.assertEqual((self.work / "video-text.pt").read_bytes(), b"checked tensor fixture")

    def test_malformed_conditioning_metadata_is_recomputed(self):
        runner = self.runner()
        record = self.cached("video_text", runner)
        path = self.root / "cache" / "conditioning" / "video_text" / record["cache_key"] / "metadata.json"
        path.write_text("[]", encoding="utf-8")
        self.assertFalse(self.cached("video_text", runner)["cache_hit"])

    def test_cached_conditioning_respects_cancel_without_loading_models(self):
        runner = self.runner()
        self.cached("video_text", runner)
        with self.assertRaises(InterruptedError):
            self.cached("video_text", runner, lambda: True)
        self.assertEqual(runner.call_count, 1)

    def prepare_weight_fixture(self):
        path = self.root / "models" / "experimental-neodragon" / "derived" / "transformer-cpu-int8-v2.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"prepared INT8 fixture")
        spec = json.loads((self.app / "neodragon.lock.json").read_text(encoding="utf-8"))
        source = next(item for item in spec["files"] if item["filename"].endswith("diffusion_transformer_320p/diffusion_pytorch_model.safetensors"))
        runtime = json.loads((self.app / "neodragon-runtime.lock.json").read_text(encoding="utf-8"))
        version = next(item["version"] for item in runtime["packages"] if item["name"] == "torch")
        record = {"conversion": "neodragon-cpu-per-channel-int8-v2", "source_sha256": source["sha256"],
                  "source_revision": source["revision"], "torch": version,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".provenance.json").write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_valid_prepared_weights_do_not_launch_a_torch_process(self):
        self.prepare_weight_fixture()
        with patch.object(neodragon.subprocess, "Popen") as process, patch.object(neodragon.media, "_memory") as memory:
            result = neodragon.run_stage(self.root, self.work, "video_pack", self.request,
                                        lambda update: None, lambda: False)
        self.assertTrue(result["reused_derived_weights"])
        self.assertEqual(result["peak_working_set_bytes"], 0)
        process.assert_not_called()
        memory.assert_not_called()

    def test_same_size_modified_prepared_weights_are_not_reused(self):
        path = self.prepare_weight_fixture()
        self.assertIsNotNone(prepared.verify(self.root, self.app, lambda: False))
        info = path.stat()
        path.write_bytes(b"X" * info.st_size)
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns+1_000_000_000))
        self.assertIsNone(prepared.verify(self.root, self.app, lambda: False))

    def test_prepared_weights_keep_source_and_runtime_binding(self):
        path = self.prepare_weight_fixture()
        notice = path.with_suffix(".provenance.json")
        record = json.loads(notice.read_text(encoding="utf-8"))
        record["source_sha256"] = "0" * 64
        notice.write_text(json.dumps(record), encoding="utf-8")
        self.assertIsNone(prepared.verify(self.root, self.app, lambda: False))

    def test_prepared_verification_is_cancellable(self):
        self.prepare_weight_fixture()
        with self.assertRaises(InterruptedError):
            prepared.verify(self.root, self.app, lambda: True)


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import installer
from local_video import accelerator, neodragon, skyreels

GIB = 2**30


class SettingsTests(unittest.TestCase):
    def test_15_seconds_is_one_generation_of_377_frames_at_24_fps(self):
        settings = skyreels.settings_for(24 * GIB, 15)
        self.assertEqual((settings["frames"], settings["fps"]), (377, 24))
        self.assertEqual(skyreels.frames_for(10), 257)
        self.assertEqual(skyreels.frames_for(2), 49)
        for duration in (2, 4, 8, 10, 15):
            self.assertEqual((skyreels.frames_for(duration) - 1) % 4, 0, "Wan VAE needs 4n+1 frames")

    def test_settings_shrink_with_vram(self):
        big, mid, small, minimum = (skyreels.settings_for(v * GIB, 15) for v in (32, 16, 12, 8))
        self.assertEqual((big["width"], big["height"], big["base_num_frames"], big["text_encoder_device"]), (960, 544, 97, "cuda"))
        self.assertEqual((mid["width"], mid["base_num_frames"], mid["text_encoder_device"]), (960, 77, "cpu"))
        self.assertEqual((small["width"], small["height"]), (800, 448))
        self.assertEqual((minimum["width"], minimum["height"], minimum["base_num_frames"]), (672, 384, 57))
        for settings in (big, mid, small, minimum):
            self.assertEqual(settings["width"] % 16, 0)
            self.assertEqual(settings["height"] % 16, 0)
        self.assertFalse(big["vae_tiling"])
        self.assertTrue(minimum["vae_tiling"])

    def test_long_clips_use_overlapping_blocks_short_clips_do_not(self):
        long_clip, short_clip = skyreels.settings_for(16 * GIB, 15), skyreels.settings_for(16 * GIB, 2)
        self.assertEqual((long_clip["overlap_history"], long_clip["addnoise_condition"]), (17, 20))
        self.assertEqual((short_clip["overlap_history"], short_clip["base_num_frames"]), (None, 49))
        self.assertEqual(skyreels.settings_for(16 * GIB, 15, "quality")["steps"], 30)


class TierTests(unittest.TestCase):
    def tier(self, backend="cuda", discrete=True, vram=12 * GIB, installed=True, ltx_installed=False, failed=None):
        with mock.patch.object(accelerator, "recently_failed", side_effect=lambda root, name: (failed or {}).get(name)):
            return accelerator.model_tier(Path("runtime"), backend, discrete, vram, installed, ltx_installed)

    def test_nvidia_gets_skyreels_and_ltx_is_never_chosen_automatically(self):
        self.assertEqual(self.tier()[0], "skyreels")
        for kwargs in ({"backend": "xpu", "discrete": False, "vram": 7 * GIB}, {"backend": "xpu"}, {"vram": 6 * GIB}):
            tier, reason = self.tier(ltx_installed=True, **kwargs)
            self.assertEqual(tier, "neo")
            self.assertIn("ltx_opt_in_only", reason)

    def test_without_installed_long_video_models_neo_is_used_with_reasons(self):
        tier, reason = self.tier(installed=False)
        self.assertEqual(tier, "neo")
        self.assertIn("skyreels_not_installed", reason)
        self.assertIn("ltx_not_installed", reason)
        self.assertEqual(self.tier(backend="xpu", ltx_installed=False)[0], "neo")
        self.assertIn("nvidia_or_intel", self.tier(backend=None)[1])

    def test_recent_runtime_failure_routes_jobs_to_the_next_model(self):
        tier, reason = self.tier(vram=24 * GIB, ltx_installed=True, failed={"skyreels": "CUDA out of memory"})
        self.assertEqual(tier, "neo")
        self.assertIn("ltx_opt_in_only", reason)
        tier, reason = self.tier(vram=24 * GIB, ltx_installed=True, failed={"skyreels": "oom", "ltx": "UR error"})
        self.assertEqual(tier, "neo")
        self.assertIn("UR error", reason)


class DurationPolicyTests(unittest.TestCase):
    def test_neo_computers_take_long_clips_only_as_diagnostics(self):
        from local_video import control
        neo = {"accelerator": {"maximum_duration_seconds": 8, "video_model": "Neo", "model_tier_reason": "long_video_model_needs_nvidia_gpu"}}
        sky = {"accelerator": {"maximum_duration_seconds": 15, "video_model": "SkyReels"}}
        long_clip = {"continuous": True, "duration_seconds": 15, "diagnostic": False}
        with self.assertRaises(ValueError) as caught:
            control._require_supported_duration(long_clip, neo)
        self.assertIn("diagnostic=true", str(caught.exception))
        control._require_supported_duration({**long_clip, "diagnostic": True}, neo)
        control._require_supported_duration(long_clip, sky)
        control._require_supported_duration({"continuous": True, "duration_seconds": 8}, neo)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "skyreels-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "runtime"
        from PIL import Image
        Image.new("RGB", (960, 544), "orange").save(self.work / "first-frame.png")
        (self.work / "request.json").write_text(json.dumps({"width": 960, "height": 544, "frames": 377, "skyreels": {}}), encoding="utf-8")
        self.request = {"device_plan": {"model_tier": "skyreels", "torch_device": "cuda"}, "preset": "preview",
                        "continuous": True, "duration_seconds": 15, "scenes": [{"prompt": "x"}]}
        self.settings = {"width": 960, "height": 544, "frames": 377, "fps": 24}

    def test_failed_long_video_model_switches_to_neo_in_place(self):
        self.request["duration_seconds"] = 8
        neo_calls = []
        def runner(root, work, name, request, progress, cancelled):
            raise RuntimeError("CUDA error: out of memory while decoding")
        def cached_runner(root, work, name, request, progress, cancelled, inner):
            neo_calls.append(name)
            return {"name": name}
        progress = []
        phases = neodragon.generate_frames(self.root, self.work, self.request, self.settings, progress.append,
                                           lambda: False, runner, cached_runner)
        self.assertEqual(neo_calls, list(neodragon.NEO_FRAME_STAGES))
        self.assertEqual(len(phases), 5)
        self.assertEqual((self.settings["width"], self.settings["height"], self.settings["frames"]), (384, 256, 193))
        from PIL import Image
        with Image.open(self.work / "first-frame.png") as image:
            self.assertEqual(image.size, (384, 256))
        saved = json.loads((self.work / "request.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["width"], saved["frames"]), (384, 193))
        self.assertNotIn("skyreels", saved)
        self.assertEqual(self.request["device_plan"]["model_tier"], "neo")
        self.assertTrue(accelerator.recently_failed(self.root, "skyreels"))
        self.assertEqual(progress[-1]["stage"], "long_video_model_failed_using_neo")

    def test_long_clips_are_not_handed_to_neo_when_the_long_video_model_fails(self):
        self.request["device_plan"]["model_tier"] = "ltx"
        def runner(root, work, name, request, progress, cancelled):
            self.assertEqual(name, "ltx_generate")
            raise RuntimeError("UR error")
        with self.assertRaises(RuntimeError) as caught:
            neodragon.generate_frames(self.root, self.work, self.request, self.settings, lambda update: None,
                                      lambda: False, runner, lambda *args: self.fail("Neo must not make a 15s clip"))
        self.assertIn("Neo", str(caught.exception))
        self.assertTrue(accelerator.recently_failed(self.root, "ltx"))

    def test_memory_shortage_is_not_treated_as_model_failure(self):
        def runner(*args):
            raise MemoryError("low RAM")
        with self.assertRaises(MemoryError):
            neodragon.generate_frames(self.root, self.work, self.request, self.settings, lambda update: None,
                                      lambda: False, runner, lambda *args: {})
        self.assertIsNone(accelerator.recently_failed(self.root, "skyreels"))

    def test_successful_long_video_model_skips_neo(self):
        def runner(root, work, name, request, progress, cancelled):
            return {"name": name, "state": "completed"}
        phases = neodragon.generate_frames(self.root, self.work, self.request, self.settings, lambda update: None,
                                           lambda: False, runner, lambda *args: self.fail("Neo must not run"))
        self.assertEqual([phase["name"] for phase in phases], ["skyreels_generate"])


class LtxTests(unittest.TestCase):
    def test_15_seconds_is_one_pass_within_the_temporal_position_range(self):
        from local_video import ltx
        settings = ltx.settings_for(15)
        self.assertEqual((settings["frames"], settings["fps"]), (361, 24))
        self.assertEqual((settings["frames"] - 1) % 8, 0, "LTX VAE needs 8n+1 frames")
        self.assertLessEqual(settings["frames"] / settings["fps"], 20, "positional_embedding_max_pos time range is 20s")
        self.assertEqual((settings["width"] % 32, settings["height"] % 32), (0, 0))
        self.assertEqual(len(settings["timesteps"]), 8)
        self.assertEqual(settings["guidance_scale"], 1.0)

    def test_lock_and_install_only_for_intel_gpus(self):
        lock = json.loads((installer.SOURCE / "ltx.lock.json").read_text(encoding="utf-8"))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in lock["files"]))
        self.assertEqual(sum(item["convert_to_bf16"] for item in lock["files"]), 4)
        self.assertIn("ltxv-2b-0.9.8-distilled.safetensors", [item["path"] for item in lock["files"]])
        self.assertTrue((installer.SOURCE / "docs" / "licenses" / "LTX-Video-Open-Weights-License-0.X.txt").is_file())
        root = Path(os.environ["TEMP"]) / "ltx-not-installed"
        self.assertEqual(installer.ltx_needed(root, "neodragon", False, lambda r: ("xpu", "xpu")), ["models/ltx-video-2b-0.9.8"])
        self.assertEqual(installer.ltx_needed(root, "neodragon", False, lambda r: ("cuda", "cuda128")), [])
        self.assertEqual(installer.ltx_needed(root, "neodragon", True, lambda r: ("xpu", "xpu")), [])

    def test_distilled_schedule_is_applied_exactly(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed")
        from local_video import ltx, ltx_stage
        scheduler = type("Scheduler", (), {})()
        ltx_stage.fixed_schedule(scheduler, ltx.TIMESTEPS, torch)
        scheduler.set_timesteps(8, torch.device("cpu"), sigmas=[0.5], mu=1.2)
        self.assertEqual(scheduler.num_inference_steps, 8)
        self.assertTrue(torch.allclose(scheduler.timesteps, torch.tensor(ltx.TIMESTEPS, dtype=torch.float32)))
        self.assertEqual(float(scheduler.sigmas[-1]), 0.0)


class InstallTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "skyreels-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = json.loads((installer.SOURCE / skyreels.LOCK).read_text(encoding="utf-8"))

    def test_lock_pins_every_file_and_marks_fp32_shards_for_bf16(self):
        self.assertEqual(len(self.lock["revision"]), 40)
        self.assertTrue(all(len(item["sha256"]) == 64 and item["size"] > 0 for item in self.lock["files"]))
        converted = [item["path"] for item in self.lock["files"] if item["convert_to_bf16"]]
        self.assertEqual(len(converted), 7)
        self.assertTrue(all(path.endswith(".safetensors") and path.split("/")[0] in {"text_encoder", "transformer"} for path in converted))
        self.assertTrue((installer.SOURCE / "docs" / "licenses" / "SkyReels-V2-Skywork-LICENSE.txt").is_file())

    def test_only_cuda_computers_download_the_long_video_model(self):
        needed = lambda backend: installer.skyreels_needed(self.root, "neodragon", False, lambda root: (backend, None))
        self.assertEqual(needed("cuda"), ["models/skyreels-v2-df-1.3b"])
        self.assertEqual(needed("xpu"), [])
        self.assertEqual(needed(None), [])
        self.assertEqual(installer.skyreels_needed(self.root, "neodragon", True, lambda root: ("cuda", None)), [])

    def test_install_verifies_converts_and_resumes(self):
        downloads, conversions = [], []
        def fake_download(url, destination, expected_sha, size):
            downloads.append(destination.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"{}" if destination.suffix == ".json" else b"x")
        with mock.patch.object(installer, "download", side_effect=fake_download), \
             mock.patch.object(installer.shutil, "disk_usage", return_value=mock.Mock(free=100 * GIB)):
            installer.install_skyreels(self.root, convert=lambda path: conversions.append(path.name))
            self.assertEqual(len(downloads), len(self.lock["files"]))
            self.assertEqual(len(conversions), 7)
            self.assertTrue(skyreels.installed(self.root, installer.SOURCE))
            downloads.clear()
            installer.install_skyreels(self.root, convert=lambda path: conversions.append(path.name))
            self.assertEqual(downloads, [], "a finished install must not download again")

    def test_not_enough_disk_is_reported_before_downloading(self):
        with mock.patch.object(installer.shutil, "disk_usage", return_value=mock.Mock(free=5 * GIB)), \
             mock.patch.object(installer, "download") as download:
            with self.assertRaises(ValueError):
                installer.install_skyreels(self.root, convert=lambda path: None)
        download.assert_not_called()


class ConversionTests(unittest.TestCase):
    def test_fp32_shard_becomes_bf16_with_same_names_and_index_total(self):
        try:
            import torch
            from safetensors.torch import load_file, save_file
        except ImportError:
            self.skipTest("torch and safetensors are needed")
        parent = Path(os.environ["TEMP"]) / "skyreels-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            shard = Path(folder) / "model-00001-of-00001.safetensors"
            original = {"weight": torch.randn(64, 32), "steps": torch.arange(5)}
            save_file(original, str(shard))
            before = shard.stat().st_size
            skyreels.convert_to_bf16(shard)
            converted = load_file(str(shard))
            self.assertEqual(converted["weight"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(converted["weight"], original["weight"].to(torch.bfloat16)))
            self.assertTrue(torch.equal(converted["steps"], original["steps"]))
            self.assertLess(shard.stat().st_size, before)
            index = Path(folder) / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": {"weight": shard.name, "steps": shard.name}}), encoding="utf-8")
            skyreels.fix_index_total_size(index)
            self.assertEqual(json.loads(index.read_text(encoding="utf-8"))["metadata"]["total_size"], shard.stat().st_size // 2 * 2)


if __name__ == "__main__":
    unittest.main()

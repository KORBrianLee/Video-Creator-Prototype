"""Continuity requests, composition cache reuse, and jump signals."""
import unittest
import tempfile
import hashlib
from unittest.mock import patch
from pathlib import Path

from local_video import control, neodragon, site_profiles, temporal_quality


class SiteTests(unittest.TestCase):
    def normalize(self, value):
        with patch.object(control, "load_config", return_value=({
            "backend": "cpu", "model_profile": "neodragon", "threads": 4,
            "reserve_ram_gib": 1.0, "minimum_free_ram_gib": 3.0,
            "maximum_working_set_gib": 5.0}, Path(r"D:\CursorVideoLocal"))), \
             patch.object(control, "manifest", return_value={"files": []}):
            return control.normalize(value)

    def test_preset_expands_locally_without_training_claim(self):
        request = self.normalize({"site_template": "excavator_spotter", "script": "굴착기와 작업자"})
        self.assertEqual(request["script"], "굴착기와 작업자")
        self.assertTrue(request["continuous"])
        self.assertEqual(neodragon.settings_for(request)["frames"], 49)
        self.assertEqual(len(request["scenes"]), 1)
        self.assertFalse(site_profiles.catalog()["weights_finetuned"])

    def test_continuous_is_not_assembled_from_multiple_scenes(self):
        value = {"continuous": True, "scenes": [{"prompt": "excavator"}, {"prompt": "workers"}]}
        with self.assertRaises(ValueError):
            self.normalize(value)
        with self.assertRaises(ValueError):
            neodragon.settings_for({**value, "preset": "preview", "duration_seconds": 4})

    def test_lengths_have_causal_frame_count_without_fps_slowdown(self):
        for seconds, frames in [(2, 49), (4, 97), (8, 193), (10, 241), (15, 361)]:
            request = self.normalize({"site_template": "loader_guidance", "duration_seconds": seconds})
            settings = neodragon.settings_for(request)
            self.assertEqual((settings["frames"], settings["fps"]), (frames, 24))
            # Neo's temporal position limit is 200 latent frames (8 video frames each).
            self.assertLess((frames - 1) // 8 + 1, 200)
            self.assertEqual((frames-1) % 8, 0)
            self.assertEqual(request["maximum_working_set_gib"], 5.0)

    def test_stage_time_limit_grows_with_clip_length(self):
        self.assertEqual(neodragon.stage_time_limit({"continuous": False}), 1800)
        self.assertEqual(neodragon.stage_time_limit({"continuous": True, "duration_seconds": 8}), 3860)
        self.assertEqual(neodragon.stage_time_limit({"continuous": True, "duration_seconds": 15}), 7220)
        self.assertEqual(neodragon.stage_time_limit({"maximum_scene_seconds": 60, "continuous": True, "duration_seconds": 15}), 60)
        self.assertEqual(neodragon.stage_time_limit({"threads": 4}), 1800)

    def test_invalid_lengths_and_conflicting_input_rejected(self):
        for extra in [{"duration_seconds": True}, {"duration_seconds": 16}, {"continuous": "true"},
                      {"preset": "smoke"}, {"scenes": [{"prompt": "custom action"}]}]:
            with self.assertRaises(ValueError):
                self.normalize({"site_template": "excavator_spotter", **extra})
        with self.assertRaises(ValueError):
            self.normalize({"continuous": False, "duration_seconds": 4, "scenes": [{"prompt": "workers"}]})
        with self.assertRaises(ValueError):
            self.normalize({"site_template": "unknown"})

    def test_composition_cache_reused_between_motion_variants(self):
        a = self.normalize({"site_template": "excavator_spotter"})["scenes"][0]
        b = self.normalize({"site_template": "excavator_loading"})["scenes"][0]
        settings = neodragon.PRESETS["preview"]
        self.assertNotEqual(a["prompt"], b["prompt"])
        key = lambda scene: neodragon.first_frame_key(scene, settings, "cpu-hash", "model-hash")
        self.assertEqual(key(a), key(b))
        for field, value in [("first_frame_prompt", "different machine"), ("negative_prompt", "new exclusions"), ("seed", 43)]:
            changed = {**b, field: value}
            self.assertNotEqual(key(a), key(changed))

    def test_first_frame_prompt_is_validated(self):
        for prompt in ["", " ", None, 123, "a"*3001]:
            with self.assertRaises(ValueError):
                self.normalize({"scenes": [{"prompt": "machine moves", "first_frame_prompt": prompt}]})

    def test_quality_uses_mapped_originals_with_same_resource_cap(self):
        request = self.normalize({"site_template": "excavator_spotter", "preset": "quality"})
        self.assertEqual(request["cpu_precision"], "bf16_stream")
        self.assertEqual(request["maximum_working_set_gib"], 5.0)
        with patch.object(neodragon.media, "_memory", side_effect=AssertionError("No conversion subprocess needed")):
            result = neodragon.run_stage(Path(r"D:\CursorVideoLocal"), Path(r"D:\CursorVideoLocal\tmp"),
                "video_pack", request, lambda update: None, lambda: False)
        self.assertTrue(result["not_required_original_mapped_weights"])
        with self.assertRaises(ValueError):
            self.normalize({"scenes": [{"prompt": "workers"}], "cpu_precision": "cuda"})

    def test_temporal_jump_and_hold_signals(self):
        ordinary = [0.5]*96
        self.assertEqual(temporal_quality.analyze(ordinary)["state"], "no_large_jump_hold_or_darkening_detected")
        cut = ordinary[:]
        cut[48] = 40
        self.assertEqual(temporal_quality.analyze(cut)["possible_jump_frames"], [49])
        frozen = [0.5]*10+[0.0]*15+[0.5]*20
        result = temporal_quality.analyze(frozen)
        self.assertEqual(result["possible_hold_runs"], [{"first_frame": 10, "last_frame": 25, "frame_pairs": 15}])
        self.assertEqual(temporal_quality.analyze([])["state"], "insufficient_frames")

    def test_gradual_blackout_is_flagged_without_a_cut(self):
        levels = [100-i for i in range(97)]
        result = temporal_quality.analyze([1.0]*96, levels)
        self.assertEqual(result["possible_jump_frames"], [])
        self.assertEqual(result["state"], "review_required")
        self.assertLess(result["possible_darkening"]["end_start_ratio"], 0.55)
        self.assertIsNone(temporal_quality.analyze([1.0]*96, [80]*97)["possible_darkening"])

    def test_modified_licensed_reference_rejected_before_model_loading(self):
        parent = Path(tempfile.gettempdir()) / "site-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            image = root / "reference.png"
            image.write_bytes(b"original")
            profile = {"ref": {"prompt": "worker moves", "first_frame_prompt": "worker",
                "reference": {"relative_path": "reference.png", "sha256": hashlib.sha256(b"original").hexdigest()}}}
            with patch.object(site_profiles, "PROFILES", profile):
                self.assertEqual(site_profiles.expand({"site_template": "ref"}, root)["scenes"][0]["image_path"], str(image))
                image.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "변경"):
                    site_profiles.expand({"site_template": "ref"}, root)

    def test_reference_license_and_source_identity_are_pinned(self):
        from tools.prepare_site_data import validate_info
        item = {"id": "test", "license": "CC BY 4.0", "sha1": "abc", "bytes": 100}
        info = {"sha1": "abc", "size": 100, "url": "https://upload.wikimedia.org/original.webm",
            "extmetadata": {"LicenseShortName": {"value": "CC BY 4.0"}, "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"}}}
        self.assertEqual(validate_info(item, info), info["url"])
        for changed in [{"sha1": "changed"}, {"size": 101}, {"url": "https://unverified.example/video"},
                        {"extmetadata": {**info["extmetadata"], "LicenseShortName": {"value": "CC BY-NC 4.0"}}}]:
            with self.assertRaises(ValueError):
                validate_info(item, {**info, **changed})

    def test_failed_construction_quality_cannot_reuse_a_production_job(self):
        request = {"domain": "construction", "diagnostic": False}
        with patch.object(control, "construction_validation", return_value={"production_ready": False}):
            with self.assertRaisesRegex(RuntimeError, "중장비"):
                control.check_domain_request(request, Path(r"D:\CursorVideoLocal"))
            control.check_domain_request({**request, "diagnostic": True}, Path(r"D:\CursorVideoLocal"))
            control.check_domain_request({"domain": "general", "diagnostic": False}, Path(r"D:\CursorVideoLocal"))


if __name__ == "__main__":
    unittest.main()

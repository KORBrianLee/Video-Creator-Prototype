"""Rebuilds must preserve other Cursor servers and the user's resource config."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import installer


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "deployment-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "README.md").write_text("updated release", encoding="utf-8")
        self.app = self.root / "app"
        (self.app / ".cursor").mkdir(parents=True)

    def test_redeploy_preserves_other_servers_and_resource_preferences(self):
        path = self.app / ".cursor" / "mcp.json"
        other = {"command": "some-other-server", "args": ["keep"]}
        path.write_text(json.dumps({"mcpServers": {"other": other}, "custom": True}), encoding="utf-8")
        resource = self.app / "video.config.json"
        resource.write_text(json.dumps({"runtime_dir": str(self.root), "threads": 2}), encoding="utf-8")
        with patch.object(installer, "SOURCE", self.source):
            installer.deploy(self.root)
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["mcpServers"]["other"], other)
        self.assertTrue(result["custom"])
        self.assertIn("local-video", result["mcpServers"])
        self.assertEqual(json.loads(resource.read_text(encoding="utf-8"))["threads"], 2)

    def test_default_install_requires_vulkan_engine_and_update_fetches_it_when_missing(self):
        self.assertEqual(installer.missing_engines(self.root, True), ["engines/vulkan"])
        self.assertEqual(installer.missing_engines(self.root, False), [])
        engine = self.root / "engines" / "vulkan"
        engine.mkdir(parents=True)
        (engine / "sd-cli.exe").write_bytes(b"x")
        self.assertEqual(installer.missing_engines(self.root, True), [])

    def test_every_torch_gpu_lock_is_hash_pinned_and_matches_its_backend(self):
        expected = {"xpu": "+xpu", "cuda128": "+cu128", "cuda126": "+cu126"}
        for key, suffix in expected.items():
            lock = json.loads((installer.SOURCE / installer.TORCH_LOCKS[key]).read_text(encoding="utf-8"))
            torch = next(p for p in lock["packages"] if p["name"] == "torch")
            self.assertTrue(torch["version"].endswith(suffix), key)
            self.assertTrue(all(len(p["sha256"]) == 64 for p in lock["packages"]), key)
            self.assertTrue(lock["index_url"].startswith("https://download.pytorch.org/whl/"), key)

    def test_the_downloaded_runtime_follows_the_main_gpu_vendor(self):
        self.assertEqual(installer.torch_runtime_for("intel"), ("xpu", "xpu"))
        self.assertEqual(installer.torch_runtime_for("nvidia", 8.9), ("cuda", "cuda128"))
        self.assertEqual(installer.torch_runtime_for("nvidia", 12.0), ("cuda", "cuda128"))
        self.assertEqual(installer.torch_runtime_for("nvidia", 6.1), ("cuda", "cuda126"))
        self.assertEqual(installer.torch_runtime_for("nvidia", None), ("cuda", "cuda128"))
        self.assertEqual(installer.torch_runtime_for("amd"), (None, None))
        self.assertEqual(installer.torch_runtime_for("other"), (None, None))

    def test_torch_runtime_is_only_requested_for_a_matching_gpu_and_tracks_its_lock(self):
        intel = lambda root: ("xpu", "xpu")
        nvidia = lambda root: ("cuda", "cuda128")
        nothing = lambda root: (None, None)
        needed = lambda detect, profile="neodragon", skip=False: installer.torch_runtime_needed(self.root, profile, skip, detect)
        self.assertEqual(needed(intel), ["runtime/xpu-site"])
        self.assertEqual(needed(nvidia), ["runtime/cuda-site"])
        self.assertEqual(needed(nothing), [])
        self.assertEqual(needed(intel, skip=True), [])
        self.assertEqual(needed(intel, profile="lightning"), [])
        folder = self.root / "runtime" / "xpu-site"
        folder.mkdir(parents=True)
        (folder / "install-record.json").write_text(json.dumps({"lock": "old"}), encoding="utf-8")
        self.assertEqual(needed(intel), ["runtime/xpu-site"])
        current = installer.lock_digest(installer.SOURCE / installer.TORCH_LOCKS["xpu"])
        (folder / "install-record.json").write_text(json.dumps({"lock": current}), encoding="utf-8")
        self.assertEqual(needed(intel), [])
        self.assertEqual(needed(nvidia), ["runtime/cuda-site"])

    def test_vulkan_is_default_and_can_be_skipped_only_explicitly(self):
        self.assertTrue(installer.wants_vulkan("neodragon", False))
        self.assertFalse(installer.wants_vulkan("neodragon", True))
        self.assertTrue(installer.wants_vulkan("wan", True))

    def test_malformed_existing_mcp_config_is_not_overwritten(self):
        path = self.app / ".cursor" / "mcp.json"
        path.write_text("[]", encoding="utf-8")
        with patch.object(installer, "SOURCE", self.source), self.assertRaisesRegex(ValueError, "덮어쓰지"):
            installer.deploy(self.root)
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")

    def test_portable_copy_rebinds_selected_ssd_and_keeps_resource_limits(self):
        cfg = {"runtime_dir": r"D:\CursorVideoLocal", "threads": 2,
               "backend": "intel-gpu", "maximum_working_set_gib": 4.5}
        path = self.app / "video.config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with patch.object(installer, "SOURCE", self.source):
            installer.deploy(self.root)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["runtime_dir"], str(self.root))
        self.assertEqual(saved["threads"], 2)
        self.assertEqual(saved["backend"], "intel-gpu")
        self.assertEqual(saved["maximum_working_set_gib"], 4.5)
        connection = json.loads((self.app / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(connection["mcpServers"]["local-video"]["env"]["CVL_RUNTIME_DIR"], str(self.root))

    def test_old_fixed_defaults_become_adaptive_and_source_config_never_overwrites(self):
        (self.source / "video.config.json").write_text(json.dumps({"backend": "auto", "threads": 4}), encoding="utf-8")
        cfg = {"runtime_dir": str(self.root), "backend": "cpu", "threads": 4, "minimum_free_ram_gib": 3.0,
               "reserve_ram_gib": 1.5, "maximum_working_set_gib": 5.0}
        path = self.app / "video.config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with patch.object(installer, "SOURCE", self.source):
            installer.deploy(self.root)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["backend"], "cpu")
        for key in ["threads", "minimum_free_ram_gib", "reserve_ram_gib", "maximum_working_set_gib"]:
            self.assertEqual(saved[key], "auto")


if __name__ == "__main__":
    unittest.main()

"""제어기 검증. 추론을 실행하지 않으며 임시 파일은 D 드라이브만 사용한다."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import control


class ControllerTests(unittest.TestCase):
    """모의 작업과 모의 추론기를 사용해 상태 전이·자원 보호를 검사한다."""

    @classmethod
    def setUpClass(cls):
        from local_video.storage import runtime_root, default_runtime
        parent = runtime_root(os.environ.get("CVL_TEST_DIR", str(Path(default_runtime()) / "tmp" / "control-tests")))
        parent.mkdir(parents=True, exist_ok=True)
        cls.parent = parent

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="isolated-", dir=self.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.app = self.root / "app"
        self.app.mkdir()
        self.cfg = {"runtime_dir": str(self.root), "backend": "cpu", "model_profile": "lightning", "threads": 4,
                    "minimum_free_ram_gib": 4.0, "reserve_ram_gib": 1.0,
                    "maximum_working_set_gib": 6.0}
        self.lock = {"schema_version": 1, "model_id": "test-only-lightning",
                     "model_license": "CreativeML-OpenRAIL-M", "files": [],
                     "engine": {"commit": "test-only"}}
        self.save_config()
        self.save_lock()
        self.enterContext(patch.object(control, "APP", self.app))
        self.enterContext(patch.dict(os.environ, {"CVL_RUNTIME_DIR": "", "TEMP": str(self.root),
                                                "TMP": str(self.root), "TMPDIR": str(self.root)}))
        self.request = {"title": "검증 요청", "script": "파도가 2초간 움직인다.", "diagnostic": True,
                        "preset": "preview", "seed": 42,
                        "scenes": [{"id": "sea", "prompt": "Ocean waves roll onto a shore."}]}

    def save_config(self):
        control.write_json(self.app / "video.config.json", self.cfg)

    def save_lock(self):
        control.write_json(self.app / "lightning.lock.json", self.lock)
        wan = {**self.lock, "model_id": "test-only-wan", "model_license": "Apache-2.0"}
        control.write_json(self.app / "models.lock.json", wan)

    def make_job(self, state="queued", request=None, result=None, digest=None, jid="0123456789ab"):
        folder = self.root / "jobs" / jid
        folder.mkdir(parents=True, exist_ok=True)
        control.write_json(folder / "request.json", request or control.normalize(self.request))
        saved = {"job_id": jid, "state": state, "request_hash": digest or "test-only", "created_at": 0}
        if result is not None:
            saved["result"] = result
        control.write_json(folder / "status.json", saved)
        return jid, folder

    def digest(self, request=None):
        normalized = control.normalize(request or self.request)
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def fake_lock(self, acquired=True):
        record = types.SimpleNamespace(acquires=0, releases=0)

        class Lock:
            def __init__(self, path):
                record.path = path

            def acquire(self):
                record.acquires += 1
                return acquired

            def release(self):
                record.releases += 1

        self.enterContext(patch.object(control, "EngineLock", Lock))
        return record

    def fake_backend(self, callback):
        module = types.ModuleType("local_video.backend")
        module.generate_project = callback
        self.enterContext(patch.dict(sys.modules, {"local_video.backend": module}))

    def make_junction(self, link, target):
        """D의 격리 폴더 안에서만 연결을 만들고 해당 연결만 제거한다."""
        boundary = self.root.resolve()
        if not link.parent.resolve().is_relative_to(boundary) or not target.resolve().is_relative_to(boundary):
            raise RuntimeError("검증용 연결과 대상은 격리 테스트 폴더 안이어야 합니다.")
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True, errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        def remove_only_junction():
            if link.exists():
                if link.resolve() != target.resolve() or not link.parent.resolve().is_relative_to(boundary):
                    raise RuntimeError("검증용 연결의 위치나 대상이 변경됐습니다.")
                os.rmdir(link)

        self.addCleanup(remove_only_junction)
        return link

    def test_normalize_preserves_korean_script_and_stable_scene_seed(self):
        self.request["scenes"].append({"prompt": "Water recedes from the wet sand."})
        result = control.normalize(self.request)
        self.assertEqual(result["script"], self.request["script"])
        self.assertEqual([s["seed"] for s in result["scenes"]], [42, 43])
        self.assertEqual(result["scenes"][1]["id"], "s02")
        self.assertEqual(result["_runtime_dir"], str(self.root.resolve()))
        self.assertEqual(result["model_profile"], "lightning")
        self.assertEqual(result["_model_spec"]["model_license"], "CreativeML-OpenRAIL-M")

    def test_explicit_wan_profile_selects_its_manifest_and_memory_policy(self):
        request = {**self.request, "model_profile": "wan"}
        result = control.normalize(request)
        self.assertEqual(result["model_profile"], "wan")
        self.assertEqual(result["_model_spec"]["model_license"], "Apache-2.0")
        self.assertEqual(result["maximum_working_set_gib"], self.cfg["maximum_working_set_gib"])
        self.assertEqual(result["minimum_free_ram_gib"], 8.0)
        self.assertNotEqual(self.digest(request), self.digest())
        with self.assertRaises(ValueError):
            control.normalize({**self.request, "model_profile": "unsupported"})

    def test_anchor_end_is_optional_and_bounded(self):
        self.assertNotIn("anchor_end", control.normalize(self.request))
        self.assertEqual(control.normalize({**self.request, "anchor_end": 0.5})["anchor_end"], 0.5)
        for bad in (0, 1.5, True, "0.5"):
            with self.assertRaises(ValueError):
                control.normalize({**self.request, "anchor_end": bad})

    def test_anchor_end_image_needs_anchor_and_a_runtime_path(self):
        image = self.root / "end.png"
        image.write_bytes(b"png")
        result = control.normalize({**self.request, "anchor_end": 0.8, "anchor_end_image": str(image)})
        self.assertEqual(result["anchor_end_image"]["path"], str(image.resolve()))
        self.assertEqual(len(result["anchor_end_image"]["sha256"]), 64)
        with self.assertRaises(ValueError):
            control.normalize({**self.request, "anchor_end_image": str(image)})
        with self.assertRaises(ValueError):
            control.normalize({**self.request, "anchor_end": 0.8, "anchor_end_image": "relative.png"})

    def test_middle_keyframes_are_checked_and_hashed(self):
        image = self.root / "mid.png"
        image.write_bytes(b"png")
        result = control.normalize({**self.request, "anchor_end": 0.9,
                                    "keyframes": [{"image_path": str(image), "at": 0.5}]})
        self.assertEqual(result["keyframes"][0]["at"], 0.5)
        self.assertEqual(result["keyframes"][0]["strength"], 0.9)
        for bad in ([{"image_path": str(image), "at": 1.0}], [{"at": 0.5}], [{"image_path": "x.png", "at": 0.5}]):
            with self.assertRaises(ValueError):
                control.normalize({**self.request, "anchor_end": 0.9, "keyframes": bad})
        with self.assertRaises(ValueError):
            control.normalize({**self.request, "keyframes": [{"image_path": str(image), "at": 0.5}]})

    def test_normalize_rejects_invalid_inputs_and_scene_path_escape(self):
        cases = [None, [], {**self.request, "seed": True}, {**self.request, "seed": -1},
                 {**self.request, "preset": "unknown"}, {**self.request, "scenes": []},
                 {**self.request, "scenes": [None]},
                 {**self.request, "scenes": [{"id": "../escape", "prompt": "waves"}]},
                 {**self.request, "scenes": [{"id": "D:\\outside", "prompt": "waves"}]},
                 {**self.request, "scenes": [{"prompt": "   "}]},
                 {**self.request, "scenes": [{"prompt": "waves", "seed": False}]},
                 {**self.request, "scenes": [{"id": "same", "prompt": "a"}, {"id": "same", "prompt": "b"}]}]
        for invalid in cases:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                control.normalize(invalid)

    def test_invalid_job_identifier_never_resolves_a_path(self):
        for invalid in [None, "", "../outside", "D:\\outside", "0123456789AB", "a" * 13]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                control.job_path(invalid)

    def test_storage_root_rejects_other_drives_root_relative_and_parent_segments(self):
        for value in ["C:/", "D:/", r"D:relative", r"D:\safe\..\outside", r"\\server\share\video", "C:/safe/../outside"]:
            self.cfg["runtime_dir"] = value
            self.save_config()
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.load_config()

    def test_runtime_environment_override_remains_on_d_drive(self):
        with patch.dict(os.environ, {"CVL_RUNTIME_DIR": str(self.root / "portable")}):
            cfg, root = control.load_config()
            self.assertEqual(root, (self.root / "portable").resolve())
            self.assertEqual(cfg["runtime_dir"], str(self.root / "portable"))
        with patch.dict(os.environ, {"CVL_RUNTIME_DIR": "C:/"}), self.assertRaises(ValueError):
            control.load_config()

    def test_c_ssd_is_accepted_and_runtime_paths_remain_contained(self):
        self.cfg["runtime_dir"] = str(Path(os.environ["USERPROFILE"]) / "Documents" / "Codex" / "CursorVideoRuntime")
        self.save_config()
        cfg, root = control.load_config()
        self.assertEqual(root.drive.upper(), "C:")
        with self.assertRaises(ValueError):
            control.runtime_path(root, "..", "outside")

    @unittest.skipUnless(os.name == "nt", "Windows junction 경계 검증")
    def test_runtime_path_rejects_junction_outside_selected_runtime(self):
        runtime = self.root / "selected-runtime"
        outside = self.root / "outside-runtime"
        runtime.mkdir()
        outside.mkdir()
        sentinel = outside / "sentinel.marker"
        sentinel.write_bytes(b"test-only-external-target")
        self.make_junction(runtime / "tmp", outside)
        with self.assertRaises(ValueError):
            control.runtime_path(runtime, "tmp", "sentinel.marker")
        self.assertEqual(sentinel.read_bytes(), b"test-only-external-target")
        self.assertEqual(control.runtime_path(runtime, "jobs", "new-job"), runtime / "jobs" / "new-job")

    @unittest.skipUnless(os.name == "nt", "Windows junction 결과 경계 검증")
    def test_completed_result_cannot_be_reused_from_outside_runtime_junction(self):
        runtime = self.root / "selected-runtime"
        outside = self.root / "outside-runtime"
        runtime.mkdir()
        outside.mkdir()
        marker = outside / "simulated-result.marker"
        marker.write_bytes(b"test-only-external-result")
        self.make_junction(runtime / "output", outside)
        state = {"result": {"video_path": str(runtime / "output" / marker.name),
                            "video_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}}
        self.assertFalse(control.completed_result_intact(state, runtime))

    def test_invalid_memory_policy_cannot_disable_resource_protection(self):
        for name in ["minimum_free_ram_gib", "reserve_ram_gib", "maximum_working_set_gib"]:
            original = self.cfg[name]
            for value in [-1, 0, True, "bad", float("nan"), float("inf")]:
                self.cfg[name] = value
                self.save_config()
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    control.load_config()
            self.cfg[name] = original
        self.cfg["maximum_working_set_gib"] = self.cfg["reserve_ram_gib"]
        self.save_config()
        with self.assertRaises(ValueError):
            control.load_config()

    def test_optional_model_absence_reports_recovery_in_selected_runtime(self):
        self.lock["files"] = [{"role": "motion", "filename": "optional.safetensors", "size": 4,
                               "sha256": hashlib.sha256(b"test").hexdigest()}]
        self.save_lock()
        files = control.check_model_files(self.root, "lightning", self.lock, verify=True)
        available = control.profile_availability(self.root, "lightning", files)
        self.assertFalse(available["weights_present"])
        self.assertEqual(available["recovery"]["setup_arguments"], ["-RuntimeDir", str(self.root), "-Profile", "lightning"])
        self.assertFalse(available["recovery"]["automatic_download"])
        self.assertFalse((self.root / "jobs").exists())
        model = self.root / "models" / "optional.safetensors"
        model.parent.mkdir()
        model.write_bytes(b"fail")  # Right size but wrong hash cannot pass verify.
        self.assertEqual(control.check_model_files(self.root, "lightning", self.lock, True)[0]["integrity"], "sha256_mismatch")
        model.write_bytes(b"test")
        checked = control.check_model_files(self.root, "lightning", self.lock, True)
        self.assertTrue(control.profile_availability(self.root, "lightning", checked)["weights_present"])
        self.assertEqual(checked[0]["integrity"], "sha256_verified")

    def test_missing_optional_weights_block_job_before_spawn(self):
        self.lock["files"] = [{"role": "motion", "filename": "removed.safetensors", "size": 4, "sha256": "0" * 64}]
        self.save_lock()
        with patch.object(control.subprocess, "Popen") as spawn, self.assertRaisesRegex(RuntimeError, "removed.safetensors"):
            control.submit(self.request)
        spawn.assert_not_called()
        self.assertFalse((self.root / "jobs").exists())

    def test_low_free_ram_blocks_submit_without_creating_a_job(self):
        with patch.object(control, "doctor", return_value={"ready": False, "errors": ["RAM 부족"]}), \
                patch.object(control.subprocess, "Popen") as spawn, self.assertRaises(RuntimeError):
            control.submit(self.request)
        spawn.assert_not_called()
        self.assertFalse((self.root / "jobs").exists())

    def test_wan_runs_in_low_memory_mode_below_its_recommended_free_ram(self):
        engine = self.root / "engines" / "cpu" / "sd-cli.exe"
        engine.parent.mkdir(parents=True)
        engine.write_bytes(b"test-only-marker-never-executed")
        with patch.object(control, "memory", return_value={"total_ram_gib": 16.0, "free_ram_gib": 7.0}), \
                patch.object(control.importlib.util, "find_spec", return_value=object()), \
                patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 * 2**30)):
            lightning = control.doctor(profile="lightning")
            wan = control.doctor(profile="wan")
        self.assertTrue(lightning["ready"])
        self.assertFalse(lightning["ready_for_generation"])
        self.assertEqual(lightning["quality_status"], "failed_visual_validation")
        # Elastic memory: 7 GiB free is below Wan's recommendation, so it runs slower, not refused.
        self.assertTrue(wan["ready"])
        self.assertGreaterEqual(wan["minimum_free_ram_gib"], 8.0)
        self.assertTrue(wan["resource_plan"]["low_memory_mode"])
        self.assertFalse(lightning["resource_plan"]["low_memory_mode"])
        self.assertEqual(wan["model_license"], "Apache-2.0")

    def test_request_hash_changes_when_generation_settings_change(self):
        baseline = self.digest()
        for key, value in [("seed", 43), ("preset", "quality")]:
            changed = copy.deepcopy(self.request)
            changed[key] = value
            self.assertNotEqual(baseline, self.digest(changed))
        changed = copy.deepcopy(self.request)
        changed["scenes"][0]["prompt"] += " Strong wind."
        self.assertNotEqual(baseline, self.digest(changed))
        self.lock["engine"]["commit"] = "new-test-revision"
        self.save_lock()
        self.assertNotEqual(baseline, self.digest())

    def test_repeated_running_request_reuses_job_without_spawning(self):
        jid, folder = self.make_job(state="running", digest=self.digest())
        with patch.object(control, "doctor") as doctor, patch.object(control.subprocess, "Popen") as spawn:
            result = control.submit(self.request)
        self.assertEqual(result["job_id"], jid)
        self.assertTrue(result["reused_existing_job"])
        spawn.assert_not_called()
        doctor.assert_not_called()
        self.assertEqual(len(list((self.root / "jobs").iterdir())), 1)

    def test_completed_request_reuses_only_existing_result_with_valid_sha256(self):
        marker = self.root / "simulated-result.marker"
        marker.write_text("모의 결과이며 영상이 아님", encoding="utf-8")
        self.make_job(state="completed", result={"video_path": str(marker),
                      "video_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}, digest=self.digest())
        with patch.object(control.subprocess, "Popen") as spawn:
            result = control.submit(self.request)
        self.assertTrue(result["reused_existing_job"])
        spawn.assert_not_called()
        marker.unlink()
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), \
                patch.object(control.subprocess, "Popen", return_value=types.SimpleNamespace(pid=100)) as spawn:
            result = control.submit(self.request)
        self.assertFalse(result.get("reused_existing_job", False))
        self.assertEqual(spawn.call_count, 1)

    def test_repeat_request_index_avoids_scanning_the_job_history(self):
        jid, _ = self.make_job(state="running", digest=self.digest())
        with patch.object(control.subprocess, "Popen") as spawn:
            self.assertEqual(control.submit(self.request)["job_id"], jid)
            with patch.object(Path, "glob", side_effect=AssertionError("Indexed reuse must not scan history")) as scan:
                repeated = control.submit(self.request)
        self.assertEqual(repeated["job_id"], jid)
        self.assertTrue(repeated["reused_existing_job"])
        spawn.assert_not_called()
        scan.assert_not_called()

    def test_wrong_request_index_cannot_reuse_a_different_job(self):
        digest = self.digest()
        expected, _ = self.make_job(state="running", digest=digest)
        other, _ = self.make_job(state="running", digest="different-request", jid="abcdef012345")
        control._remember_request(self.root, digest, other)
        with patch.object(control.subprocess, "Popen") as spawn:
            reused = control.submit(self.request)
        self.assertEqual(reused["job_id"], expected)
        self.assertEqual(control._indexed_request(self.root, digest)["job_id"], expected)
        spawn.assert_not_called()

    def test_missing_malformed_and_path_escape_indices_fall_back_to_the_real_job(self):
        digest = self.digest()
        jid, _ = self.make_job(state="running", digest=digest)
        path = self.root / "cache" / "request-index" / (digest + ".json")
        for hint in [[], {"request_hash": digest, "job_id": "../outside"},
                     {"request_hash": digest, "job_id": "ffffffffffff"},
                     {"request_hash": "wrong", "job_id": jid}]:
            control.write_json(path, hint)
            with self.subTest(hint=hint), patch.object(control.subprocess, "Popen") as spawn:
                self.assertEqual(control.submit(self.request)["job_id"], jid)
                spawn.assert_not_called()

    def test_index_does_not_trust_a_changed_completed_video(self):
        marker = self.root / "simulated-result.marker"
        marker.write_bytes(b"AAAA-result")
        jid, _ = self.make_job(state="completed", digest=self.digest(), result={
            "video_path": str(marker), "video_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()})
        control._remember_request(self.root, self.digest(), jid)
        marker.write_bytes(b"BBBB-result")
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), \
                patch.object(control.subprocess, "Popen", return_value=types.SimpleNamespace(pid=100)) as spawn:
            submitted = control.submit(self.request)
        self.assertNotEqual(submitted["job_id"], jid)
        self.assertFalse(submitted.get("reused_existing_job", False))
        self.assertEqual(spawn.call_count, 1)

    def test_index_cannot_bypass_production_readiness(self):
        request = {**self.request, "diagnostic": False}
        jid, _ = self.make_job(state="running", request=control.normalize(request), digest=self.digest(request))
        control._remember_request(self.root, self.digest(request), jid)
        with patch.object(control, "doctor", return_value={"ready": False, "errors": ["RAM 부족"]}), \
                patch.object(control, "_indexed_request") as lookup, \
                patch.object(control.subprocess, "Popen") as spawn, self.assertRaises(RuntimeError):
            control.submit(request)
        lookup.assert_not_called()
        spawn.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows index junction boundary")
    def test_index_link_outside_runtime_is_not_read_or_written(self):
        runtime = self.root / "selected-runtime"
        outside = self.root / "outside-runtime"
        (runtime / "cache").mkdir(parents=True)
        outside.mkdir()
        self.make_junction(runtime / "cache" / "request-index", outside)
        digest = self.digest()
        control._remember_request(runtime, digest, "0123456789ab")
        self.assertIsNone(control._indexed_request(runtime, digest))
        self.assertEqual(list(outside.iterdir()), [])

    def test_same_size_result_corruption_is_not_reused(self):
        marker = self.root / "simulated-result.marker"
        original = b"AAAA-controller-only-simulated-result"
        marker.write_bytes(original)
        jid, folder = self.make_job(state="completed", result={"video_path": str(marker),
                      "video_sha256": hashlib.sha256(original).hexdigest()}, digest=self.digest())
        marker.write_bytes(b"BBBB" + original[4:])
        self.assertEqual(marker.stat().st_size, len(original))
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), \
                patch.object(control.subprocess, "Popen", return_value=types.SimpleNamespace(pid=100)) as spawn:
            result = control.submit(self.request)
        self.assertFalse(result.get("reused_existing_job", False))
        self.assertNotEqual(result["job_id"], jid)
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse(control.completed_result_intact(control.read_json(folder / "status.json"), self.root))

    def test_completed_result_without_verified_hash_is_not_reused(self):
        marker = self.root / "simulated-result.marker"
        marker.write_bytes(b"test-only-unverified-result")
        for missing_or_invalid in [None, "short", "z" * 64]:
            state = {"result": {"video_path": str(marker), "video_sha256": missing_or_invalid}}
            with self.subTest(sha256=missing_or_invalid):
                self.assertFalse(control.completed_result_intact(state, self.root))

    def test_submit_child_inherits_d_tmp_and_bytecode_disabled(self):
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), \
                patch.object(control.subprocess, "Popen", return_value=types.SimpleNamespace(pid=100)) as spawn:
            result = control.submit(self.request)
        call = spawn.call_args.kwargs
        self.assertEqual(call["cwd"], self.root.resolve())
        self.assertEqual(call["env"]["TEMP"], str(self.root / "tmp"))
        self.assertEqual(call["env"]["TMP"], str(self.root / "tmp"))
        self.assertEqual(call["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(control.status(result["job_id"])["state"], "queued")

    def test_process_start_failure_records_terminal_failure(self):
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), \
                patch.object(control.subprocess, "Popen", side_effect=OSError("모의 시작 실패")), \
                self.assertRaises(OSError):
            control.submit(self.request)
        records = list((self.root / "jobs").glob("*/status.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(control.read_json(records[0])["state"], "failed")

    def test_worker_completion_records_visual_review_required_and_releases_lock(self):
        jid, folder = self.make_job()
        record = self.fake_lock()
        paths = []

        def simulated_backend(request, output, cache, models, progress, cancelled):
            paths.extend([output, cache, models])
            progress({"stage": "simulated-controller-validation"})
            return {"test_only": True}

        self.fake_backend(simulated_backend)
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}):
            control.worker(jid)
        state = control.status(jid)
        self.assertEqual(state["state"], "completed")
        self.assertEqual(state["visual_review"], "required_not_automatically_verified")
        self.assertEqual(record.releases, 1)
        self.assertTrue(all(path.is_relative_to(self.root) for path in paths))

    def test_worker_backend_failure_records_failure_and_releases_lock(self):
        jid, folder = self.make_job()
        record = self.fake_lock()

        def failure(*args):
            raise RuntimeError("모의 추론기 실패")

        self.fake_backend(failure)
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), patch.object(control.sys, "stderr", io.StringIO()):
            control.worker(jid)
        self.assertEqual(control.status(jid)["state"], "failed")
        self.assertEqual(record.releases, 1)

    def test_cancel_before_loading_never_invokes_backend_and_releases_lock(self):
        jid, folder = self.make_job()
        control.cancel(jid)
        record = self.fake_lock()
        calls = []
        self.fake_backend(lambda *args: calls.append(args))
        with patch.object(control.sys, "stderr", io.StringIO()):
            control.worker(jid)
        self.assertEqual(control.status(jid)["state"], "cancelled")
        self.assertEqual(calls, [])
        self.assertEqual(record.releases, 1)

    def test_cancel_while_waiting_for_lock_releases_controller_resources(self):
        jid, folder = self.make_job()
        record = self.fake_lock(acquired=False)

        def cancel_after_poll(seconds):
            (folder / "cancel.request").write_text("cancel", encoding="ascii")

        with patch.object(control.time, "sleep", side_effect=cancel_after_poll), \
                patch.object(control.sys, "stderr", io.StringIO()):
            control.worker(jid)
        self.assertEqual(control.status(jid)["state"], "cancelled")
        self.assertEqual(record.releases, 1)
        self.assertEqual(record.acquires, 2)

    def test_cancel_during_backend_prevents_completion_state(self):
        jid, folder = self.make_job()
        record = self.fake_lock()

        def cancelling_backend(*args):
            (folder / "cancel.request").write_text("cancel", encoding="ascii")
            return {"test_only": True}

        self.fake_backend(cancelling_backend)
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": True}), patch.object(control.sys, "stderr", io.StringIO()):
            control.worker(jid)
        self.assertEqual(control.status(jid)["state"], "cancelled")
        self.assertNotIn("result", control.status(jid))
        self.assertEqual(record.releases, 1)

    def test_resource_recheck_failure_releases_lock_without_loading_backend(self):
        jid, folder = self.make_job()
        record = self.fake_lock()
        calls = []
        self.fake_backend(lambda *args: calls.append(args))
        with patch.object(control, "doctor", return_value={"ready": False, "errors": ["RAM 변화"]}), \
                patch.object(control.sys, "stderr", io.StringIO()):
            control.worker(jid)
        self.assertEqual(control.status(jid)["state"], "failed")
        self.assertEqual(calls, [])
        self.assertEqual(record.releases, 1)

    def test_terminal_cancel_is_idempotent_and_wait_returns_immediately(self):
        jid, folder = self.make_job(state="failed")
        before = (folder / "status.json").read_bytes()
        self.assertEqual(control.cancel(jid)["state"], "failed")
        self.assertFalse((folder / "cancel.request").exists())
        self.assertEqual(before, (folder / "status.json").read_bytes())
        with patch.object(control.time, "sleep") as sleep:
            self.assertEqual(control.wait(jid, 60)["state"], "failed")
        sleep.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows 파일 잠금 검증")
    def test_real_engine_lock_is_exclusive_and_reusable_after_release(self):
        first = control.EngineLock(self.root / "test-engine.lock")
        second = control.EngineLock(self.root / "test-engine.lock")
        self.addCleanup(first.release)
        self.addCleanup(second.release)
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire())
        self.assertIsNone(second.stream)
        first.release()
        self.assertIsNone(first.stream)
        self.assertTrue(second.acquire())
        second.release()
        self.assertIsNone(second.stream)


class WriteJsonTests(unittest.TestCase):
    def test_retries_while_a_reader_holds_the_file(self):
        real_replace, calls = os.replace, []

        def flaky(source, target):
            calls.append(target)
            if len(calls) < 8:
                raise PermissionError("busy")
            real_replace(source, target)

        with tempfile.TemporaryDirectory(dir=os.environ.get("TEMP")) as folder, \
                patch.object(control.os, "replace", flaky), patch.object(control.time, "sleep"):
            target = Path(folder) / "status.json"
            control.write_json(target, {"state": "running"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["state"], "running")
            self.assertEqual(len(calls), 8)

    def test_writes_in_place_when_rename_stays_blocked(self):
        def blocked(source, target):
            raise PermissionError("held by scanner")

        with tempfile.TemporaryDirectory(dir=os.environ.get("TEMP")) as folder, \
                patch.object(control.os, "replace", blocked), patch.object(control.time, "sleep"):
            target = Path(folder) / "status.json"
            target.write_text("{}", encoding="utf-8")
            control.write_json(target, {"state": "completed"})
            self.assertEqual(control.read_json(target)["state"], "completed")
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["status.json"])


class MCPTests(unittest.TestCase):
    """표준 입출력의 응답 형식만 확인하며 영상 생성은 모의 호출한다."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("controller_validation_mcp", SOURCE / "mcp_server.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def exchange(self, messages):
        incoming = io.BytesIO(messages)
        outgoing = io.BytesIO()
        with patch.object(sys, "stdin", types.SimpleNamespace(buffer=incoming)), \
                patch.object(sys, "stdout", types.SimpleNamespace(buffer=outgoing)):
            self.module.main()
        return [json.loads(line) for line in outgoing.getvalue().splitlines()]

    def test_notification_is_silent_and_ping_returns_matching_id(self):
        messages = [dict(jsonrpc="2.0", method="notifications/initialized"),
                    dict(jsonrpc="2.0", id=7, method="ping")]
        result = self.exchange(b"\n".join(json.dumps(x).encode() for x in messages) + b"\n")
        self.assertEqual(result, [{"jsonrpc": "2.0", "id": 7, "result": {}}])

    def test_tool_failure_remains_a_tool_error(self):
        with patch.object(control, "submit", side_effect=RuntimeError("RAM 부족")):
            result = self.module.dispatch("tools/call", {"name": "video_generate", "arguments": {"scenes": [], "diagnostic": True}})
        self.assertTrue(result["isError"])
        self.assertIn("RAM 부족", result["content"][0]["text"])

    def test_compact_completed_site_response_keeps_warnings_attribution_and_evidence(self):
        state = {"job_id": "0123456789ab", "state": "completed", "stage": "complete", "steps": 20,
            "scene_index": 1, "log_path": r"D:\runtime\jobs\0123456789ab\worker.log", "result": {
                "video_path": r"D:\runtime\jobs\0123456789ab\output\video.mp4",
                "attribution_path": r"D:\runtime\jobs\0123456789ab\output\ATTRIBUTION.md",
                "site_template": "rail_lifting_crew", "weights_finetuned": False,
                "settings": {"frames": 49}, "target_iris_generation_verified": False,
                "continuity": {"state": "review_required", "possible_jump_frames": [12],
                    "possible_hold_runs": [{"first_frame": 24, "last_frame": 40}],
                    "possible_darkening": {"end_start_ratio": 0.2},
                    "limitations": "Brightness signals do not certify realism", "jump_threshold": 12},
                "shots": [{"cache_hit": True, "phases": []}]}}
        original = copy.deepcopy(state)
        compacted = self.module.compact(state)
        result = compacted["result"]
        self.assertNotIn("steps", compacted)
        self.assertNotIn("log_path", compacted)
        self.assertFalse(result["production_realism_verified"])
        self.assertFalse(result["weights_finetuned"])
        self.assertFalse(result["target_iris_generation_verified"])
        self.assertIn("초안", result["quality_note"])
        self.assertEqual(result["settings"]["frames"], 49)
        self.assertEqual(result["attribution_path"], original["result"]["attribution_path"])
        self.assertTrue(result["details_path"].endswith("result.json"))
        for key in ["state", "possible_jump_frames", "possible_hold_runs", "possible_darkening"]:
            self.assertEqual(result["continuity"][key], original["result"]["continuity"][key])
        self.assertEqual(state, original)

    def test_compact_live_and_failed_jobs_keep_progress_and_diagnostic_paths(self):
        live = {"job_id": "0123456789ab", "state": "running", "stage": "video_infer",
                "step": 2, "steps": 3, "working_set_gib": 4.0, "free_ram_gib": 6.0,
                "request_path": r"D:\runtime\request.json", "log_path": r"D:\runtime\worker.log"}
        self.assertEqual(self.module.compact(live), live)
        failed = {**live, "state": "failed", "error": "메모리 보호로 중단"}
        self.assertEqual(self.module.compact(failed), failed)

    def test_failed_quality_does_not_submit_a_generation_job(self):
        with patch.object(control, "doctor", return_value={"ready": True, "ready_for_generation": False}), \
                patch.object(control, "submit") as submit:
            result = self.module.dispatch("tools/call", {"name": "video_generate", "arguments": {"scenes": [{"prompt": "Ocean waves roll."}]}})
        submit.assert_not_called()
        self.assertTrue(result["isError"])
        self.assertIn("품질 검증", result["content"][0]["text"])

    def test_unknown_method_uses_method_not_found(self):
        result = self.exchange(b'{"jsonrpc":"2.0","id":3,"method":"unknown"}\n')
        self.assertEqual(result[0]["error"]["code"], -32601)

    def test_modern_request_works_without_initialization(self):
        request = {"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {"_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {}}}}
        result = self.exchange((json.dumps(request)+"\n").encode())
        self.assertEqual(result[0]["result"]["resultType"], "complete")
        self.assertEqual(len(result[0]["result"]["tools"]), 7)

    def test_modern_missing_capabilities_is_rejected(self):
        request = {"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {"_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28"}}}
        result = self.exchange((json.dumps(request)+"\n").encode())
        self.assertEqual(result[0]["error"]["code"], -32602)

    def test_unknown_modern_version_returns_supported_versions(self):
        request = {"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {"_meta": {
            "io.modelcontextprotocol/protocolVersion": "2099-01-01"}}}
        result = self.exchange((json.dumps(request)+"\n").encode())
        self.assertEqual(result[0]["error"]["code"], -32022)
        self.assertIn("2026-07-28", result[0]["error"]["data"]["supported"])

    def test_malformed_json_uses_parse_error(self):
        result = self.exchange(b'{invalid json\n')
        self.assertEqual(result[0]["error"]["code"], -32700)

    def test_non_jsonrpc_request_is_rejected(self):
        result = self.exchange(b'{"jsonrpc":"1.0","id":3,"method":"ping"}\n')
        self.assertEqual(result[0]["error"]["code"], -32600)


if __name__ == "__main__":
    unittest.main(verbosity=2)

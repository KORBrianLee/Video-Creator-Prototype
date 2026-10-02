import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from local_video import accelerator, neodragon

GIB = 2**30


class BackendChoiceTests(unittest.TestCase):
    def test_vendor_decides_backend(self):
        self.assertEqual(accelerator.backend_for_vendor("nvidia"), "cuda")
        self.assertEqual(accelerator.backend_for_vendor("intel"), "xpu")
        self.assertIsNone(accelerator.backend_for_vendor("amd"))
        self.assertIsNone(accelerator.backend_for_vendor("other"))

    def choose(self, device, probe, setting="auto"):
        with mock.patch.object(accelerator, "probe", return_value=probe):
            return accelerator.choose(Path("runtime"), device, setting)

    def test_each_gpu_kind_gets_its_own_device_and_memory_model(self):
        iris = {"vendor": "intel", "shared_memory": True}
        arc = {"vendor": "intel", "shared_memory": False}
        rtx = {"vendor": "nvidia", "shared_memory": False}
        ok = {"ok": True, "device": "gpu", "total_memory_bytes": 8 * GIB}
        self.assertEqual(self.choose(iris, ok)["torch_device"], "xpu")
        self.assertFalse(self.choose(iris, ok)["discrete"])
        self.assertTrue(self.choose(arc, ok)["discrete"])
        chosen = self.choose(rtx, ok)
        self.assertEqual((chosen["torch_device"], chosen["discrete"]), ("cuda", True))

    def test_unsupported_failed_or_disabled_gpus_use_the_cpu_and_opencl_path(self):
        ok = {"ok": True}
        self.assertEqual(self.choose({"vendor": "amd", "shared_memory": False}, ok)["torch_device"], "cpu")
        self.assertIn("amd", self.choose({"vendor": "amd", "shared_memory": False}, ok)["reason"])
        failed = self.choose({"vendor": "nvidia", "shared_memory": False}, {"ok": False, "reason": "no kernel image"})
        self.assertEqual(failed["torch_device"], "cpu")
        self.assertIn("no kernel image", failed["reason"])
        self.assertEqual(self.choose({"vendor": "nvidia", "shared_memory": False}, ok, "cpu")["torch_device"], "cpu")
        self.assertEqual(self.choose(None, ok)["torch_device"], "cpu")
        with mock.patch.dict(os.environ, {"CVL_NO_TORCH_GPU": "1"}):
            self.assertEqual(self.choose({"vendor": "intel", "shared_memory": True}, ok)["torch_device"], "cpu")

    def test_weights_stay_resident_only_when_they_fit_a_discrete_gpu(self):
        model = int(4.2 * GIB)
        self.assertTrue(accelerator.weights_resident(model, True, 12 * GIB))
        self.assertFalse(accelerator.weights_resident(model, True, 6 * GIB))
        self.assertFalse(accelerator.weights_resident(model, True, None))
        self.assertFalse(accelerator.weights_resident(model, False, 64 * GIB))

    def test_attention_tiles_follow_free_vram_on_discrete_gpus(self):
        class Module:
            def __init__(self, free):
                self.free = free
            def mem_get_info(self):
                return self.free, 24 * GIB
        class Torch:
            cuda = Module(20 * GIB)
        device = mock.Mock(type="cuda")
        self.assertEqual(accelerator.attention_budget_bytes(Torch, device, True), 1024 * 2**20)
        Torch.cuda = Module(2 * GIB)
        self.assertEqual(accelerator.attention_budget_bytes(Torch, device, True), int(2 * GIB * 0.1))
        Torch.cuda = Module(100 * 2**20)
        self.assertEqual(accelerator.attention_budget_bytes(Torch, device, True), 64 * 2**20)
        with mock.patch("local_video.resources.memory_bytes", return_value=(16 * GIB, 6 * GIB)):
            shared = accelerator.attention_budget_bytes(Torch, device, False)
        self.assertTrue(48 * 2**20 <= shared <= 384 * 2**20)

    def test_failure_record_blocks_a_backend_until_the_next_probe_window(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEMP")) as folder:
            root = Path(folder)
            (root / "runtime" / "cuda-site").mkdir(parents=True)
            (root / "runtime" / "neodragon-python").mkdir(parents=True)
            (root / "runtime" / "neodragon-python" / "python.exe").write_bytes(b"x")
            accelerator.mark_failed(root, "cuda", "video_infer: CUDA error: no kernel image")
            record = json.loads(accelerator._record_path(root, "cuda").read_text(encoding="utf-8"))
            self.assertFalse(record["ok"])
            self.assertTrue(record["failed_at_runtime"])
            with mock.patch("subprocess.run") as run:
                again = accelerator.probe(root, "cuda")
            run.assert_not_called()
            self.assertFalse(again["ok"])


class StageFallbackTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ["TEMP"]) / "fallback-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "runtime"
        (self.root / "runtime" / "cuda-site").mkdir(parents=True)
        (self.work / "request.json").write_text(json.dumps({"device_plan": {"torch_device": "cuda"}}), encoding="utf-8")
        self.progress = []

    def run_with(self, runner, plan=None, stage="video_infer"):
        request = {"device_plan": plan if plan is not None else {"torch_device": "cuda"}}
        result = neodragon.run_stage_adaptive(self.root, self.work, stage, request, self.progress.append, lambda: False, runner)
        return result, request

    def test_hardware_failure_reruns_the_stage_on_the_cpu_path(self):
        calls = []
        def runner(root, work, name, request, progress, cancelled):
            calls.append(request["device_plan"]["torch_device"])
            if request["device_plan"]["torch_device"] == "cuda":
                raise RuntimeError("CUDA error: no kernel image is available")
            return {"name": name, "state": "completed"}
        result, request = self.run_with(runner)
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(result["state"], "completed")
        self.assertIn("no kernel image", request["device_plan"]["torch_fallback_reason"])
        saved = json.loads((self.work / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["device_plan"]["torch_device"], "cpu")
        self.assertEqual(self.progress[-1]["stage"], "gpu_backend_failed_using_cpu_path")
        self.assertFalse(json.loads(accelerator._record_path(self.root, "cuda").read_text(encoding="utf-8"))["ok"])

    def test_memory_cancellation_and_timeouts_are_not_hardware_failures(self):
        for error in (MemoryError("low RAM"), InterruptedError("cancelled"), TimeoutError("slow")):
            calls = []
            def runner(root, work, name, request, progress, cancelled, error=error):
                calls.append(1)
                raise error
            with self.assertRaises(type(error)):
                self.run_with(runner)
            self.assertEqual(len(calls), 1)

    def test_cpu_plans_and_non_gpu_stages_run_once_without_fallback(self):
        calls = []
        def runner(root, work, name, request, progress, cancelled):
            calls.append(name)
            raise RuntimeError("real bug")
        with self.assertRaises(RuntimeError):
            self.run_with(runner, plan={"torch_device": "cpu"})
        with self.assertRaises(RuntimeError):
            self.run_with(runner, stage="safety")
        self.assertEqual(calls, ["video_infer", "safety"])


if __name__ == "__main__":
    unittest.main()

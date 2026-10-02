"""Adapter identity, shared RAM policy and forced-GPU failure boundaries."""
import unittest
from unittest.mock import patch
from local_video import devices, opencl_linear


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.intel = {"id": "Vulkan0", "name": "Intel Iris Xe Graphics", "vendor": "intel", "shared_memory": True}
        self.nvidia = {"id": "Vulkan1", "name": "NVIDIA RTX", "vendor": "nvidia", "shared_memory": False}

    def test_discrete_gpu_is_preferred_even_when_adapter_zero_is_integrated(self):
        self.assertEqual(devices.choose_device([self.intel, self.nvidia])["id"], "Vulkan1")

    def test_integrated_gpu_is_used_when_no_discrete_gpu_exists(self):
        self.assertEqual(devices.choose_device([self.intel])["id"], "Vulkan0")

    def test_cpu_auto_fallback_does_not_claim_gpu_execution(self):
        self.assertIsNone(devices.choose_device([], "auto"))

    def test_forced_gpu_and_invalid_id_never_fall_back_silently(self):
        for backend, selected in [("gpu", None), ("auto", "Vulkan9"), ("intel-gpu", "Vulkan1")]:
            with self.subTest(backend=backend, selected=selected), self.assertRaises(RuntimeError):
                devices.choose_device([self.nvidia] if selected else [], backend, selected)

    def test_explicit_cpu_does_not_accept_a_gpu_id(self):
        with self.assertRaises(ValueError):
            devices.choose_device([self.nvidia], "cpu", "Vulkan1")

    def test_intel_force_uses_intel_even_when_discrete_gpu_is_present(self):
        self.assertEqual(devices.choose_device([self.nvidia, self.intel], "intel-gpu"), self.intel)

    def test_driver_uma_property_overrides_arc_brand_name(self):
        text = "Vulkan0\tIntel Arc Graphics\nVulkan1\tIntel Arc A770\n0 = Intel Arc Graphics uma:1 fp16:1\n1 = Intel Arc A770 uma:0 fp16:1"
        result = devices.parse_devices(text)
        self.assertTrue(result[0]["shared_memory"])
        self.assertFalse(result[1]["shared_memory"])
        self.assertEqual(devices.choose_device(result)["id"], "Vulkan1")

    def test_log_messages_and_cpu_are_not_adapter_ids(self):
        self.assertEqual(devices.parse_devices("CPU\tIntel Core\n[INFO] Vulkan0 driver initialized\n0 = Intel uma:1"), [])

    def test_opencl_path_uses_same_small_buffer_budget_on_discrete_gpu(self):
        item = {**self.nvidia, "id": "OpenCL0", "maximum_work_group_size": 1024}
        with patch.object(opencl_linear, "inventory", return_value=[item]):
            result = opencl_linear.plan("auto")
        self.assertTrue(result["gpu_inference"])
        self.assertEqual(result["maximum_gpu_budget_gib"], 0.0625)
        self.assertEqual(result["minimum_free_ram_gib"], 3.0)
        self.assertEqual(result["gpu_scope"], "video_transformer_linear_layers_only")

    def test_opencl_driver_failure_is_visible_and_forced_gpu_fails(self):
        with patch.object(opencl_linear, "inventory", side_effect=OSError("driver missing")):
            result = opencl_linear.plan("auto")
            self.assertFalse(result["gpu_inference"])
            self.assertIn("driver missing", result["fallback_reason"])
            with self.assertRaises(RuntimeError):
                opencl_linear.plan("gpu")

    def test_old_gpu_with_small_workgroups_is_rejected(self):
        with patch.object(opencl_linear, "inventory", return_value=[{**self.intel, "maximum_work_group_size": 128}]):
            with self.assertRaises(RuntimeError):
                opencl_linear.plan("intel-gpu")


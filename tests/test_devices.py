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

    def test_opencl_plan_keeps_momentary_buffer_budget_out_of_cache_identity(self):
        item = {**self.nvidia, "id": "OpenCL0", "maximum_work_group_size": 1024}
        with patch.object(opencl_linear, "inventory", return_value=[item]):
            result = opencl_linear.plan("auto")
        self.assertTrue(result["gpu_inference"])
        self.assertEqual(result["gpu_buffer_policy"], "adaptive_per_stage")
        self.assertNotIn("maximum_gpu_budget_gib", result)
        self.assertEqual(result["minimum_free_ram_gib"], 3.0)
        self.assertEqual(result["gpu_scope"], "video_transformer_linear_layers_only")

    def test_gpu_buffer_follows_shared_ram_or_dedicated_vram(self):
        from local_video import resources
        gib, mib = 2**30, 2**20
        shared = {"shared_memory": True}
        self.assertEqual(resources.gpu_buffer_bytes(shared, 16 * gib, 3 * gib), 64 * mib)
        self.assertEqual(resources.gpu_buffer_bytes(shared, 16 * gib, 1 * gib), resources.GPU_FLOOR)
        self.assertEqual(resources.gpu_buffer_bytes(shared, 32 * gib, 20 * gib), resources.GPU_SHARED_CEILING)
        discrete = {"shared_memory": False, "global_memory_bytes": 8 * gib}
        self.assertEqual(resources.gpu_buffer_bytes(discrete, 32 * gib, 2 * gib), resources.GPU_DISCRETE_CEILING)
        self.assertEqual(resources.gpu_buffer_bytes({"shared_memory": False, "global_memory_bytes": 2 * gib}, 16 * gib, 2 * gib), 512 * mib)
        self.assertEqual(resources.gpu_buffer_bytes(shared, 16 * gib, 8 * gib, 48), 48 * mib)
        self.assertIsNone(resources.gpu_buffer_bytes(None, 16 * gib, 8 * gib))

    def test_ram_limits_scale_with_computer_and_current_free_memory(self):
        from local_video import resources
        gib = 2**30
        with patch.object(resources, "memory_bytes", return_value=(16 * gib, 4 * gib)):
            gram = resources.stage_limits({}, "neodragon")
        with patch.object(resources, "memory_bytes", return_value=(64 * gib, 40 * gib)):
            large = resources.stage_limits({}, "neodragon")
        self.assertEqual(gram["maximum_working_set_gib"], resources.WORKING_SET_FLOOR_GIB)
        self.assertEqual(gram["minimum_free_ram_gib"], 3.0)
        self.assertLess(gram["reserve_ram_gib"], large["reserve_ram_gib"])
        self.assertEqual(large["maximum_working_set_gib"], 32.0)
        with patch.object(resources, "memory_bytes", return_value=(64 * gib, 40 * gib)):
            manual = resources.stage_limits({"maximum_working_set_gib": 6, "reserve_ram_gib": 2}, "neodragon")
        self.assertEqual((manual["maximum_working_set_gib"], manual["reserve_ram_gib"]), (6.0, 2.0))

    def test_shared_gpu_buffer_shrinks_under_ram_pressure_and_recovers(self):
        from local_video import backend, resources
        gib, mib = 2**30, 2**20
        engine = opencl_linear.LinearEngine.__new__(opencl_linear.LinearEngine)
        engine.reserve, engine.ceiling, engine.floor, engine.step = 1.5 * gib, 128 * mib, resources.GPU_FLOOR, resources.GPU_STEP
        engine.adaptive, engine.budget, engine.lowest_budget = True, 128 * mib, 128 * mib
        engine.shrinks = engine.grows = engine.working_set_trims = 0
        with patch.object(backend, "_memory", return_value=(0, int(2.0 * gib))):
            engine.adapt()
            engine.adapt()
            engine.adapt()
        self.assertEqual(engine.budget, resources.GPU_FLOOR)
        self.assertEqual(engine.shrinks, 2)
        with patch.object(backend, "_memory", return_value=(0, 6 * gib)):
            engine.adapt()
            engine.adapt()
        self.assertEqual(engine.budget, 128 * mib)
        self.assertEqual(engine.lowest_budget, resources.GPU_FLOOR)

    def test_short_ram_dip_is_tolerated_but_sustained_or_severe_shortage_stops(self):
        from local_video import resources
        gib = 2**30
        guard = resources.PressureGuard(1.5 * gib, grace_seconds=15)
        self.assertFalse(guard.check(3 * gib, 0))
        self.assertTrue(guard.check(1.3 * gib, 1))
        self.assertTrue(guard.check(1.3 * gib, 10))
        self.assertFalse(guard.check(2 * gib, 11))
        self.assertTrue(guard.check(1.3 * gib, 12))
        with self.assertRaisesRegex(MemoryError, "초 넘게"):
            guard.check(1.3 * gib, 28)
        with self.assertRaisesRegex(MemoryError, "즉시 중지"):
            resources.PressureGuard(1.5 * gib).check(1.0 * gib, 0)

    def test_start_requirement_learns_each_stage_ram_drop_on_this_computer(self):
        import tempfile
        from local_video import resources
        gib = 2**30
        limits = {"minimum_free_ram_gib": 3.0, "reserve_ram_gib": 1.4}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(resources.start_requirement_gib(folder, "first_frame", limits), 3.0)
            resources.record_drop(folder, "first_frame", int(3.3 * gib), int(1.0 * gib))
            self.assertAlmostEqual(resources.start_requirement_gib(folder, "first_frame", limits), 2.3 * 1.1 + 1.05, places=1)
            self.assertEqual(resources.start_requirement_gib(folder, "video_text", limits), 3.0)
            resources.record_drop(folder, "video_text", int(4 * gib), int(3.5 * gib))
            self.assertGreater(resources.start_requirement_gib(folder, None, limits), 3.5)
            resources.record_drop(folder, "first_frame", int(3 * gib), int(2 * gib))
            self.assertAlmostEqual(resources.learned_drops(folder)["first_frame"], 2.07, places=2)

    def test_ram_only_shortfall_is_queued_but_other_problems_are_refused(self):
        from local_video import control
        ram_only = {"ready": False, "ram_shortfall_only": True, "errors": ["RAM"], "quality_ready": False}
        control._require_startable(ram_only, True)
        with self.assertRaisesRegex(RuntimeError, "품질"):
            control._require_startable(ram_only, False)
        with self.assertRaisesRegex(RuntimeError, "준비 미완료"):
            control._require_startable({"ready": False, "ram_shortfall_only": False, "errors": ["모델"]}, True)
        control._require_startable({"ready": True, "quality_ready": True, "errors": []}, False)

    def test_resource_values_accept_auto_or_bounded_numbers(self):
        from local_video import resources
        self.assertEqual(resources.check_value("reserve_ram_gib", None), "auto")
        self.assertEqual(resources.check_value("gpu_buffer_mib", 128), 128)
        for key, value in [("gpu_buffer_mib", 8), ("gpu_buffer_mib", 4096), ("reserve_ram_gib", -1), ("reserve_ram_gib", True)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resources.check_value(key, value)

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


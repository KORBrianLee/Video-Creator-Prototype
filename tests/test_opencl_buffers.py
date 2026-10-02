"""Allocation pressure and failure cleanup without requiring a GPU in unit tests."""
import unittest
from local_video.opencl_linear import LinearEngine


class Driver:
    def __init__(self, budget):
        self.budget, self.live, self.sequence, self.peak = budget, {}, 0, 0

    def clCreateBuffer(self, context, flags, size, host, error):
        assert sum(self.live.values()) + size <= self.budget, "temporary over-allocation"
        self.sequence += 1
        self.live[self.sequence] = size
        self.peak = max(self.peak, sum(self.live.values()))
        return self.sequence

    def clReleaseMemObject(self, handle):
        del self.live[handle]
        return 0


class BufferTests(unittest.TestCase):
    def engine(self, budget=100):
        engine = LinearEngine.__new__(LinearEngine)
        engine.dll = Driver(budget)
        engine.context, engine.buffers, engine.resources = None, {}, []
        engine.budget, engine.device = budget, {"maximum_allocation_bytes": budget}
        engine.allocations = engine.reuses = engine.releases = engine.peak = 0
        self.addCleanup(engine.close)
        return engine

    def test_repeated_and_smaller_layers_reuse_capacity(self):
        engine = self.engine()
        first = engine.prepare_buffers({"w": 40, "b": 4, "a": 20, "c": 20})
        self.assertEqual(engine.prepare_buffers({"w": 30, "b": 4, "a": 10, "c": 10}), first)
        self.assertEqual(engine.allocations, 4)
        self.assertEqual(engine.reuses, 4)

    def test_shape_change_reclaims_capacity_before_growth(self):
        engine = self.engine()
        engine.prepare_buffers({"w": 60, "b": 4, "a": 16, "c": 20})
        handles = engine.prepare_buffers({"w": 20, "b": 4, "a": 36, "c": 40})
        self.assertEqual(sum(engine.dll.live.values()), 100)
        self.assertLessEqual(engine.dll.peak, 100)
        self.assertEqual(len(handles), 4)
        engine.close()
        engine.close()
        self.assertEqual(engine.dll.live, {})

    def test_impossible_layer_preserves_live_allocations_and_cap(self):
        engine = self.engine()
        first = engine.prepare_buffers({"w": 40, "b": 4, "a": 20, "c": 20})
        for layout in [{"w": 80, "b": 4, "a": 20, "c": 20}, {"w": 101, "b": 4, "a": 1, "c": 1}]:
            with self.assertRaises(MemoryError):
                engine.prepare_buffers(layout)
        self.assertEqual({k: v[0] for k, v in engine.buffers.items()}, first)
        self.assertLessEqual(engine.dll.peak, 100)

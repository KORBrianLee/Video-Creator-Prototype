import unittest

from local_video import opencl_linear


def first_gpu():
    try:
        return opencl_linear.plan()["selected_device"]
    except Exception:
        return None


class KernelSourceTests(unittest.TestCase):
    def test_generated_kernel_matches_requested_tile(self):
        source = opencl_linear.tiled_kernel(8, 4)
        self.assertIn("As[16][128]", source)
        self.assertIn("Ws[16][64]", source)
        self.assertEqual(source.count("fma("), 8 * 4)
        self.assertEqual(opencl_linear.tiled_kernel(4, 4).count("fma("), 16)


class KernelAccuracyTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy
            import torch
        except ImportError:
            self.skipTest("numpy and torch are needed for the reference result")
        self.device = first_gpu()
        if not self.device:
            self.skipTest("no OpenCL GPU")
        self.numpy, self.torch = numpy, torch

    def reference(self, activation, bits, bias):
        weight = self.torch.from_numpy(bits.view(self.numpy.int16).copy()).view(self.torch.bfloat16).float().numpy()
        out = activation @ weight.T
        return out + bias if bias is not None else out

    def test_every_tile_and_edge_shape_matches_reference(self):
        np = self.numpy
        rng = np.random.default_rng(3)
        for tile in ((4, 4), (8, 4), (4, 8)):
            engine = opencl_linear.LinearEngine(self.device["id"], tile=tile)
            try:
                # Edge sizes cross tile borders; K=30 forces the scalar fallback kernel.
                for m, n, k in ((1, 1, 4), (37, 70, 36), (130, 131, 64), (65, 33, 30), (200, 96, 1536)):
                    activation = rng.standard_normal((m, k), dtype=np.float32)
                    bits = (rng.standard_normal((n, k), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
                    bias = rng.standard_normal(n, dtype=np.float32)
                    for used in (None, bias):
                        got = engine.linear(activation, bits, used)
                        want = self.reference(activation, bits, used)
                        scale = max(1.0, float(np.abs(want).max()))
                        self.assertLess(float(np.abs(got - want).max()) / scale, 1e-4, (tile, m, n, k))
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()

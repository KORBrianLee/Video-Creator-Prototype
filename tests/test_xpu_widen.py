import unittest


def torch_or_skip(case):
    try:
        import torch
    except ImportError:
        case.skipTest("torch is needed")
    return torch


class WidenTests(unittest.TestCase):
    def test_bit_widening_equals_exact_bf16_to_float32(self):
        torch = torch_or_skip(self)
        from local_video.neodragon_quant import MappedLinearTorchGPU
        generator = torch.Generator().manual_seed(5)
        for shape in ((1, 2), (7, 64), (33, 1536)):
            weight = (torch.randn(*shape, generator=generator) * 3).to(torch.bfloat16)
            for bitwise in (True, False):
                widened = MappedLinearTorchGPU.widen(weight, torch.device("cpu"), bitwise=bitwise)
                self.assertEqual(widened.dtype, torch.float32)
                self.assertEqual(widened.shape, weight.shape)
                self.assertTrue(torch.equal(widened, weight.float()))

    def test_negative_zero_infinity_and_subnormal_values_survive(self):
        torch = torch_or_skip(self)
        from local_video.neodragon_quant import MappedLinearTorchGPU
        weight = torch.tensor([[0.0, -0.0, float("inf"), -float("inf"), 1e-38, -3.5, 65504.0, 1.0]]).to(torch.bfloat16)
        widened = MappedLinearTorchGPU.widen(weight, torch.device("cpu"), bitwise=True)
        self.assertTrue(torch.equal(widened.view(torch.int32), weight.float().view(torch.int32)))

    def test_only_intel_gpus_use_the_bit_form(self):
        torch = torch_or_skip(self)
        from local_video.neodragon_quant import MappedLinearTorchGPU
        self.assertTrue(MappedLinearTorchGPU.bitwise_for(torch.device("xpu")))
        self.assertFalse(MappedLinearTorchGPU.bitwise_for(torch.device("cuda")))
        self.assertFalse(MappedLinearTorchGPU.bitwise_for(torch.device("cpu")))

    def test_resident_and_streamed_weights_give_identical_results(self):
        torch = torch_or_skip(self)
        from local_video.neodragon_quant import MappedLinearTorchGPU, linear_weight_bytes
        generator = torch.Generator().manual_seed(11)
        linear = torch.nn.Linear(64, 48)
        linear.weight.data = linear.weight.data.to(torch.bfloat16)
        linear.bias.data = linear.bias.data.to(torch.bfloat16)
        value = torch.randn(5, 64, generator=generator)
        streamed = MappedLinearTorchGPU(linear)
        resident = MappedLinearTorchGPU(linear)
        resident.pin(torch.device("cpu"))
        self.assertIsNone(streamed.stored)
        self.assertIsNotNone(resident.stored)
        self.assertTrue(torch.equal(streamed(value), resident(value)))
        self.assertEqual(linear_weight_bytes(torch.nn.Sequential(linear)), 64 * 48 * 2)


class GpuPacerTests(unittest.TestCase):
    def test_the_gpu_rests_a_ninth_of_each_burst_at_90_percent(self):
        from types import SimpleNamespace
        from local_video import accelerator
        now, sleeps, syncs = [0.0], [], []

        def sleep(seconds):
            sleeps.append(round(seconds, 4))
            now[0] += seconds

        gpu = SimpleNamespace(synchronize=lambda: syncs.append(now[0]))
        pacer = accelerator.GpuPacer(gpu, 90, burst=0.1, clock=lambda: now[0], sleep=sleep)
        now[0] = 0.05
        pacer()
        self.assertEqual((sleeps, syncs), ([], []))
        now[0] = 0.18
        pacer()
        self.assertEqual((sleeps, syncs), ([0.02], [0.18]))
        try:
            accelerator.install_pacer(pacer)
            now[0] += 0.27
            accelerator.pace()
            self.assertEqual(sleeps[-1], 0.03)
        finally:
            accelerator.install_pacer(None)
        accelerator.pace()
        self.assertEqual(len(sleeps), 2)

    def test_a_long_linear_runs_in_row_slices_with_a_rest_after_each(self):
        torch = torch_or_skip(self)
        from local_video import accelerator
        from local_video.neodragon_quant import MappedLinearTorchGPU
        torch.manual_seed(1)
        linear = torch.nn.Linear(6, 5)
        value = torch.randn(2, 7, 6)
        expected = linear(value)
        mapped = MappedLinearTorchGPU(linear)
        mapped.max_rows = 4
        rests = []
        try:
            accelerator.install_pacer(lambda: rests.append(1))
            with torch.no_grad():
                actual = mapped(value)
        finally:
            accelerator.install_pacer(None)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(len(rests), 4)


class SplitConvTests(unittest.TestCase):
    def test_a_conv_split_along_time_matches_one_dispatch(self):
        torch = torch_or_skip(self)
        from local_video.neodragon_quant import MappedConvTorchGPU, split_convs
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Conv3d(4, 6, (3, 3, 3), padding=(0, 1, 1)),
                                    torch.nn.Conv3d(6, 5, (3, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)))
        value = torch.randn(1, 4, 11, 5, 7)
        with torch.no_grad():
            expected = model(value)
            wrapped = torch.nn.Sequential(MappedConvTorchGPU(model[0]), MappedConvTorchGPU(model[1]))
            self.assertEqual(split_convs(wrapped, 2 * 5 * 7), 2)
            slices = []
            first = wrapped[0].original._conv_forward
            wrapped[0].original._conv_forward = lambda value, *rest: slices.append(value.shape[2]) or first(value, *rest)
            actual = wrapped(value)
        # Two output frames per dispatch read four input frames; the strided conv is left whole.
        self.assertEqual(slices, [4, 4, 4, 4, 3])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class BoundedAttentionTests(unittest.TestCase):
    def test_chunked_attention_matches_full_attention_with_and_without_mask(self):
        torch = torch_or_skip(self)
        from local_video import accelerator
        functional = torch.nn.functional
        reference = functional.scaled_dot_product_attention
        generator = torch.Generator().manual_seed(9)
        query, key, value = (torch.randn(1, 3, 200, 16, generator=generator) for _ in range(3))
        mask = torch.rand(1, 1, 200, 200, generator=generator) > 0.2
        mask[..., 0] = True
        # Room for 25 query rows of scores (3 heads x 200 keys x 4 bytes each).
        tile_budget = lambda: 3 * 200 * 4 * 25
        try:
            for use_mask in (None, mask):
                rows_seen = []

                def spy(q, *args, **kwargs):
                    rows_seen.append(q.shape[-2])
                    return reference(q, *args, **kwargs)

                functional.scaled_dot_product_attention = spy
                accelerator.bound_attention_memory(torch, budget=tile_budget, device_types=("cpu",))
                chunked = functional.scaled_dot_product_attention(query, key, value, attn_mask=use_mask)
                self.assertEqual(rows_seen, [25] * 8)
                functional.scaled_dot_product_attention = reference
                full = reference(query, key, value, attn_mask=use_mask)
                self.assertTrue(torch.allclose(chunked, full, atol=1e-5), "chunked attention changed the result")
        finally:
            functional.scaled_dot_product_attention = reference

    def test_plain_attention_matches_the_kernel_and_boolean_masks_stay_on_it(self):
        torch = torch_or_skip(self)
        from local_video import accelerator
        functional = torch.nn.functional
        reference = functional.scaled_dot_product_attention
        generator = torch.Generator().manual_seed(9)
        query, key, value = (torch.randn(1, 3, 200, 16, generator=generator) for _ in range(3))
        # The prompt mask arrives as an additive bias (0 or -10000 per key).
        bias = torch.where(torch.rand(1, 1, 1, 200, generator=generator) > 0.2, 0.0, -10000.0)
        boolean = torch.rand(1, 1, 200, 200, generator=generator) > 0.2
        boolean[..., 0] = True
        calls = []

        def spy(q, *args, **kwargs):
            calls.append(q.shape[-2])
            return reference(q, *args, **kwargs)

        try:
            functional.scaled_dot_product_attention = spy
            accelerator.bound_attention_memory(torch, budget=lambda: 3 * 200 * 4 * 25, device_types=("cpu",),
                                               plain_types=("cpu",))
            for mask in (None, bias):
                plain = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
                self.assertTrue(torch.allclose(plain, reference(query, key, value, attn_mask=mask), atol=1e-5))
            self.assertEqual(calls, [])
            functional.scaled_dot_product_attention(query, key, value, attn_mask=boolean)
            self.assertEqual(calls, [25] * 8)
        finally:
            functional.scaled_dot_product_attention = reference

    def test_short_memory_takes_heads_in_turn_with_the_same_result(self):
        torch = torch_or_skip(self)
        from local_video import accelerator
        functional = torch.nn.functional
        reference = functional.scaled_dot_product_attention
        generator = torch.Generator().manual_seed(5)
        query, key, value = (torch.randn(1, 3, 200, 16, generator=generator) for _ in range(3))
        full = reference(query, key, value)
        one_head_row = 200 * 4 * 2
        # 50 rows of one head; then two whole heads at once.
        for allowed in (one_head_row * 50, one_head_row * 200 * 2):
            try:
                accelerator.bound_attention_memory(torch, budget=lambda: allowed, device_types=("cpu",),
                                                   plain_types=("cpu",))
                chunked = functional.scaled_dot_product_attention(query, key, value)
            finally:
                functional.scaled_dot_product_attention = reference
            self.assertEqual(chunked.shape, full.shape)
            self.assertTrue(torch.allclose(chunked, full, atol=1e-5), f"budget {allowed}")


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAS_TORCH = importlib.util.find_spec("torch") is not None


class LtxRecordTests(unittest.TestCase):
    def test_timing_splits_loading_steps_and_decoding(self):
        from local_video.ltx_stage import timing_record
        record = timing_record(load_start=100.0, pipe_start=112.0, step_ends=[345.7, 491.7, 641.7], pipe_end=942.2)
        self.assertEqual(record, {"load_seconds": 12.0, "denoise_step_seconds": [233.7, 146.0, 150.0],
                                  "decode_seconds": 300.5})

    def test_text_cache_follows_prompt_budget_and_encoder_only(self):
        from local_video.ltx_stage import text_cache_key
        key = text_cache_key("a loader stops", 128, Path("models/ltx-video-2b-0.9.8"))
        self.assertEqual(key, text_cache_key("a loader stops", 128, Path("D:/elsewhere/ltx-video-2b-0.9.8")))
        self.assertEqual(len({key, text_cache_key("a loader stops", 256, Path("ltx-video-2b-0.9.8")),
                              text_cache_key("a loader stops.", 128, Path("ltx-video-2b-0.9.8")),
                              text_cache_key("a loader stops", 128, Path("ltx-video-2b-0.9.9"))}), 4)

    def test_a_prompt_longer_than_the_encoder_reads_is_marked(self):
        from local_video.ltx_stage import prompt_record
        self.assertEqual(prompt_record(142, 128), {"tokens": 142, "limit": 128, "truncated": True})
        self.assertFalse(prompt_record(128, 128)["truncated"])


@unittest.skipUnless(HAS_TORCH, "torch is only in the video runtime")
class LtxStageTests(unittest.TestCase):
    def test_noise_is_drawn_on_the_generator_device_then_moved(self):
        import torch
        from local_video.ltx_stage import generator_side_noise
        original = torch.randn
        with generator_side_noise(torch):
            noise = torch.randn((2, 3), generator=torch.Generator().manual_seed(7), device="meta")
            same = torch.randn((2, 3), generator=torch.Generator().manual_seed(7), device="cpu")
        self.assertEqual(noise.device.type, "meta")
        self.assertTrue(torch.equal(same, original((2, 3), generator=torch.Generator().manual_seed(7))))
        self.assertIs(torch.randn, original)

    def test_rope_tables_are_built_on_the_cpu(self):
        import torch
        from local_video.ltx_stage import cpu_rope
        seen = []

        class Rope(torch.nn.Module):
            def forward(self, hidden_states, num_frames, height, width, scale=None, coords=None):
                seen.append((hidden_states.device.type, hidden_states.shape[0], coords))
                table = torch.ones(hidden_states.shape[0], 4)
                return table, table

        rope = cpu_rope(Rope(), torch)
        cos, sin = rope(torch.zeros(2, 4, device="meta"), 1, 2, 2, (0.32, 32, 32), None)
        self.assertEqual(seen, [("cpu", 2, None)])
        self.assertEqual((cos.device.type, sin.device.type), ("meta", "meta"))


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("diffusers") is not None, "video runtime only")
class CpuDecodedTilesTests(unittest.TestCase):
    def test_each_tile_leaves_the_gpu_as_soon_as_it_is_decoded(self):
        import torch
        from diffusers.models.autoencoders.vae import DecoderOutput
        from local_video.ltx_stage import cpu_decoded_tiles

        class Vae:
            def tiled_decode(self, z, temb, return_dict=True):
                return DecoderOutput(sample=torch.zeros(1, 3, 2, 4, 4, device="meta"))

        vae = cpu_decoded_tiles(Vae(), torch)
        with self.assertRaises(NotImplementedError):
            vae.tiled_decode(None, None)  # meta tensors cannot be copied: proves the move to CPU happens


@unittest.skipUnless(HAS_TORCH, "torch is only in the video runtime")
class CpuSchedulerStepTests(unittest.TestCase):
    def test_per_token_step_runs_on_cpu_and_returns_to_the_sample_device(self):
        import torch
        from local_video.ltx_stage import cpu_scheduler_step
        seen = []

        class Scheduler:
            sigmas = torch.tensor([1.0, 0.5, 0.0], device="meta")
            timesteps = torch.tensor([1000.0, 500.0], device="meta")

            def step(self, model_output, timestep, sample, per_token_timesteps=None, return_dict=False):
                seen.append((sample.device.type, per_token_timesteps.device.type, self.sigmas.device.type))
                return (sample + 1,)

        scheduler = Scheduler()
        cpu_scheduler_step(scheduler)
        sample = torch.zeros(1, 4)
        scheduler.sigmas, scheduler.timesteps = torch.tensor([1.0, 0.5, 0.0]), torch.tensor([1000.0, 500.0])
        out = scheduler.step(torch.zeros(1, 4), torch.tensor(1000.0), sample, per_token_timesteps=torch.ones(1, 4))
        self.assertEqual(seen, [("cpu", "cpu", "cpu")])
        self.assertTrue(torch.equal(out[0], torch.ones(1, 4)))


if __name__ == "__main__":
    unittest.main()

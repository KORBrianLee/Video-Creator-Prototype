import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAS_TORCH = importlib.util.find_spec("torch") is not None


class HeldBackgroundTests(unittest.TestCase):
    def frames(self, temp):
        from PIL import Image, ImageDraw
        first = Image.new("RGB", (128, 64))
        first.putdata([((x * 37 + y * 91) % 256, (x * 11) % 256, (y * 7) % 256) for y in range(64) for x in range(128)])
        moved = first.copy()
        ImageDraw.Draw(moved).rectangle((36, 4, 60, 28), fill=(255, 200, 0))   # something moves in cell row 0, column 1
        paths = [Path(temp) / "first.png", Path(temp) / "key.png"]
        first.save(paths[0])
        moved.save(paths[1])
        return paths

    def test_cells_that_change_and_their_neighbours_stay_free(self):
        from local_video import ltx
        with tempfile.TemporaryDirectory() as temp:
            first, key = self.frames(temp)
            self.assertEqual(ltx.background_cells(first, [key], 128, 64), [[0, 0, 0, 1], [0, 0, 0, 1]])
            self.assertEqual(ltx.background_cells(first, [key], 128, 64, margin=0), [[1, 0, 1, 1], [1, 1, 1, 1]])
            self.assertEqual(ltx.background_cells(first, [key], 128, 64, free_boxes=[(0.8, 0.6, 1.0, 1.0)], margin=0),
                             [[1, 0, 1, 1], [1, 1, 1, 0]])

    def test_held_tokens_skip_the_first_latent_frame(self):
        from local_video import ltx
        self.assertEqual(ltx.held_tokens([[1, 0], [0, 1]], 3), [4, 7, 8, 11])
        self.assertEqual(ltx.held_tokens([[1, 0], [0, 1]], 5, range(2, 4)), [8, 11, 12, 15])

    def test_a_span_holds_the_latent_frames_that_lie_inside_it(self):
        from local_video import ltx
        # 15 s at 24 fps: 46 latent frames; latent frame k carries video frames 8k-7 to 8k.
        self.assertEqual(ltx.latent_span(8.0, 11.0, 24, 46), range(25, 34))
        self.assertEqual(ltx.latent_span(11.0, 15.0, 24, 46), range(34, 46))
        self.assertEqual(ltx.latent_span(0.0, 15.0, 24, 46), range(1, 46))


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("diffusers") is not None, "video runtime only")
class HoldSpanTests(unittest.TestCase):
    def test_a_span_holds_its_cells_to_the_picture_only_in_its_latent_frames(self):
        import torch
        from types import SimpleNamespace
        from PIL import Image
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline
        from local_video.ltx_stage import hold_background
        rows, cols, frames, channels = 2, 3, 4, 2

        class Encoded:
            """A fake VAE: every latent value is the mean of the encoded picture."""
            def __init__(self, video):
                self.latents = torch.full((1, channels, (video.shape[2] - 1) // 8 + 1, rows, cols), float(video.mean()))

        pipe = SimpleNamespace(
            prepare_latents=lambda conditions, *a, **k: (torch.zeros(1, frames * rows * cols, channels),
                                                         torch.zeros(1, frames * rows * cols), None, 0),
            vae=SimpleNamespace(encode=Encoded, latents_mean=torch.zeros(channels), latents_std=torch.ones(channels)),
            _normalize_latents=LTXConditionPipeline._normalize_latents, _pack_latents=LTXConditionPipeline._pack_latents,
            video_processor=SimpleNamespace(preprocess=lambda image, h, w: torch.full((1, 3, h, w), 0.5)))
        info = hold_background(pipe, [[1, 1, 1], [1, 0, 1]], 0.9, frames, torch, Image.new("RGB", (96, 64)), range(2, 4))
        tokens, levels, _, _ = pipe.prepare_latents([torch.full((1, 3, 1, 64, 96), -0.5)])
        per_frame = rows * cols
        held = [f * per_frame + i for f in (2, 3) for i in (0, 1, 2, 3, 5)]
        self.assertTrue(torch.allclose(tokens[0, held], torch.full((10, channels), 0.45)))
        self.assertTrue(torch.allclose(levels[0, held], torch.full((10,), 0.9)))
        untouched = [i for i in range(frames * per_frame) if i not in held]
        self.assertEqual(float(tokens[0, untouched].abs().sum() + levels[0, untouched].sum()), 0.0)
        self.assertEqual(info, {"cells": 5, "of": 6, "strength": 0.9, "latent_frames": [2, 3]})

    def test_a_still_clip_encodes_to_the_latent_of_its_single_frame(self):
        import torch
        from diffusers import AutoencoderKLLTXVideo
        # The 0.9.5 VAE layout (causal encoder, spatial/temporal/spatiotemporal downsampling) at toy width.
        torch.manual_seed(0)
        vae = AutoencoderKLLTXVideo(
            block_out_channels=(8, 16, 32, 64, 64), decoder_block_out_channels=(16, 32, 64), latent_channels=16,
            layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1),
            down_block_types=("LTXVideo095DownBlock3D",) * 4,
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            spatio_temporal_scaling=(True,) * 4, decoder_spatio_temporal_scaling=(True,) * 3,
            decoder_inject_noise=(False,) * 4, upsample_residual=(True,) * 3, upsample_factor=(2,) * 3,
            timestep_conditioning=True, encoder_causal=True, decoder_causal=False, patch_size=4, patch_size_t=1).eval()
        picture = torch.rand(1, 3, 1, 96, 128) * 2 - 1
        with torch.no_grad():
            single = vae.encode(picture).latent_dist.mode()
            clip = vae.encode(picture.repeat(1, 1, 17, 1, 1)).latent_dist.mode()
        self.assertEqual(clip.shape[2], 3)
        self.assertTrue(torch.allclose(clip, single.expand_as(clip), atol=1e-5))


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
    def test_rope_tables_are_reused_until_the_coordinates_change(self):
        import torch
        from types import SimpleNamespace
        from local_video.ltx_stage import cpu_rope
        computed = []

        def forward(probe, video_coords=None):
            computed.append(tuple(video_coords.shape))
            return video_coords.sum(1, keepdim=True).cos(), video_coords.sum(1, keepdim=True).sin()

        rope = SimpleNamespace(forward=forward)
        cpu_rope(rope, torch)
        # The pipeline runs under inference mode, whose tensors have no version counter.
        with torch.inference_mode():
            hidden, coords = torch.zeros(1, 5, 8), torch.rand(1, 3, 100)
            first = rope.forward(hidden, video_coords=coords)
            self.assertIs(rope.forward(hidden, video_coords=coords)[0], first[0])
            coords.add_(1)
            self.assertIsNot(rope.forward(hidden, video_coords=coords)[0], first[0])
            self.assertEqual(tuple(rope.forward(hidden, video_coords=torch.rand(1, 3, 400))[0].shape), (1, 1, 400))
        self.assertEqual(len(computed), 3)

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

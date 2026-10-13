"""LTX-Video 2B 0.9.8 distilled: the long-video tier for Intel GPUs (Iris Xe, Arc) through PyTorch XPU.

LTX compresses video 32x in space and 8x in time, so a 15s clip at 512x320 is about 7,400 tokens,
small enough for an integrated GPU. The distilled model needs 8 denoising steps and no
classifier-free guidance. Clips are generated in one pass (15s = 361 frames, 46 latent frames;
the model's temporal position range is 20s).
"""
from __future__ import annotations

LOCK = "ltx.lock.json"
# Official distilled schedule for the 0.9.7/0.9.8 distilled models (8 steps, last step near zero).
TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]


# Both sides are multiples of 32 (the VAE's spatial factor). 16:9 matches widescreen first frames without
# squeezing them, and 512x288 has 10% fewer tokens than 512x320. The next exact 16:9 size is 1024x576,
# four times the tokens: short clips only on an integrated GPU.
SIZES = {("16:10", "preview"): (512, 320), ("16:10", "quality"): (768, 448),
         ("16:9", "preview"): (512, 288), ("16:9", "quality"): (1024, 576)}


def vae_tiling(height, decode="lean"):
    """VAE decode tiles (sample pixels and frames). One whole-clip dispatch outlasted the Iris Xe GPU watchdog
    (DEVICE_LOST), so the clip is decoded in blended tiles. "safe" (small tiles, 8-frame stride over 16-frame tiles)
    decodes every latent frame three times and the 192 px stride adds half again. "lean" decodes frames up to 320 px
    high in one row and steps 16 frames over 24-frame tiles, with the same 64 px and 8-frame blends: on a 15 s
    512x288 clip it decoded in 440 s against 939-1396 s, at the same peak memory and without visible seams at tile
    edges or stride frames. Taller frames keep the safe tiles."""
    if decode == "lean" and height <= 320:
        return {"tile_sample_min_height": height, "tile_sample_min_width": 256, "tile_sample_min_num_frames": 24,
                "tile_sample_stride_height": height, "tile_sample_stride_width": 192, "tile_sample_stride_num_frames": 16}
    return {"tile_sample_min_height": 256, "tile_sample_min_width": 256, "tile_sample_min_num_frames": 16,
            "tile_sample_stride_height": 192, "tile_sample_stride_width": 192, "tile_sample_stride_num_frames": 8}


# Prompt tokens the T5 encoder reads: 128 is the diffusers default (about 90 words, the rest is cut off);
# the official LTX-Video pipeline encodes up to 256 (about 180 words).
def settings_for(duration, preset="preview", aspect="16:10", text_tokens=128, decode="lean"):
    width, height = SIZES[(aspect, "quality" if preset == "quality" else "preview")]
    return {"width": width, "height": height, "frames": int(duration) * 24 + 1, "fps": 24,
            "timesteps": TIMESTEPS, "guidance_scale": 1.0, "decode_timestep": 0.05, "decode_noise_scale": 0.025,
            "max_sequence_length": text_tokens, "vae_tiling": vae_tiling(height, decode)}


def installed(root, source):
    from .skyreels import installed as lock_installed
    return lock_installed(root, source, LOCK)

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


# Prompt tokens the T5 encoder reads: 128 is the diffusers default (about 90 words, the rest is cut off);
# the official LTX-Video pipeline encodes up to 256 (about 180 words).
def settings_for(duration, preset="preview", aspect="16:10", text_tokens=128):
    width, height = SIZES[(aspect, "quality" if preset == "quality" else "preview")]
    return {"width": width, "height": height, "frames": int(duration) * 24 + 1, "fps": 24,
            "timesteps": TIMESTEPS, "guidance_scale": 1.0, "decode_timestep": 0.05, "decode_noise_scale": 0.025,
            "max_sequence_length": text_tokens}


def installed(root, source):
    from .skyreels import installed as lock_installed
    return lock_installed(root, source, LOCK)

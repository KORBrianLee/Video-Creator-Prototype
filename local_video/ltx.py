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


def settings_for(duration, preset="preview"):
    width, height = (768, 448) if preset == "quality" else (512, 320)
    return {"width": width, "height": height, "frames": int(duration) * 24 + 1, "fps": 24,
            "timesteps": TIMESTEPS, "guidance_scale": 1.0, "decode_timestep": 0.05, "decode_noise_scale": 0.025,
            "max_sequence_length": 128}


def installed(root, source):
    from .skyreels import installed as lock_installed
    return lock_installed(root, source, LOCK)

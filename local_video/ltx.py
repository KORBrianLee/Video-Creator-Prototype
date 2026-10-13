"""LTX-Video 2B 0.9.8 distilled: the long-video tier for Intel GPUs (Iris Xe, Arc) through PyTorch XPU.

LTX compresses video 32x in space and 8x in time, so a 15s clip at 512x320 is about 7,400 tokens,
small enough for an integrated GPU. The distilled model needs 8 denoising steps and no
classifier-free guidance. Clips are generated in one pass (15s = 361 frames, 46 latent frames;
the model's temporal position range is 20s).
"""
from __future__ import annotations

LOCK = "ltx.lock.json"
UPSCALER_LOCK = "ltx-upscaler.lock.json"
# Official distilled schedule for the 0.9.7/0.9.8 distilled models (8 steps, last step near zero).
TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
# Official multi-scale recipe for the 0.9.8 distilled models: the first pass stops at 0.725, the latent upscaler
# doubles height and width, and a short second pass runs 0.9094, 0.725, 0.4219 at full size. The pipeline only
# starts from given latents when denoise_strength < 1, which drops leading steps, so the second schedule carries
# one extra leading step that strength 0.75 removes (int(4 * 0.75) = 3 steps from 0.9094).
FIRST_PASS = TIMESTEPS[:7]
SECOND_PASS = [1000, 909.4, 725, 421.9]
SECOND_PASS_STRENGTH = 0.75


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
    edges or stride frames. Taller frames (the 1024x576 two-pass output) decode in two rows of 384 px stepped 288 (latent
    rows 0-12 and 9-18, a 96 px blend) with the same 24-frame tiles: 276 tile decodes for 15 s instead of 828."""
    if decode == "lean" and height <= 320:
        return {"tile_sample_min_height": height, "tile_sample_min_width": 256, "tile_sample_min_num_frames": 24,
                "tile_sample_stride_height": height, "tile_sample_stride_width": 192, "tile_sample_stride_num_frames": 16}
    if decode == "lean":
        return {"tile_sample_min_height": 384, "tile_sample_min_width": 256, "tile_sample_min_num_frames": 24,
                "tile_sample_stride_height": 288, "tile_sample_stride_width": 192, "tile_sample_stride_num_frames": 16}
    return {"tile_sample_min_height": 256, "tile_sample_min_width": 256, "tile_sample_min_num_frames": 16,
            "tile_sample_stride_height": 192, "tile_sample_stride_width": 192, "tile_sample_stride_num_frames": 8}


LATENT_CELL = 32  # pixels per latent cell (VAE spatial compression)


def background_cells(first, others, width, height, free_boxes=(), threshold=3.0, margin=1):
    """Latent cells (height//32 rows of width//32, 1 = hold) that look the same in the opening frame and in every
    keyframe: the still background of a locked camera. Cells that change between them, cells touching `free_boxes`
    (0-1 units: subjects that move in place without a keyframe showing it) and a `margin` of cells around both stay
    free to move."""
    from PIL import Image, ImageChops, ImageStat
    rows, cols = height // LATENT_CELL, width // LATENT_CELL
    with Image.open(first) as image:
        base = image.convert("L").resize((width, height))
    moving = set()
    for path in others:
        with Image.open(path) as image:
            diff = ImageChops.difference(base, image.convert("L").resize((width, height)))
        for r in range(rows):
            for c in range(cols):
                cell = (c * LATENT_CELL, r * LATENT_CELL, (c + 1) * LATENT_CELL, (r + 1) * LATENT_CELL)
                if ImageStat.Stat(diff.crop(cell)).mean[0] > threshold:
                    moving.add((r, c))
    for x0, y0, x1, y1 in free_boxes:
        moving.update((r, c) for r in range(rows) for c in range(cols)
                      if c * LATENT_CELL < x1 * width and (c + 1) * LATENT_CELL > x0 * width
                      and r * LATENT_CELL < y1 * height and (r + 1) * LATENT_CELL > y0 * height)
    near = range(-margin, margin + 1)
    return [[0 if any((r + dr, c + dc) in moving for dr in near for dc in near) else 1 for c in range(cols)]
            for r in range(rows)]


def held_tokens(cells, latent_frames):
    """Packed token positions (frame-major, then rows, then columns) of the held cells in every latent frame after
    the first, which the opening-frame condition already holds."""
    rows, cols = len(cells), len(cells[0])
    per_frame = [r * cols + c for r in range(rows) for c in range(cols) if cells[r][c]]
    return [f * rows * cols + i for f in range(1, latent_frames) for i in per_frame]


# Prompt tokens the T5 encoder reads: 128 is the diffusers default (about 90 words, the rest is cut off);
# the official LTX-Video pipeline encodes up to 256 (about 180 words).
def settings_for(duration, preset="preview", aspect="16:10", text_tokens=128, decode="lean", refine=False):
    """With `refine` the motion is generated at the preview size and the output is twice as wide and high (the
    official two-pass recipe); width and height are always the output size."""
    width, height = SIZES[(aspect, "quality" if preset == "quality" and not refine else "preview")]
    settings = {"width": width, "height": height, "frames": int(duration) * 24 + 1, "fps": 24,
                "timesteps": TIMESTEPS, "guidance_scale": 1.0, "decode_timestep": 0.05, "decode_noise_scale": 0.025,
                "max_sequence_length": text_tokens, "vae_tiling": vae_tiling(height, decode)}
    if refine:
        settings.update(width=width * 2, height=height * 2, vae_tiling=vae_tiling(height * 2, decode),
                        refine={"width": width, "height": height})
    return settings


def installed(root, source):
    from .skyreels import installed as lock_installed
    return lock_installed(root, source, LOCK)


def upscaler_installed(root, source):
    from .skyreels import installed as lock_installed
    return lock_installed(root, source, UPSCALER_LOCK)

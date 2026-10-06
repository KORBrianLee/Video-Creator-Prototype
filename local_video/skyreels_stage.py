"""SkyReels-V2 DF 1.3B generation on a CUDA GPU, run as an isolated stage process.

Input: request.json (with a "skyreels" settings block) and first-frame.png in the stage folder.
Output: frame-NNN.png files and torch-device.json, the same contract as Neo's decode stage, so the
safety check, encoding and verification are shared.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root  # noqa: E402
from local_video import accelerator, skyreels  # noqa: E402


def generate(root, output):
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    settings = request["skyreels"]
    if not accelerator.enable(root, "cuda"):
        raise RuntimeError("CUDA runtime folder is missing")
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingImageToVideoPipeline, UniPCMultistepScheduler
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available to PyTorch")
    torch.set_num_threads(min(8, max(1, request.get("threads", 4))))
    lock = json.loads((SOURCE / skyreels.LOCK).read_text(encoding="utf-8"))
    model = skyreels.folder(root, lock)
    started = time.monotonic()
    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=settings["flow_shift"])
    prompt = request["prompt"] + request.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours")
    negative = request.get("negative_prompt", "")
    # The UMT5-XXL text encoder (about 11 GB in BF16) runs once per prompt; on smaller GPUs it stays on the CPU.
    text_device = torch.device(settings["text_encoder_device"])
    pipe.text_encoder.to(text_device)
    with torch.inference_mode():
        prompt_embeds, negative_embeds = pipe.encode_prompt(prompt=prompt, negative_prompt=negative or None,
                                                            do_classifier_free_guidance=settings["guidance_scale"] > 1,
                                                            max_sequence_length=512, device=text_device, dtype=torch.bfloat16)
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    if settings["vae_tiling"] and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    image = Image.open(output / "first-frame.png").convert("RGB").resize((settings["width"], settings["height"]))
    generator = torch.Generator(device="cuda").manual_seed(int(request["seed"]))
    with torch.inference_mode():
        result = pipe(image=image, prompt_embeds=prompt_embeds.to("cuda"),
                      negative_prompt_embeds=negative_embeds.to("cuda") if negative_embeds is not None else None,
                      height=settings["height"], width=settings["width"], num_frames=settings["frames"],
                      num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                      overlap_history=settings["overlap_history"], addnoise_condition=settings["addnoise_condition"],
                      base_num_frames=settings["base_num_frames"], ar_step=settings["ar_step"], fps=settings["fps"],
                      generator=generator, output_type="pil")
    frames = result.frames[0]
    if len(frames) != settings["frames"]:
        raise RuntimeError(f"SkyReels returned {len(frames)} frames, expected {settings['frames']}")
    for index, frame in enumerate(frames):
        frame.save(output / f"frame-{index:03d}.png")
    (output / "torch-device.json").write_text(json.dumps({
        "stage": "skyreels_generate", "device": "cuda", "model": "skyreels-v2-df-1.3b", "weights": "resident",
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": round(time.monotonic() - started, 1),
        "settings": settings}), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["skyreels_generate"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    output = Path(args.output).resolve(strict=True)
    if not output.is_relative_to(root):
        raise ValueError("Inference data must stay inside the runtime folder")
    generate(root, output)

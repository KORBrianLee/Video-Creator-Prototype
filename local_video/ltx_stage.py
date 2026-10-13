"""LTX-Video 2B 0.9.8 distilled generation on a PyTorch GPU (XPU or CUDA), run as an isolated stage process.

Input: request.json (with an "ltx" settings block) and first-frame.png. Output: frame-NNN.png and
torch-device.json, the same contract as the other generators, so safety, encoding and verification are shared.

Weights stay BF16 and mapped on the CPU side; every Linear is widened to FP32 on the GPU per call
(the Iris Xe driver cannot run dtype-cast kernels, so all GPU math is FP32).
"""
from __future__ import annotations
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root  # noqa: E402
from local_video import accelerator, ltx  # noqa: E402


def fixed_schedule(scheduler, timesteps, torch):
    """Use the official distilled timesteps exactly, whatever sigmas the pipeline computes."""
    sigmas = [t / 1000.0 for t in timesteps] + [0.0]

    def set_timesteps(num_inference_steps=None, device=None, sigmas_argument=None, mu=None, timesteps=None, **extra):
        values = torch.tensor(sigmas, dtype=torch.float32)
        scheduler.sigmas = values.to(device) if device is not None else values
        scheduler.timesteps = (values[:-1] * 1000.0).to(device) if device is not None else values[:-1] * 1000.0
        scheduler.num_inference_steps = len(sigmas) - 1
        scheduler._step_index, scheduler._begin_index = None, None

    scheduler.set_timesteps = lambda num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None, **extra: \
        set_timesteps(num_inference_steps, device, sigmas, mu, timesteps)


def fp32_dtype(module, torch):
    """Streamed modules mix BF16 (CPU-side) and FP32 (GPU) parameters; report FP32 so pipelines never cast to BF16."""
    base = type(module)
    module.__class__ = type(f"FP32{base.__name__}", (base,), {"dtype": property(lambda self: torch.float32)})
    return module


def hashlib_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_rope(rope, torch):
    """Rotary tables are built on the CPU: `theta ** linspace` (pow) fails with a UR error on Iris Xe.

    The tables are small (one cos/sin row per token), so only the result is moved to the GPU.
    """
    forward = rope.forward

    def to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, (tuple, list)):
            return type(value)(to_cpu(item) for item in value)
        return value

    def cpu_forward(hidden_states, *args, **kwargs):
        device = hidden_states.device
        probe = torch.empty((hidden_states.shape[0], 0), dtype=torch.float32)
        cos, sin = forward(probe, *to_cpu(args), **{key: to_cpu(value) for key, value in kwargs.items()})
        return cos.to(device), sin.to(device)

    rope.forward = cpu_forward
    return rope


@contextlib.contextmanager
def generator_side_noise(torch):
    """The pipeline's decode noise calls torch.randn(device=gpu, generator=cpu), which torch rejects.

    The generator stays on the CPU because XPU random streams are not reproducible for a fixed seed;
    noise is drawn where the generator lives and then moved.
    """
    original = torch.randn

    def randn(*args, generator=None, device=None, **kwargs):
        if generator is not None and device is not None and torch.device(device).type != generator.device.type:
            return original(*args, generator=generator, device=generator.device, **kwargs).to(device)
        return original(*args, generator=generator, device=device, **kwargs)

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = original


def cpu_video_coords(pipe):
    """The condition pipeline casts integer token coordinates to float on the GPU (UR error on Iris Xe).

    Coordinates only feed the rotary tables, which are built on the CPU anyway (cpu_rope).
    """
    prepare = pipe.prepare_latents

    def prepare_latents(*args, **kwargs):
        latents, mask, coords, extra = prepare(*args, **kwargs)
        coords = coords.cpu().float()
        # Encoding the keyframes leaves VAE activations cached in shared GPU memory; the first transformer
        # step of an 8s clip then fails to upload its weights (UR_RESULT_ERROR_UNKNOWN).
        import gc
        gc.collect()
        module = getattr(__import__("torch"), latents.device.type, None)
        if module is not None and hasattr(module, "empty_cache"):
            module.synchronize()
            module.empty_cache()
        return latents, mask, coords, extra

    pipe.prepare_latents = prepare_latents
    cpu_scheduler_step(pipe.scheduler)
    return pipe


def hold_background(pipe, cells, strength, latent_frames, torch):
    """Locked camera: in every latent frame after the first, the held cells start from the opening frame kept still
    (encoded as a short still clip) and carry `strength` as their conditioning level, so the transformer treats them
    like the first-frame condition and the framing has nothing to drift with. Token edits run on the CPU (integer
    indexing on Iris Xe is not trusted)."""
    from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
    prepare = pipe.prepare_latents
    positions = ltx.held_tokens(cells, latent_frames)

    def prepare_latents(conditions, *args, **kwargs):
        latents, mask, coords, extra = prepare(conditions, *args, **kwargs)
        if not positions:
            return latents, mask, coords, extra
        still = conditions[0].repeat(1, 1, 17, 1, 1)
        held = retrieve_latents(pipe.vae.encode(still), sample_mode="argmax")
        held = pipe._normalize_latents(held, pipe.vae.latents_mean, pipe.vae.latents_std).float().cpu()
        packed = pipe._pack_latents(held[:, :, -1:].repeat(1, 1, latent_frames, 1, 1), 1, 1)
        index = torch.tensor(positions, dtype=torch.long)
        tokens, levels = latents.float().cpu(), mask.float().cpu()
        tokens[:, extra + index] = torch.lerp(tokens[:, extra + index], packed[:, index], strength)
        levels[:, extra + index] = torch.clamp(levels[:, extra + index], min=strength)
        import gc
        gc.collect()
        module = getattr(torch, latents.device.type, None)
        if module is not None and hasattr(module, "empty_cache"):
            module.synchronize()
            module.empty_cache()
        return tokens.to(latents.device, latents.dtype), levels.to(mask.device, mask.dtype), coords, extra

    pipe.prepare_latents = prepare_latents
    rows, cols = len(cells), len(cells[0])
    return {"cells": sum(map(sum, cells)), "of": rows * cols, "strength": strength}


def cpu_scheduler_step(scheduler):
    """Per-token keyframe steps multiply a bool mask by sigmas, a cast Iris Xe cannot run.

    The latents of one step are about 1 MB, so the whole Euler step runs on the CPU and only the
    result returns to the GPU.
    """
    step = scheduler.step

    def cpu_step(model_output, timestep, sample, *args, per_token_timesteps=None, **kwargs):
        if per_token_timesteps is None:
            return step(model_output, timestep, sample, *args, **kwargs)
        device = sample.device
        # The step looks up its index and sigmas in these tables; the denoise loop iterates its own copy.
        scheduler.sigmas, scheduler.timesteps = scheduler.sigmas.cpu(), scheduler.timesteps.cpu()
        moved = [value.cpu() if hasattr(value, "cpu") else value for value in (model_output, timestep, sample)]
        result = step(*moved, *args, per_token_timesteps=per_token_timesteps.cpu().float(), **kwargs)
        if isinstance(result, tuple):
            return tuple(item.to(device) if hasattr(item, "to") else item for item in result)
        result.prev_sample = result.prev_sample.to(device)
        return result

    scheduler.step = cpu_step
    return scheduler


def cpu_decoded_tiles(vae, torch):
    """Each decoded time tile goes to the CPU at once, so blending, concatenation and post-processing
    never hold the whole clip on the GPU (15 s at 512x320 is about 700 MB in FP32, which ended in
    UR_RESULT_ERROR_DEVICE_LOST on Iris Xe shared memory)."""
    from diffusers.models.autoencoders.vae import DecoderOutput
    tiled_decode = vae.tiled_decode

    def decode(z, temb, return_dict=True):
        sample = tiled_decode(z, temb, return_dict=True).sample
        module = getattr(torch, sample.device.type, None)
        sample = sample.cpu()
        if module is not None and hasattr(module, "empty_cache"):
            module.empty_cache()
        return DecoderOutput(sample=sample) if return_dict else (sample,)

    vae.tiled_decode = decode
    return vae


def open_gpu(root, plan):
    backend = plan.get("torch_device")
    if backend not in accelerator.BACKENDS or not accelerator.enable(root, backend):
        raise RuntimeError("LTX needs a PyTorch GPU runtime (XPU or CUDA)")
    import torch
    if not getattr(torch, backend).is_available():
        raise RuntimeError(f"{backend} GPU is not available to PyTorch")
    return torch, torch.device(backend)


def model_folder(root):
    lock = json.loads((SOURCE / ltx.LOCK).read_text(encoding="utf-8"))
    return root / "models" / lock["folder"]


def prompt_record(needed, limit):
    """How much of the prompt the model reads: tokens past `limit` are cut off without any error."""
    return {"tokens": needed, "limit": limit, "truncated": needed > limit}


def timing_record(load_start, pipe_start, step_ends, pipe_end):
    """Seconds for loading the weights, each denoising step (the first also prepares the conditions) and
    decoding, which is everything after the last step."""
    marks = [pipe_start, *step_ends]
    return {"load_seconds": round(pipe_start - load_start, 1),
            "denoise_step_seconds": [round(after - before, 1) for before, after in zip(marks, marks[1:])],
            "decode_seconds": round(pipe_end - marks[-1], 1)}


def text_cache_key(prompt, max_length, folder):
    """Same prompt, token budget and encoder: same embeddings, whatever the seed or the conditions."""
    import hashlib
    value = json.dumps({"prompt": prompt, "max_length": max_length, "encoder": Path(folder).name}, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encode_text(torch, device, folder, prompt, max_length, cache_dir=None):
    """T5-XXL encoder run once per prompt on the CPU (128 tokens, tens of seconds), then released.

    T5's relative-position buckets need integer/bool casts that the Iris Xe driver cannot run, and the
    encoder is used once per clip, so the CPU is the robust place for it. BF16 weights stay mapped and
    each Linear is widened to FP32 per call. Returns FP32 embeddings and mask on the GPU, and the number
    of tokens the whole prompt needs. With `cache_dir`, a retry of the same prompt (another seed or
    keyframe) reads them back instead of loading the 4.7B-parameter encoder again.
    """
    cached = Path(cache_dir) / f"{text_cache_key(prompt, max_length, folder)}.pt" if cache_dir else None
    if cached is not None and cached.is_file():
        try:
            stored = torch.load(cached)
            return stored["embeds"].to(device), stored["mask"].to(device), int(stored["needed"])
        except Exception:  # a cut-off cache file only costs the encoder run below
            cached.unlink(missing_ok=True)
    from local_video import readonly_weights
    readonly_weights.install(transformers=True)
    from transformers import T5EncoderModel, T5TokenizerFast
    from local_video.neodragon_quant import float_non_linear_parameters, stream_linears
    tokenizer = T5TokenizerFast.from_pretrained(folder / "tokenizer", local_files_only=True)
    needed = len(tokenizer(prompt, add_special_tokens=True).input_ids)
    tokens = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True,
                       add_special_tokens=True, return_tensors="pt")
    encoder = T5EncoderModel.from_pretrained(folder / "text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                             local_files_only=True).eval()
    stream_linears(encoder)
    float_non_linear_parameters(encoder)
    fp32_dtype(encoder, torch)
    with torch.inference_mode():
        mask = tokens.attention_mask.float()
        embeds = encoder(tokens.input_ids, attention_mask=mask).last_hidden_state.float()
    del encoder
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeds": embeds, "mask": mask, "needed": needed}, cached)
        except OSError:  # the clip does not depend on the cache
            pass
    # Masks are made FP32 on the CPU: integer-to-float casts on the GPU are what this driver cannot run.
    return embeds.to(device), mask.to(device), needed


def generate(root, output):
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    settings = request["ltx"]
    plan = request.get("device_plan", {})
    torch, device = open_gpu(root, plan)
    torch.set_num_threads(min(8, max(1, request.get("threads", 4))))
    started = time.monotonic()
    prompt = request["prompt"] + request.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours")
    prompt_embeds, mask, needed = encode_text(torch, device, model_folder(root), prompt, settings["max_sequence_length"],
                                              root / "cache" / "ltx-text")
    text_seconds = time.monotonic() - started
    frames, info = render(torch, device, root, output / "first-frame.png", settings, prompt_embeds, mask,
                          int(request["seed"]), bool(plan.get("torch_discrete")))
    for index, frame in enumerate(frames):
        frame.save(output / f"frame-{index:03d}.png")
    info["timing"] = {"text_seconds": round(text_seconds, 1), **info["timing"]}
    (output / "torch-device.json").write_text(json.dumps({
        "stage": "ltx_generate", "device": device.type, "model": "ltx-video-2b-0.9.8-distilled", **info,
        "prompt": prompt_record(needed, settings["max_sequence_length"]),
        "text_seconds": round(text_seconds, 1), "seconds": round(time.monotonic() - started, 1), "settings": settings}),
        encoding="utf-8")


def render(torch, device, root, first_frame, settings, prompt_embeds, mask, seed, discrete):
    """Transformer (streamed or resident by VRAM) and VAE in FP32; returns (PIL frames, info)."""
    from PIL import Image
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXImageToVideoPipeline,
                           LTXVideoTransformer3DModel)
    from local_video.neodragon_quant import float_non_linear_parameters, linear_weight_bytes, stream_convs, stream_linears
    from local_video import readonly_weights
    load_start = time.monotonic()
    readonly_weights.install()
    folder = model_folder(root)
    checkpoint = folder / "ltxv-2b-0.9.8-distilled.safetensors"
    configs = folder / "config-0.9.5"
    accelerator.bound_attention_memory(torch, discrete=discrete)
    gpu = getattr(torch, device.type)
    transformer = LTXVideoTransformer3DModel.from_single_file(str(checkpoint), config=str(configs), subfolder="transformer",
                                                              torch_dtype=torch.bfloat16, local_files_only=True).eval()
    resident = accelerator.weights_resident(linear_weight_bytes(transformer), discrete, accelerator.vram_free_bytes(torch, device))
    stream_linears(transformer, None, device, resident)
    float_non_linear_parameters(transformer, device)
    fp32_dtype(transformer, torch)
    if device.type != "cpu":
        cpu_rope(transformer.rope, torch)
    vae = AutoencoderKLLTXVideo.from_single_file(str(checkpoint), config=str(configs), subfolder="vae",
                                                 torch_dtype=torch.bfloat16, local_files_only=True).eval()
    # 2.3 GB BF16 (4.6 GB FP32): convolution and linear weights stay on the CPU side, widened per call.
    stream_convs(vae)
    stream_linears(vae, None, device)
    float_non_linear_parameters(vae, device)
    fp32_dtype(vae, torch)
    if hasattr(vae, "enable_tiling"):
        # Tiled spatially and in time: one whole-clip decode dispatch outlasts the Windows GPU watchdog on
        # Iris Xe (UR_RESULT_ERROR_DEVICE_LOST at 97 frames). See ltx.vae_tiling.
        vae.enable_tiling(**(settings.get("vae_tiling") or ltx.vae_tiling(settings["height"])))
        vae.use_framewise_decoding = True
        cpu_decoded_tiles(vae, torch)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(configs / "scheduler", local_files_only=True)
    fixed_schedule(scheduler, settings["timesteps"], torch)
    image = Image.open(first_frame).convert("RGB").resize((settings["width"], settings["height"]))
    anchor = settings.get("anchor_end")
    held = None
    if anchor:
        # The first frame is also a keyframe near the end, so camera and cast must return to the opening
        # composition within one generation (no clip joining).
        from diffusers import LTXConditionPipeline
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        pipe = LTXConditionPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
        cpu_video_coords(pipe)
        last = (settings["frames"] - 1) // 8 * 8
        end = image
        if settings.get("anchor_end_image"):
            end_path = Path(settings["anchor_end_image"]["path"])
            if hashlib_sha256(end_path) != settings["anchor_end_image"]["sha256"]:
                raise RuntimeError("anchor_end_image changed after the request")
            end = Image.open(end_path).convert("RGB").resize((settings["width"], settings["height"]))
        conditions = [LTXVideoCondition(image=image, frame_index=0, strength=1.0)]
        for key in settings.get("keyframes", []):
            key_path = Path(key["path"])
            if hashlib_sha256(key_path) != key["sha256"]:
                raise RuntimeError("a keyframe changed after the request")
            frame = max(8, round(key["at"] * (settings["frames"] - 1)) // 8 * 8)
            conditions.append(LTXVideoCondition(
                image=Image.open(key_path).convert("RGB").resize((settings["width"], settings["height"])),
                frame_index=min(frame, last - 8), strength=float(key["strength"])))
        conditions.append(LTXVideoCondition(image=end, frame_index=last, strength=float(anchor)))
        if settings.get("lock_background"):
            lock = settings["lock_background"]
            others = [key["path"] for key in settings.get("keyframes", [])]
            others += [settings["anchor_end_image"]["path"]] if settings.get("anchor_end_image") else []
            cells = ltx.background_cells(first_frame, others, settings["width"], settings["height"],
                                         lock.get("free_boxes", ()))
            held = hold_background(pipe, cells, float(lock["strength"]), (settings["frames"] - 1) // 8 + 1, torch)
        inputs = {"conditions": conditions,
                  "image_cond_noise_scale": float(settings.get("image_cond_noise_scale", 0.15))}
    else:
        pipe = LTXImageToVideoPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
        inputs = {"image": image}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    step_ends = []

    def mark_step(pipeline, index, timestep, tensors):
        # GPU work is queued asynchronously; without the wait, a step's time would leak into the next one.
        if hasattr(gpu, "synchronize"):
            gpu.synchronize()
        step_ends.append(time.monotonic())
        return tensors

    pipe_start = time.monotonic()
    with torch.inference_mode(), generator_side_noise(torch):
        result = pipe(**inputs, prompt_embeds=prompt_embeds, prompt_attention_mask=mask,
                      height=settings["height"], width=settings["width"], num_frames=settings["frames"],
                      frame_rate=settings["fps"], num_inference_steps=len(settings["timesteps"]),
                      guidance_scale=settings["guidance_scale"], decode_timestep=settings["decode_timestep"],
                      decode_noise_scale=settings["decode_noise_scale"], generator=generator, output_type="pil",
                      max_sequence_length=settings["max_sequence_length"], callback_on_step_end=mark_step)
    pipe_end = time.monotonic()
    frames = result.frames[0]
    if len(frames) != settings["frames"]:
        raise RuntimeError(f"LTX returned {len(frames)} frames, expected {settings['frames']}")
    return frames, {"weights": "resident" if resident else "streamed", "peak_gpu_allocated_bytes": gpu.max_memory_allocated(),
                    "anchor_end": anchor, "timing": timing_record(load_start, pipe_start, step_ends, pipe_end),
                    **({"held_background": held} if held else {})}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["ltx_generate"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    output = Path(args.output).resolve(strict=True)
    if not output.is_relative_to(root):
        raise ValueError("Inference data must stay inside the runtime folder")
    # After a device error the Level Zero teardown can hang forever, leaving a live process the parent
    # keeps waiting on, so the stage always leaves without interpreter shutdown.
    code = 0
    try:
        generate(root, output)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

"""Isolated CPU inference stages; official vendor sources remain unchanged."""
from __future__ import annotations
import gc, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside
from local_video import accelerator
def stage(root, output, name):
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    plan = request.get("device_plan", {})
    backend = plan.get("torch_device") if name in accelerator.GPU_STAGES else None
    if backend in accelerator.BACKENDS and not accelerator.enable(root, backend):
        backend = None
    import torch
    from PIL import Image
    torch.set_num_threads(min(8, max(1, request.get("threads", 4))))
    torch.set_num_interop_threads(1)
    torch.manual_seed(request["seed"])
    if backend != "cuda":
        assert torch.version.cuda is None, "The CPU and XPU stages require a PyTorch build without CUDA"
    available = backend is not None and getattr(torch, backend).is_available()
    device = torch.device(backend if available else "cpu")
    discrete = bool(plan.get("torch_discrete"))
    sys.path.insert(0, str(inside(root, "engines", "experimental-neodragon")))
    models = inside(root, "models", "experimental-neodragon")
    prompt = request["prompt"] + request.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours")
    width, height = request["width"], request["height"]
    def load_model(cls, folder, dtype, variant=None):
        if folder in {"ssd_1b_unet", "diffusion_transformer_320p"}:
            from local_video.neodragon_quant import load_mapped_model
            return load_mapped_model(cls, models / folder, dtype)
        arguments = {"torch_dtype": dtype, "local_files_only": True, "low_cpu_mem_usage": True}
        if variant:
            arguments["variant"] = variant
        return cls.from_pretrained(models / folder, **arguments).eval()

    def read_tensor(filename):
        return torch.load(output / filename, map_location="cpu", weights_only=True)

    if name in {"first_text", "first_latent"}:
        from diffusers import LCMScheduler
        from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
        scheduler = LCMScheduler(set_alpha_to_one=False, original_inference_steps=4, steps_offset=1)
    with torch.inference_mode():
        if name == "first_text":
            from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
            te1 = load_model(CLIPTextModel, "ssd_1b_text_encoder", torch.float16, "fp16")
            te2 = load_model(CLIPTextModelWithProjection, "ssd_1b_text_encoder_2", torch.float16, "fp16")
            pipe = SSD1B_FirstFrameGeneratorPipeline(
                vae=None, unet=None, text_encoder=te1, text_encoder_2=te2,
                tokenizer=CLIPTokenizer.from_pretrained(models / "ssd_1b_tokenizer", local_files_only=True),
                tokenizer_2=CLIPTokenizer.from_pretrained(models / "ssd_1b_tokenizer_2", local_files_only=True),
                scheduler=scheduler, add_watermarker=False)
            encoded = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"), do_classifier_free_guidance=False)
            torch.save({"prompt_embeds": encoded[0], "pooled_prompt_embeds": encoded[2]}, output / "first-text.pt")
        elif name == "first_latent":
            from diffusers import UNet2DConditionModel
            from local_video.neodragon_quant import quantize_linears
            unet = load_model(UNet2DConditionModel, "ssd_1b_unet", torch.bfloat16)
            print(json.dumps({"quantized_linear_count": quantize_linears(unet)}), flush=True)
            # Iris-era CPUs commonly lack native BF16 arithmetic. Keep INT8
            # matrices, but run convolutions and activations in FP32.
            unet.float()
            pipe = SSD1B_FirstFrameGeneratorPipeline(vae=None, unet=unet, text_encoder=None, text_encoder_2=None,
                                                       tokenizer=None, tokenizer_2=None, scheduler=scheduler, add_watermarker=False)
            embeds = read_tensor("first-text.pt")
            embeds = {key: value.float() for key, value in embeds.items()}
            result = pipe(prompt=None, **embeds, width=width, height=height, guidance_scale=0.0,
                          num_inference_steps=4, timesteps=[999, 749, 499, 249], output_type="latent",
                          generator=torch.Generator(device="cpu").manual_seed(request["seed"]))
            torch.save(result.images, output / "first-latent.pt")
        elif name == "first_decode":
            from diffusers import AutoencoderKL
            vae = load_model(AutoencoderKL, "ssd_1b_vae", torch.float32, "fp16")
            latent = read_tensor("first-latent.pt").float()
            image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
            from diffusers.image_processor import VaeImageProcessor
            VaeImageProcessor().postprocess(image, output_type="pil")[0].save(output / "first-frame.png")
        elif name == "video_text":
            from neodragon.text_encoder_bundle import TextEncoderBundle
            from neodragon.context_adapter import ContextAdapter
            from local_video.neodragon_quant import quantize_linears
            bundle = TextEncoderBundle.from_pretrained(str(models), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval()
            encoded = bundle(prompt, torch.device("cpu"))
            del bundle
            gc.collect()
            adapter = load_model(ContextAdapter, "context_adapter", torch.bfloat16)
            print(json.dumps({"quantized_linear_count": quantize_linears(adapter)}), flush=True)
            torch.save({"prompt_embeds": adapter(encoded[0]), "attention_mask": encoded[1], "pooled_prompt_embeds": encoded[2]}, output / "video-text.pt")
        elif name == "video_encode":
            import numpy as np
            from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
            vae = load_model(AsymmetricCausalVideoVAE, "causal_video_vae", torch.float32)
            image = Image.open(output / "first-frame.png").convert("RGB")
            values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].float() / 127.5 - 1.0
            latent = vae.encode(values).latent_dist.sample()
            torch.save(latent, output / "video-first-latent.pt")
        elif name == "video_pack":
            from neodragon.pyramid_mmdit import PyramidMMDiT
            from local_video.neodragon_quant import quantize_linears
            packed = inside(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.pt")
            packed.parent.mkdir(parents=True, exist_ok=True)
            notice = inside(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.provenance.json")
            source_sha = "ccc929881ee6a0474325998b2b80dcab199a39282ae773abf17a18c7fde0e31d"
            if packed.exists() and notice.exists():
                record = json.loads(notice.read_text(encoding="utf-8"))
                with packed.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if record.get("conversion") == "neodragon-cpu-per-channel-int8-v2" and record.get("source_sha256") == source_sha and record.get("sha256") == digest and record.get("torch") == torch.__version__:
                    print(json.dumps({"reused_derived_weights": True}), flush=True)
                    return
            dit = load_model(PyramidMMDiT, "diffusion_transformer_320p", torch.bfloat16)
            count = quantize_linears(dit)
            temporary = inside(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.partial")
            torch.save(dit.state_dict(), temporary)
            temporary.replace(packed)
            with packed.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            notice.write_text(json.dumps({"conversion": "neodragon-cpu-per-channel-int8-v2", "source_sha256": source_sha,
                "source_revision": "5bdc87a4895a4fd148ca14c03181345d278d1040", "sha256": digest,
                "torch": torch.__version__, "quantized_linear_count": count,
                "changes": "Linear weights quantized to INT8 with separate output-channel scales; other learned parameters retained.",
                "license": "BSD-3-Clause-Clear AND Qualcomm-Responsible-AI-License"}, indent=2), encoding="utf-8")
            print(json.dumps({"quantized_linear_count": count, "derived_bytes": packed.stat().st_size}), flush=True)
        elif name == "video_infer":
            from neodragon.pyramid_mmdit import PyramidMMDiT
            from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
            from neodragon.utils import generation_utils as gen
            from local_video.neodragon_quant import quantize_linears, stream_linears, float_non_linear_parameters
            if device.type == "xpu":
                accelerator.patch_vendor_for_fp32_gpu(torch)
            if device.type != "cpu":
                accelerator.bound_attention_memory(torch, discrete=discrete)
            gpu_engine = None
            weight_mode = "streamed"
            if device.type == "cpu" and request.get("device_plan", {}).get("gpu_inference"):
                from local_video.opencl_linear import LinearEngine
                gpu_engine = LinearEngine(request["device_plan"]["selected_device"]["id"],
                                          buffer_mib=request.get("gpu_buffer_mib", "auto"),
                                          reserve_gib=request.get("reserve_ram_gib", "auto"))
            if request.get("cpu_precision") == "bf16_stream":
                dit = load_model(PyramidMMDiT, "diffusion_transformer_320p", torch.bfloat16)
                from local_video.neodragon_quant import linear_weight_bytes
                resident = device.type != "cpu" and accelerator.weights_resident(
                    linear_weight_bytes(dit), discrete, accelerator.vram_free_bytes(torch, device))
                count = stream_linears(dit, gpu_engine, device, resident)
                float_non_linear_parameters(dit, None if device.type == "cpu" else device)
                dit.eval()
                weight_mode = "resident" if resident else "streamed"
                print(json.dumps({"mapped_exact_linear_count": count}), flush=True)
            else:
                packed = inside(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.pt")
                with torch.device("meta"):
                    dit = PyramidMMDiT.from_config(models / "diffusion_transformer_320p" / "config.json")
                count = quantize_linears(dit, packed_placeholder=True)
                state = torch.load(packed, map_location="cpu", weights_only=True, mmap=True)
                dit.load_state_dict(state, strict=True, assign=True)
                dit.eval().float()
                print(json.dumps({"loaded_int8_linear_count": count}), flush=True)
            encoded = read_tensor("video-text.pt")
            encoded = {key: (value.float() if value.is_floating_point() else value).to(device) for key, value in encoded.items()}
            first_latent = read_tensor("video-first-latent.pt").float().to(device)
            cfg = json.loads((models / "causal_video_vae" / "config.json").read_text(encoding="utf-8"))
            cached_vae = SimpleNamespace(config=SimpleNamespace(**cfg),
                encode=lambda image: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: first_latent)))
            cached_text = lambda *arguments: (encoded["prompt_embeds"], encoded["attention_mask"], encoded["pooled_prompt_embeds"])
            latents = gen.generate(cached_text, dit, torch.nn.Identity(), cached_vae,
                                   PyramidFlowMatchEulerDiscreteScheduler(), prompt=request["prompt"],
                                   image=Image.open(output / "first-frame.png"), width=width, height=height,
                                   num_frames=request["frames"], num_inference_steps=[1, 1, 1],
                                   video_num_inference_steps=[1, 1, 1], do_classifier_free_guidance=False,
                                   guidance_scale=0.0, video_guidance_scale=0.0, output_type="latent",
                                   device=device, dtype=torch.float32)
            latents = latents.cpu()
            assert torch.isfinite(latents).all(), "Generated latent contains non-finite values"
            torch.save(latents, output / "video-latent.pt")
            peak_gpu = getattr(torch, device.type).max_memory_allocated() if device.type != "cpu" else 0
            (output / "torch-device.json").write_text(json.dumps({"stage": name, "device": device.type, "weights": weight_mode,
                                                                  "peak_gpu_allocated_bytes": peak_gpu}), encoding="utf-8")
            if gpu_engine is not None:
                metrics = gpu_engine.metrics()
                if metrics["kernel_calls"] < 1:
                    raise RuntimeError("GPU 신경망 연산이 실행되지 않았습니다.")
                (output / "opencl-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(json.dumps({"actual_gpu_inference": metrics}), flush=True)
                gpu_engine.close()
        elif name == "video_decode":
            from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
            from neodragon.utils import generation_utils as gen
            if device.type != "cpu":
                accelerator.bound_attention_memory(torch, discrete=discrete)
            vae = load_model(AsymmetricCausalVideoVAE, "causal_video_vae", torch.float32).to(device)
            # The official decoder exposes equivalent causal traversal. Its
            # default parallel=True allocates all temporal activations at once.
            from types import MethodType
            from neodragon.asymmetric_causal_video_vae.decoder import apply_model_with_memblocks
            def sequential_decode(decoder, values):
                values = apply_model_with_memblocks(decoder.blocks, values.transpose(1, 2),
                                                     parallel=False, show_progress_bar=True)
                return values[:, decoder.frames_to_trim:].transpose(1, 2)
            vae.decoder.forward = MethodType(sequential_decode, vae.decoder)
            frames = gen._decode_latent(vae, read_tensor("video-latent.pt").float().to(device))
            assert len(frames) == request["frames"], "Decoded frame count mismatch"
            for index, frame in enumerate(frames):
                frame.save(output / f"frame-{index:03d}.png")
        elif name == "safety":
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPConfig, CLIPImageProcessor
            from neodragon.safety_checker import check_generated_video_safety
            path = inside(root, "models", "experimental-safety-checker")
            cfg = CLIPConfig.from_pretrained(path, local_files_only=True)
            from accelerate import init_empty_weights
            with init_empty_weights():
                checker = StableDiffusionSafetyChecker(cfg)
            weights = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            # Older Transformers serialized this now-nonpersistent buffer.
            # Verify its exact value, retaining the live positional IDs.
            old_ids = weights.pop("vision_model.vision_model.embeddings.position_ids", None)
            if old_ids is not None:
                assert torch.equal(old_ids, checker.vision_model.vision_model.embeddings.position_ids)
            checker.load_state_dict(weights, strict=True, assign=True)
            del weights
            checker.eval()
            extractor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
            frames = [Image.open(p).convert("RGB") for p in sorted(output.glob("frame-*.png"))]
            unsafe, flags = check_generated_video_safety(frames, checker, extractor, torch.device("cpu"), torch.float32, num_frames=3)
            (output / "safety.json").write_text(json.dumps({"unsafe": unsafe, "flags": flags}), encoding="utf-8")
            assert not unsafe, "Output rejected by the required safety checker"
        else:
            raise ValueError("Unknown CPU inference stage: " + name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    output = Path(args.output).resolve(strict=True)
    if not output.is_relative_to(root):
        raise ValueError("Inference data must stay inside the D runtime")
    stage(root, output, args.stage)

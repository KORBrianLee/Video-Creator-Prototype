---
name: local-video
description: Make low-resource local video drafts through this project's Cursor MCP server, with Intel integrated or optional discrete GPU inference and explicit quality limits.
---

# Local video

Design for the user's 16GB LG Gram with Intel Iris graphics and little free RAM. The current RTX computer may test compatibility only. Use the installed local-video MCP tools. Runtime files stay in the selected C/D SSD folder.

1. Call video_doctor once. Report the selected device, actual GPU scope and readiness. Default Neo streams BF16 video Linear matrices to OpenCL with 64MiB explicit buffers and 5GiB process working-set limit; attention, text, VAE and safety remain CPU. Auto prefers a discrete GPU, then integrated GPU. Never imply that encoding or a device flag proves neural GPU execution. Forced GPU failure must be visible. Do not raise limits or close unrelated applications.
2. Preserve the Korean script. Translate it once to a short English motion prompt, with a fixed composition, stable light and one modest action. Use a matching site_template from video_site_presets only when it matches the requested action. Otherwise use scenes and domain="construction". Neo prompts should stay under 75 CLIP content tokens including the modifier. Prepared photos and text are conditioning, not weight training.
3. Site realism has not passed. For the user's authorized prototype work explain the draft limitation and use diagnostic=true. Start with preview, continuous=true, duration_seconds=2. Preview is 384x256 with 49 real frames at 24fps. Quality is 512x320, same short duration. Four-second site videos have drift/darkening history; eight seconds is unverified. Do not join clips, interpolate or slow playback to imply natural continuity.
4. Reference images must be inside the chosen runtime folder. Keep attribution with a licensed photo. Neo GPU uses original bf16_stream; leave cpu_precision unspecified. CPU preview can use INT8. Wan 1.3B supports text only and rejects image_path. It needs substantially more RAM and is diagnostic, not the Gram default.
5. Use video_wait up to 60 seconds instead of frequent polling. Cancel with video_cancel. Show a completed MP4 only after required safety checks and full decoding succeed. Review worker count, bodies, PPE, boom joints, tracks, contact and camera motion. Frame variation and successful GPU kernels do not prove realism. Never bypass a rejected safety output.

Keep normal results compact; raw result.json contains device, kernel calls, peak explicit GPU buffers and phase RAM/time. These are current-host measurements, not Iris/Gram benchmarks. Read [resource and model evidence](../../../docs/LOW_RESOURCE_GPU.md), [site evidence](../../../docs/SITE_ADAPTATION.md) and [licenses](../../../docs/MODEL_LICENSES.md).

"""Compact job digests for the Cursor agent (token budget: one small JSON per finished job).

The local program measures and checks; the agent reads only this digest. Quality signals are computed
locally from sampled frames so the agent does not need to open images to notice a collapsing video.
They are warnings, not a quality pass: content review and user acceptance stay separate.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SAMPLES = 12
DIGEST_LIMIT = 2048
# Strongest local change at 96 px width: an LTX 15 s tower clip whose truck and walking worker barely moved measured
# 20, an 8 s clip whose crane load travelled across the bay 114. Sampling grain and compression alone stay below.
LITTLE_MOTION_P99 = 40
# Median share of the change from the first sample that a small zoom/shift of it explains, at 96 px width: an LTX
# tower clip whose framing kept alternating measured 0.59 and a 2 s tracking shot 0.33, while locked clips stayed at
# or below 0.06, including dump trucks driving at the camera and a receding cart (single samples up to 0.23).
CAMERA_MOVED_FIT = 0.25


def frame_signals(video, runtime_tmp, samples=SAMPLES):
    """Mean luma and colour saturation of evenly sampled frames, plus warnings for collapse/static clips."""
    import imageio_ffmpeg
    from PIL import Image, ImageStat
    video = Path(video)
    folder = Path(tempfile.mkdtemp(prefix="digest-", dir=runtime_tmp))
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video),
                        "-vf", f"select='not(mod(n\\,{max(1, count_frames(video) // samples)}))',scale=96:-1",
                        "-vsync", "0", str(folder / "s%03d.png")], capture_output=True, timeout=300, creationflags=flags)
        luma, saturation, grays = [], [], []
        for path in sorted(folder.glob("s*.png"))[:samples + 1]:
            with Image.open(path) as image:
                luma.append(round(ImageStat.Stat(image.convert("L")).mean[0], 1))
                saturation.append(round(ImageStat.Stat(image.convert("HSV")).mean[1], 1))
                grays.append(image.convert("L"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    motion = motion_signals(grays)
    return {"luma": luma, "saturation": saturation, "motion": motion,
            "warnings": quality_warnings(luma, saturation) + motion_warnings(motion)}


def camera_fit(first, later):
    """Share of the change from `first` to `later` that a small zoom and shift of `first` explains: high when the
    camera moved, near 0 when a subject moves in front of a still, textured background."""
    from PIL import Image, ImageChops, ImageStat
    w, h = first.size
    box = (10, 8, w - 10, h - 8)
    target = later.crop(box)
    base = ImageStat.Stat(ImageChops.difference(first.crop(box), target)).mean[0]
    if base <= 0.5:
        return 0.0
    best = base
    for scale in (0.92, 0.96, 1.0, 1.04, 1.08):
        a = 1 / scale
        for dy in range(-10, 11, 2):
            for dx in range(-8, 9, 2):
                if scale == 1.0 and dx == dy == 0:
                    continue
                moved = first.transform((w, h), Image.AFFINE, (a, 0, w / 2 + dx - a * w / 2, 0, a, h / 2 + dy - a * h / 2),
                                        resample=Image.BILINEAR)
                best = min(best, ImageStat.Stat(ImageChops.difference(moved.crop(box), target)).mean[0])
    return round(1 - best / base, 2)


def motion_signals(grays):
    """How much the picture changes: mean change between neighbouring samples (camera or large motion), the
    strongest local change from the first sample (the 99th percentile pixel: a subject moving in a still view) and
    the median camera fit (the camera itself moved)."""
    from PIL import ImageChops, ImageStat
    if len(grays) < 2:
        return {}
    steps = [ImageStat.Stat(ImageChops.difference(a, b)).mean[0] for a, b in zip(grays, grays[1:])]
    local = 0
    for later in grays[1:]:
        histogram = ImageChops.difference(grays[0], later).histogram()
        threshold, seen = sum(histogram) * 0.99, 0
        for value, count in enumerate(histogram):
            seen += count
            if seen >= threshold:
                local = max(local, value)
                break
    fits = sorted(camera_fit(grays[0], later) for later in grays[1:])
    return {"step_mean": round(sum(steps) / len(steps), 2), "step_max": round(max(steps), 2), "local_p99": local,
            "camera_fit": fits[len(fits) // 2]}


def motion_warnings(motion):
    """`little_motion` when no subject visibly moves, `camera_moved` when the view itself moved (a failure for a
    locked tower or tripod shot); thresholds from measured clips (see tests)."""
    warnings = []
    if motion and motion["local_p99"] < LITTLE_MOTION_P99:
        warnings.append(f"little_motion: p99 {motion['local_p99']}")
    if motion and motion.get("camera_fit", 0) >= CAMERA_MOVED_FIT:
        warnings.append(f"camera_moved: fit {motion['camera_fit']}")
    return warnings


def review_strip(video, target, box=(0.0, 0.0, 1.0, 1.0), start=0.0, end=None, count=8, columns=4, width=320):
    """Zoomed time strip for judging a key action: `count` evenly spaced frames between `start` and `end` seconds,
    cropped to `box` (0-1 units) and tiled with second labels, so a worker stepping out from behind an object or a
    short reverse can be checked on one image instead of the whole video."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    video, target = Path(video), Path(target)
    total, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    fps = total / seconds if seconds else 24
    end = seconds if end is None else min(end, seconds)
    times = [start + (end - start) * i / max(count - 1, 1) for i in range(count)]
    indices = sorted({min(total - 1, max(0, round(t * fps))) for t in times})
    folder = Path(tempfile.mkdtemp(prefix="strip-", dir=target.parent))
    tiles = []
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        select = "+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video),
                        "-vf", f"select='{select}'", "-vsync", "0", str(folder / "f%03d.png")],
                       capture_output=True, timeout=300, creationflags=flags)
        for path in sorted(folder.glob("f*.png")):
            with Image.open(path) as image:
                w, h = image.size
                crop = image.convert("RGB").crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
                tiles.append(crop.resize((width, max(1, round(crop.height * width / crop.width))), Image.LANCZOS))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if not tiles:
        raise RuntimeError("영상에서 프레임을 꺼내지 못했습니다.")
    columns = min(columns, len(tiles))
    tile_h = tiles[0].height + 16
    sheet = Image.new("RGB", (columns * width, (len(tiles) + columns - 1) // columns * tile_h), "black")
    draw = ImageDraw.Draw(sheet)
    for k, (tile, index) in enumerate(zip(tiles, indices)):
        x, y = k % columns * width, k // columns * tile_h
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile.height + 2), f"{index / fps:.1f}s", fill="white")
    sheet.save(target, quality=90)
    return {"path": str(target), "seconds": [round(i / fps, 1) for i in indices[:len(tiles)]]}


def count_frames(video):
    import imageio_ffmpeg
    try:
        return max(1, imageio_ffmpeg.count_frames_and_secs(str(video))[0])
    except Exception:
        return SAMPLES


def quality_warnings(luma, saturation):
    """Heuristics measured on real failures: Neo 15s went dark then blue noise; frozen clips barely change."""
    warnings = []
    if len(luma) < 3:
        return ["too_few_samples"]
    third = max(1, len(luma) // 3)
    start = sum(luma[:third]) / third
    # The lowest later frame catches dips that recover (Neo 8s fell from 93 to 49 mid-clip, then rose again).
    lowest = min(luma[third:])
    if start > 0 and lowest < start * 0.6:
        warnings.append(f"darkening: luma {start:.0f}->{lowest:.0f}")
    if any(value < 12 for value in luma[1:]):
        warnings.append("near_black_frames")
    sat_start, sat_end = sum(saturation[:third]) / third, sum(saturation[-third:]) / third
    if sat_start > 0 and sat_end > max(sat_start * 1.8, sat_start + 40):
        warnings.append(f"colour_blowout: saturation {sat_start:.0f}->{sat_end:.0f}")
    if max(luma) - min(luma) < 1.0 and max(saturation) - min(saturation) < 1.0:
        warnings.append("static_suspected")
    return warnings


def long_video_notes(phases):
    """Timing and prompt checks from a long-video stage: seconds for text, weight loading, the first denoising step
    (it also prepares the conditions), the mean of the rest and decoding, plus a warning when the prompt was cut."""
    timing, warnings = None, []
    for phase in phases:
        gpu = phase.get("gpu") or {}
        recorded = gpu.get("timing") or {}
        steps = recorded.get("denoise_step_seconds") or []
        if steps:
            rest = steps[1:] or steps
            timing = {"text": recorded.get("text_seconds"), "load": recorded.get("load_seconds"),
                      "first_step": steps[0], "per_step": round(sum(rest) / len(rest), 1), "steps": len(steps),
                      "decode": recorded.get("decode_seconds")}
        prompt = gpu.get("prompt") or {}
        if prompt.get("truncated"):
            warnings.append(f"prompt_truncated: {prompt['tokens']}>{prompt['limit']} tokens")
    return timing, warnings


def job_digest(job_id, with_signals=True):
    """Small summary of one job: outcome, timing per stage, memory peaks, GPU use and quality warnings."""
    from . import control
    state = control.status(job_id)
    _, root = control.load_config()
    digest = {"job": job_id, "state": state.get("state"), "stage": state.get("stage")}
    if state.get("error"):
        digest["error"] = str(state["error"])[:240]
    if state.get("started_at") and state.get("updated_at"):
        digest["seconds"] = round(state["updated_at"] - state["started_at"])
    result = state.get("result") or {}
    shots = result.get("shots") or []
    if shots:
        shot = shots[0]
        settings = shot.get("settings") or result.get("settings") or {}
        digest["video"] = {key: settings.get(key) for key in ("width", "height", "frames", "fps")}
        plan = shot.get("device_plan") or {}
        digest["model"] = plan.get("model_tier") or "neo"
        if plan.get("model_tier_fallback_reason"):
            digest["fallback"] = str(plan["model_tier_fallback_reason"]).split(". 자세한 기록")[0][:200]
        stages = {}
        for phase in shot.get("phases", []):
            if phase.get("cache_hit"):
                continue
            # [seconds, private resident GiB, commit GiB]; commit is what Windows actually limits.
            entry = [round(phase.get("elapsed_seconds") or 0)]
            for key in ("peak_private_resident_bytes", "peak_commit_bytes"):
                if phase.get(key):
                    entry.append(round(phase[key] / 2**30, 2))
            stages[phase.get("name")] = entry
            if (phase.get("gpu") or {}).get("backend"):
                digest["gpu"] = phase["gpu"]["backend"]
        digest["stages_s_gib"] = stages
        timing, notes = long_video_notes(shot.get("phases", []))
        if timing:
            digest["timing"] = timing
        for phase in shot.get("phases", []):
            for key in ("profile", "held_background"):
                if (phase.get("gpu") or {}).get(key):
                    digest[key] = phase["gpu"][key]
        digest["safety_unsafe"] = (shot.get("safety") or {}).get("unsafe")
        continuity = (shot.get("verification") or {}).get("continuity") or {}
        digest["continuity"] = continuity.get("state")
        video = Path(result.get("video_path", ""))
        if with_signals and video.is_file():
            signals = frame_signals(video, root / "tmp")
            digest["warnings"] = signals["warnings"] + notes
            digest["motion"] = signals["motion"]
            digest["luma"] = signals["luma"]
        elif notes:
            digest["warnings"] = notes
        digest["review_image"] = result.get("contact_sheet_path")
        digest["video_path"] = result.get("video_path")
    text = json.dumps(digest, ensure_ascii=False)
    if len(text) > DIGEST_LIMIT:
        digest.pop("luma", None)
    return digest

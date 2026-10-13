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


def compare_strips(videos, target, box=(0.0, 0.0, 1.0, 1.0), seconds=(0.0, 1.0, 2.0, 3.0), width=320, height=None):
    """One image comparing clips of the same scene made different ways: a row per clip with the same box (0-1 units,
    so clips of different resolution line up) at the same seconds, every tile scaled to one size."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    target = Path(target)
    rows = []
    for video in map(Path, videos):
        total, length = imageio_ffmpeg.count_frames_and_secs(str(video))
        fps = total / length if length else 24
        indices = [min(total - 1, max(0, round(s * fps))) for s in seconds]
        folder = Path(tempfile.mkdtemp(prefix="compare-", dir=target.parent))
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            select = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video),
                            "-vf", f"select='{select}'", "-vsync", "0", str(folder / "c%03d.png")],
                           capture_output=True, timeout=300, creationflags=flags)
            frames = {}
            for index, path in zip(sorted(set(indices)), sorted(folder.glob("c*.png"))):
                with Image.open(path) as image:
                    w, h = image.size
                    frames[index] = image.convert("RGB").crop((round(box[0] * w), round(box[1] * h),
                                                              round(box[2] * w), round(box[3] * h)))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        rows.append((video, [frames.get(i) for i in indices]))
    first = next(tile for _, tiles in rows for tile in tiles if tile is not None)
    height = height or max(1, round(first.height * width / first.width))
    label = 14
    sheet = Image.new("RGB", (len(seconds) * width, len(rows) * (height + label)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, (video, tiles) in enumerate(rows):
        y = r * (height + label)
        name = video.parent.parent.name if video.name == "video.mp4" else video.stem
        for c, tile in enumerate(tiles):
            if tile is not None:
                sheet.paste(tile.resize((width, height), Image.LANCZOS), (c * width, y + label))
            draw.text((c * width + 4, y + 1), f"{name} {seconds[c]:.1f}s", fill="white")
    sheet.save(target, quality=90)
    return {"path": str(target), "rows": [str(video) for video, _ in rows]}


def vest_boxes(image, scan_width=512, largest=0.015, smallest=(0.08, 0.28)):
    """Boxes (0-1 units) around workers found by their high-visibility vests: fluorescent lime or strongly saturated
    orange, grouped into blobs. Blobs larger than `largest` of the frame (an orange truck, a yellow machine), much
    wider than tall or thin as a pole are dropped; each vest box is grown up for the head and down for the legs, to at
    least `smallest` (width, height) of the frame so a vest fragment still shows the whole worker."""
    # A distant worker's vest is a few pixels across: shrinking the frame blends it into the jacket below the threshold.
    scan_width = min(scan_width, image.width)
    scan = image.convert("RGB").resize((scan_width, max(1, round(image.height * scan_width / image.width))))
    hsv = scan.convert("HSV").load()
    width, height = scan.size
    vest = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            hue, sat, val = hsv[x, y]
            # Safety orange sits at hue 10-22 (PIL 0-255); mustard and yellow machine paint at 28-36.
            vest[y][x] = (45 <= hue < 62 and sat > 110 and val > 170) or (8 <= hue < 23 and sat > 170 and val > 150)
    seen, boxes = set(), []
    for y in range(height):
        for x in range(width):
            if not vest[y][x] or (x, y) in seen:
                continue
            stack, blob = [(x, y)], []
            seen.add((x, y))
            while stack:
                px, py = stack.pop()
                blob.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and vest[ny][nx] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs, ys = [p[0] for p in blob], [p[1] for p in blob]
            w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
            if len(blob) < 3 or len(blob) > largest * width * height or w > 1.6 * h + 2 or h > 3 * w + 2:
                continue
            x0, y0 = max(0, min(xs) - 0.6 * w) / width, max(0, min(ys) - 1.4 * h) / height
            x1, y1 = min(width, max(xs) + 1 + 0.6 * w) / width, min(height, max(ys) + 1 + 2.8 * h) / height
            grow_x, grow_y = max(0, smallest[0] - (x1 - x0)) / 2, max(0, smallest[1] - (y1 - y0)) / 2
            boxes.append((max(0, x0 - grow_x), max(0, y0 - grow_y), min(1, x1 + grow_x), min(1, y1 + grow_y)))
    # Reflective stripes split one vest into several blobs: overlapping person boxes are one worker.
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(boxes)


def moments(video, count, scratch):
    """(seconds, RGB frame at native size) at `count` evenly spaced moments, from one ffmpeg pass."""
    import imageio_ffmpeg
    from PIL import Image
    total, length = imageio_ffmpeg.count_frames_and_secs(str(video))
    fps = total / length if length else 24
    indices = sorted({min(total - 1, round(i * (total - 1) / max(count - 1, 1))) for i in range(count)})
    folder = Path(tempfile.mkdtemp(prefix="moments-", dir=scratch))
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        select = "+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video),
                        "-vf", f"select='{select}'", "-vsync", "0", str(folder / "m%03d.png")],
                       capture_output=True, timeout=300, creationflags=flags)
        frames = []
        for index, path in zip(indices, sorted(folder.glob("m*.png"))):
            with Image.open(path) as frame:
                frames.append((index / fps, frame.convert("RGB")))
        return frames
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def people_counts(video, scratch, count=8):
    """Workers found by vest colour at `count` evenly spaced moments. A count that drops mid-clip or differs from
    the script points at a vanished, half-cut or added worker; cones and lime-lit machines can add one."""
    return [len(vest_boxes(frame)) for _, frame in moments(video, count, scratch)]


def people_sheet(video, target, count=6, tile_height=160, per_frame=6):
    """Native-resolution crops of every worker found by vest colour at `count` evenly spaced moments: a row per
    moment, labelled with seconds, so smeared bodies, vanishing props or extra people are seen without guessing
    coordinates. Machines painted orange or lime can still show up; the person check stays visual."""
    from PIL import Image, ImageDraw
    target = Path(target)
    rows = []
    for second, frame in moments(video, count, target.parent):
        crops = []
        for x0, y0, x1, y1 in vest_boxes(frame)[:per_frame]:
            crop = frame.crop((round(x0 * frame.width), round(y0 * frame.height),
                               round(x1 * frame.width), round(y1 * frame.height)))
            scale = tile_height / max(1, crop.height)
            crops.append(crop.resize((max(1, round(crop.width * scale)), tile_height), Image.NEAREST))
        rows.append((second, crops))
    label = 14
    sheet_width = max([sum(c.width + 4 for c in crops) for _, crops in rows] + [240])
    sheet = Image.new("RGB", (sheet_width, len(rows) * (tile_height + label)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, (second, crops) in enumerate(rows):
        y = r * (tile_height + label)
        draw.text((4, y + 1), f"{second:.1f}s  {len(crops)} found", fill="white")
        x = 0
        for crop in crops:
            sheet.paste(crop, (x, y + label))
            x += crop.width + 4
    sheet.save(target, quality=92)
    return {"path": str(target), "found": {f"{second:.1f}": len(crops) for second, crops in rows}}


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
            for key in ("profile", "held_background", "held_spans", "refine"):
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
            digest["people"] = people_counts(video, root / "tmp")
            digest["luma"] = signals["luma"]
        elif notes:
            digest["warnings"] = notes
        digest["review_image"] = result.get("contact_sheet_path")
        digest["video_path"] = result.get("video_path")
    text = json.dumps(digest, ensure_ascii=False)
    if len(text) > DIGEST_LIMIT:
        digest.pop("luma", None)
    return digest

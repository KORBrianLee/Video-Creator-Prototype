"""First-frame candidates for one scene: N seeds, one labelled contact sheet, a tiny JSON summary.

The agent looks at a single sheet and passes the chosen frame to a plan scene as image_path, so
seed selection costs one first-frame render per seed instead of one full video per seed.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time

MAX_SEEDS = 6


def contact_sheet(frames, labels, destination, columns=3, width=320):
    from PIL import Image, ImageDraw
    images = [Image.open(path).convert("RGB") for path in frames]
    try:
        height = round(images[0].height * width / images[0].width)
        rows = -(-len(images) // columns)
        sheet = Image.new("RGB", (columns * width, rows * height), "black")
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(zip(images, labels)):
            x, y = index % columns * width, index // columns * height
            sheet.paste(image.resize((width, height)), (x, y))
            draw.rectangle((x, y, x + 9 * len(label) + 8, y + 18), fill="black")
            draw.text((x + 4, y + 3), label, fill="yellow")
        sheet.save(destination)
    finally:
        for image in images:
            image.close()


def pick(spec, control, neodragon, report_dir, clock=time.monotonic):
    seeds = [int(seed) for seed in spec["seeds"]][:MAX_SEEDS]
    if not seeds:
        raise ValueError("seeds가 필요합니다.")
    scene = {key: spec[key] for key in ("prompt", "negative_prompt", "first_frame_prompt") if key in spec}
    request = control.normalize({"title": spec.get("name", "frames"), "preset": spec.get("preset", "preview"),
                                 "duration_seconds": spec.get("duration_seconds", 2), "seed": seeds[0],
                                 "diagnostic": True, "scenes": [scene]})
    settings = neodragon.settings_for(request)
    root = Path(request["_runtime_dir"]).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    frames, results = [], []
    started = clock()
    for seed in seeds:
        with tempfile.TemporaryDirectory(dir=root / "tmp") as work:
            work = Path(work)
            stage = neodragon.first_frame(root, work, {**request["scenes"][0], "seed": seed}, settings, request,
                                          lambda _event: None, lambda: False)
            target = report_dir / f"seed-{seed}.png"
            shutil.copy2(work / "first-frame.png", target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        frames.append(target)
        results.append({"seed": seed, "path": str(target), "sha256": digest,
                        "seconds": round(stage.get("elapsed_seconds", 0)), "cache_hit": bool(stage.get("cache_hit"))})
    contact_sheet(frames, [f"seed {seed}" for seed in seeds], report_dir / "sheet.png")
    summary = {"name": spec.get("name"), "sheet": str(report_dir / "sheet.png"), "size": [settings["width"], settings["height"]],
               "minutes": round((clock() - started) / 60, 1), "frames": results}
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return summary


IMPORT_SIZE = (1024, 640)  # 16:10, the LTX 512x320 / 768x448 family; stages resize from here


def import_frames(sources, root, size=IMPORT_SIZE):
    """Move images made outside the runtime (e.g. Cursor's GenerateImage, which saves on C) into
    <runtime>/assets/first-frames, centre-cropped to the video aspect; the originals are deleted so
    nothing stays on the system drive. Returns the D-side paths to use as scene image_path."""
    from PIL import Image
    target_dir = Path(root) / "assets" / "first-frames"
    target_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    for source in map(Path, sources):
        with Image.open(source) as image:
            image = image.convert("RGB")
            ratio = size[0] / size[1]
            width, height = image.size
            if width / height > ratio:
                crop = round(height * ratio)
                box = ((width - crop) // 2, 0, (width - crop) // 2 + crop, height)
            else:
                crop = round(width / ratio)
                box = (0, (height - crop) // 2, width, (height - crop) // 2 + crop)
            target = target_dir / (source.stem + ".png")
            image.crop(box).resize(size, Image.LANCZOS).save(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source.unlink()
        imported.append({"path": str(target), "sha256": digest, "size": list(size)})
    return {"imported": imported}


def main(spec_file):
    from . import control, neodragon
    spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    _, root = control.load_config()
    report = root / "reports" / f'{spec.get("name", "frames")}-{time.strftime("%Y%m%d-%H%M%S")}'
    return pick(spec, control, neodragon, report)

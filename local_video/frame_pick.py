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


def compose_keyframe(start, end, keep, target, feather=24):
    """End keyframe whose static parts are pixel-identical to the start frame.

    Image editing models redraw the whole frame, so a person or a background that should not move
    shifts slightly, and the video model turns that shift into a camera move. `keep` is a list of
    (x0, y0, x1, y1) boxes in 0-1 units copied from `start` onto `end` with a soft edge."""
    from PIL import Image, ImageDraw, ImageFilter
    with Image.open(start) as first, Image.open(end) as last:
        first, last = first.convert("RGB"), last.convert("RGB").resize(first.size, Image.LANCZOS)
        mask = Image.new("L", first.size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = first.size
        for x0, y0, x1, y1 in keep:
            draw.rectangle((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        Image.composite(first, last, mask).save(target)
    with Path(target).open("rb") as stream:
        return {"path": str(target), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def fill_rows(mask, start=0.0):
    """Close holes in an upright figure: each row from `start` (0-1 of the height) down is filled
    between its first and last set pixel. Dark clothing over dark ground leaves gaps in a difference
    mask, and a see-through figure in a keyframe turns into a smear in the video. Rows above `start`
    are left alone so the gap between a raised arm and the head is not filled with background."""
    width, height = mask.size
    pixels = mask.load()
    for y in range(round(start * height), height):
        row = [x for x in range(width) if pixels[x, y] > 127]
        if row:
            for x in range(row[0], row[-1] + 1):
                pixels[x, y] = 255
    return mask


def ppe_mask(patch):
    """Silhouette of a worker in high-visibility PPE without a clean plate: saturated orange or
    yellow vest, bright white helmet and dark work clothes, against grey sky, buildings, gravel and
    grass. A generated clean plate never matches the real background pixel for pixel, so a
    difference mask also picks up building edges around the worker."""
    from PIL import Image
    hsv = patch.convert("HSV")
    gray = patch.convert("L")
    mask = Image.new("L", patch.size, 0)
    out, h, g = mask.load(), hsv.load(), gray.load()
    for y in range(patch.height):
        for x in range(patch.width):
            hue, sat, val = h[x, y]
            vest = sat > 120 and val > 110 and (hue < 45 or hue > 240)
            helmet = g[x, y] > 225 and sat < 40
            clothes = g[x, y] < 55
            if vest or helmet or clothes:
                out[x, y] = 255
    return mask


def paste_region(source, box, base, at, target, scale=1.0, feather=6, clean=None, threshold=28, solid=False,
                 ppe=False):
    """Place a person or object cut from `source` (box in 0-1 units) onto `base` with its top-left
    corner at `at` (0-1 units), optionally scaled. Image editing models cannot move a person to a
    stated position reliably; pasting the same pixels keeps identity and gives exact placement.
    `base` should be a clean plate (the scene without that person) to avoid a double. With `clean`
    (the source scene without the person) only pixels that differ from it are pasted, so the
    person's surroundings do not come along as a rectangle."""
    from PIL import Image, ImageChops, ImageFilter
    with Image.open(source) as src, Image.open(base) as plate:
        src, plate = src.convert("RGB"), plate.convert("RGB").resize(src.size, Image.LANCZOS)
        width, height = src.size
        x0, y0, x1, y1 = (round(box[0] * width), round(box[1] * height), round(box[2] * width), round(box[3] * height))
        patch = src.crop((x0, y0, x1, y1))
        mask = Image.new("L", patch.size, 0)
        inner = Image.new("L", (max(1, patch.width - 2 * feather), max(1, patch.height - 2 * feather)), 255)
        mask.paste(inner, (feather, feather))
        mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
        if clean is not None or ppe:
            if ppe:
                shape = ppe_mask(patch).filter(ImageFilter.MaxFilter(3))
            else:
                with Image.open(clean) as empty:
                    empty = empty.convert("RGB").resize(src.size, Image.LANCZOS).crop((x0, y0, x1, y1))
                shape = ImageChops.difference(patch, empty).convert("L").point(lambda v: 255 if v > threshold else 0)
                shape = shape.filter(ImageFilter.MaxFilter(5))
            if solid is not False and solid is not None:
                shape = fill_rows(shape, 0.0 if solid is True else float(solid))
            shape = shape.filter(ImageFilter.GaussianBlur(1.5))
            mask = ImageChops.multiply(mask, shape)
        if scale != 1.0:
            size = (max(1, round(patch.width * scale)), max(1, round(patch.height * scale)))
            patch, mask = patch.resize(size, Image.LANCZOS), mask.resize(size, Image.LANCZOS)
        plate.paste(patch, (round(at[0] * width), round(at[1] * height)), mask)
        plate.save(target)
    with Path(target).open("rb") as stream:
        return {"path": str(target), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def main(spec_file):
    from . import control, neodragon
    spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    _, root = control.load_config()
    report = root / "reports" / f'{spec.get("name", "frames")}-{time.strftime("%Y%m%d-%H%M%S")}'
    return pick(spec, control, neodragon, report)

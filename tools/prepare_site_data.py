"""Explicit small licensed-reference preparation; generation stays offline.

Originals, attribution and derivatives are kept together on D. This script does
not train weights and does not claim the references improve the video model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside
from local_video.control import write_json

ALLOWED = {"CC BY 4.0": "https://creativecommons.org/licenses/by/4.0",
           "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0"}
MAX_FILE_BYTES = 40 * 2**20


def validate_info(item, info):
    metadata = info["extmetadata"]
    name = metadata["LicenseShortName"]["value"]
    url = metadata["LicenseUrl"]["value"].rstrip("/")
    if name not in ALLOWED or name != item["license"] or url != ALLOWED[name]:
        raise ValueError("Reference license changed or is not supported: " + item["id"])
    if info["sha1"] != item["sha1"] or info["size"] != item["bytes"] or not 0 < info["size"] <= MAX_FILE_BYTES:
        raise ValueError("Reference file identity changed: " + item["id"])
    parsed = urllib.parse.urlparse(info["url"])
    if parsed.scheme != "https" or parsed.netloc != "upload.wikimedia.org":
        raise ValueError("Reference download must be an original Wikimedia file")
    return urllib.parse.urlunparse(parsed._replace(query=""))


def file_hash(path, name="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, name).hexdigest()


def metadata(title):
    query = urllib.parse.urlencode({"action": "query", "format": "json", "titles": title,
        "prop": "imageinfo", "iiprop": "url|size|sha1|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Attribution|UsageTerms|ImageDescription"})
    request = urllib.request.Request("https://commons.wikimedia.org/w/api.php?"+query,
        headers={"User-Agent": "CursorVideoLocal/0.6 (local licensed reference preparation)"})
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = response.read(2**20+1)
    if len(payload) > 2**20:
        raise ValueError("Reference metadata is unexpectedly large")
    data = json.loads(payload)
    return next(iter(data["query"]["pages"].values()))["imageinfo"][0]


def fetch(url, path, item):
    if path.is_file() and path.stat().st_size == item["bytes"] and file_hash(path, "sha1") == item["sha1"]:
        return
    pending = path.with_suffix(".webm.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "CursorVideoLocal/0.6"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, pending.open("wb") as stream:
            if urllib.parse.urlparse(response.url).netloc != "upload.wikimedia.org":
                raise ValueError("Unexpected reference redirect")
            size = 0
            while block := response.read(2**20):
                size += len(block)
                if size > item["bytes"]:
                    raise ValueError("Reference exceeds the pinned download size")
                stream.write(block)
        if pending.stat().st_size != item["bytes"] or file_hash(pending, "sha1") != item["sha1"]:
            raise ValueError("Reference checksum mismatch")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def process(ffmpeg, command, log):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    with log.open("wb") as stream:
        result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-threads", "2", *command],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, creationflags=flags, timeout=60)
    if result.returncode:
        raise RuntimeError("Reference preparation failed: " + str(log))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", default=__import__("local_video.storage", fromlist=["default_runtime"]).default_runtime())
    parser.add_argument("--assets", nargs="+", help="선택한 source id만 준비. 생략 시 고정된 자료 전체.")
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    lock = json.loads((SOURCE / "site-data.lock.json").read_text(encoding="utf-8"))
    base = inside(root, "datasets", "construction-v1")
    base.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records, samples, unavailable = [], [], []
    for item in lock["assets"]:
        if args.assets and item["id"] not in args.assets:
            continue
        folder = inside(root, "datasets", "construction-v1", item["id"])
        folder.mkdir(exist_ok=True)
        saved_metadata = folder / "source-metadata.json"
        original = inside(root, "datasets", "construction-v1", item["id"], "original.webm")
        if saved_metadata.is_file() and original.is_file() and file_hash(original, "sha1") == item["sha1"]:
            info = json.loads(saved_metadata.read_text(encoding="utf-8"))
        else:
            info = metadata(item["title"])
        url = validate_info(item, info)
        write_json(folder / "source-metadata.json", info)
        try:
            fetch(url, original, item)
        except (urllib.error.URLError, TimeoutError) as exc:
            # Preserve prepared references and explicitly record unavailable
            # originals; do not hammer a rate-limited source or invent a clip.
            unavailable.append({"id": item["id"], "error": str(exc), "state": "not_prepared"})
            print(json.dumps(unavailable[-1], ensure_ascii=False), flush=True)
            continue
        attribution = {"title": item["title"], "author": item["author"], "source_url": info["descriptionurl"],
            "original_url": url, "license": item["license"], "license_url": item["license_url"],
            "original_sha256": file_hash(original), "changes": "4-second segments; 512x320 letterbox; 24fps; audio removed; first frames extracted",
            "share_alike_required_for_distributed_derivatives": item["license"] == "CC BY-SA 4.0",
            "no_endorsement": True}
        write_json(folder / "ATTRIBUTION.json", attribution)
        for start in item["starts"]:
            name = f"segment-{start:02d}"
            clip, frame = folder / (name+".mp4"), folder / (name+".png")
            process(ffmpeg, ["-y", "-ss", str(start), "-i", str(original), "-an", "-map", "0:v:0",
                "-vf", "fps=24,scale=512:320:force_original_aspect_ratio=decrease,pad=512:320:(ow-iw)/2:(oh-ih)/2",
                "-frames:v", "97", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(clip)], folder / (name+".log"))
            process(ffmpeg, ["-y", "-i", str(clip), "-frames:v", "1", str(frame)], folder / (name+"-frame.log"))
            record = {"id": item["id"]+"-"+str(start), "source_id": item["id"], "source_start_seconds": start,
                "video": str(clip), "first_frame": str(frame), "video_sha256": file_hash(clip), "first_frame_sha256": file_hash(frame),
                "frames": 97, "fps": 24, "caption": item["caption"], "caption_status": "source_description_only_visual_review_required",
                "license": item["license"], "attribution": str(folder / "ATTRIBUTION.json"), "weight_training_used": False}
            write_json(folder / (name+".json"), record)
            samples.append(record)
        records.append(attribution)
        print(json.dumps({"prepared": item["id"], "license": item["license"], "original_bytes": item["bytes"]}, ensure_ascii=False), flush=True)
    result = {"schema": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "state": "partial_references_prepared_weights_not_trained" if unavailable else "references_prepared_weights_not_trained",
        "weights_finetuned": False, "neural_training_eligibility": "not_assessed_no_training_run", "asset_count": len(records),
        "unavailable": unavailable,
        "samples": samples, "sources": records, "preparation_note": "Reference frame-rate normalization can repeat source frames. These are reference clips, never claimed as newly generated video.",
        "redistribution": "Preserve each source attribution and license; CC BY-SA derivatives keep share-alike terms. Model licenses remain separate."}
    write_json(base / "manifest.json", result)
    (base / "ATTRIBUTION.md").write_text("\n\n".join(
        f"{x['title']} — {x['author']}\nSource: {x['source_url']}\nLicense: {x['license']} ({x['license_url']})\nChanges: {x['changes']}"
        for x in records), encoding="utf-8")
    print(json.dumps({"manifest": str(base / "manifest.json"), "asset_count": len(records), "sample_count": len(samples), "weights_finetuned": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

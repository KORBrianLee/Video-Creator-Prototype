"""Read official package/model metadata; no model download or execution."""
import json
import urllib.request
import sys
sys.stdout.reconfigure(encoding="utf-8")

URLS = {
    "sdcpp_release": "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest",
    "wan_gguf": "https://huggingface.co/api/models/calcuis/wan-1.3b-gguf?blobs=true",
    "umt5_gguf": "https://huggingface.co/api/models/city96/umt5-xxl-encoder-gguf?blobs=true",
    "wan_components": "https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged?blobs=true",
    "ltx_license": "https://huggingface.co/Lightricks/LTX-Video/raw/main/LTX-Video-Open-Weights-License-0.X.txt",
    "ltx_old_license": "https://huggingface.co/Lightricks/LTX-Video/raw/main/LICENSE",
    "pillow": "https://pypi.org/pypi/Pillow/json",
    "imageio_ffmpeg": "https://pypi.org/pypi/imageio-ffmpeg/json",
    "sd15_gguf": "https://huggingface.co/api/models/gpustack/stable-diffusion-v1-5-GGUF?blobs=true",
    "ad_lightning": "https://huggingface.co/api/models/ByteDance/AnimateDiff-Lightning?blobs=true",
}

for name, url in URLS.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CursorVideoLocal/0.2-source-audit"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        if "license" in name:
            print(json.dumps({"source": name, "text": raw.decode("utf-8")}, ensure_ascii=False))
            continue
        data = json.loads(raw)
        if name == "sdcpp_release":
            result = {"tag": data["tag_name"], "published_at": data["published_at"], "assets": [{"name": x["name"], "size": x["size"], "url": x["browser_download_url"], "digest": x.get("digest")} for x in data["assets"] if "win" in x["name"].lower()]}
        elif "info" in data:
            result = {"version": data["info"]["version"]}
        else:
            result = {"sha": data["sha"], "card": data.get("cardData", {}), "files": [{"name": x["rfilename"], "size": x.get("size"), "sha256": x.get("lfs", {}).get("sha256")} for x in data.get("siblings", []) if any(s in x["rfilename"].lower() for s in ["q4_k", "q4_0", "q3_k_s", "4step_diffusers", "4step_comfyui", "vae.safetensors", "license", "readme"])]}
        print(json.dumps({"source": name, "data": result}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"source": name, "error": str(e)}, ensure_ascii=False))

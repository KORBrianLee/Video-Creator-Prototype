"""Fetch only public metadata and small configuration/license files; never weights."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import urllib.parse

ROOT = Path(r"D:\CursorVideoLocal\audits\model-survey-2026-10-01")
REPOS = ["BAAI/nova-d48w1024-osp480", "Lightricks/LTX-Video-2B-0.9.6-Distilled-04-25",
         "Qualcomm-AI-Research/Neodragon", "Qualcomm-AI-Research/mobilewan",
         "Motif-Technologies/Motif-Video-2B", "THUDM/CogVideoX-2b",
         "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "FastVideo/FastWan-QAD-1.3B",
         "Efficient-Large-Model/SANA-Video_2B_480p_LongLive"]

def read(url):
    req = urllib.request.Request(url, headers={"User-Agent": "CursorVideoLocal-model-survey/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(2*1024*1024+1)
    if len(raw) > 2*1024*1024:
        raise ValueError("Metadata limit exceeded")
    return raw

def inspect(repo):
    folder = ROOT / repo.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        raw = read("https://huggingface.co/api/models/"+repo+"?blobs=true")
        info = json.loads(raw)
        (folder / "metadata.json").write_bytes(raw)
        files = []
        for item in info.get("siblings", []):
            name = item["rfilename"]
            files.append({"name": name, "size": item.get("size"), "lfs": item.get("lfs")})
        small = [f for f in files if f["name"].endswith(("config.json", "model_index.json", "README.md"))
                 or "LICENSE" in Path(f["name"]).name.upper()]
        for f in small:
            if f.get("size", 0) and f["size"] > 128*1024:
                continue
            url = "https://huggingface.co/"+repo+"/resolve/"+info["sha"]+"/"+urllib.parse.quote(f["name"], safe="/")
            try:
                payload = read(url)
                (folder / f["name"].replace("/", "--")).write_bytes(payload)
                f.update(source_url=url, sha256=hashlib.sha256(payload).hexdigest())
            except Exception as exc:
                f["read_error"] = str(exc)
        return {"repo": repo, "revision": info.get("sha"), "last_modified": info.get("lastModified"),
                "license": info.get("cardData", {}).get("license"), "downloads": info.get("downloads"),
                "files": files, "snapshots": small}
    except Exception as exc:
        return {"repo": repo, "error": str(exc)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", help="Inspect only these public repositories; append to the saved catalog.")
    parser.add_argument("--search", action="append", help="Save public Hugging Face search metadata; never download weights.")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    searches_path = ROOT / "searches.json"
    search_results = json.loads(searches_path.read_text(encoding="utf-8")) if searches_path.exists() else []
    new_searches = []
    for query in args.search or []:
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({"search": query, "limit": 50})
        raw = read(url)
        new_searches.append({"query": query, "source_url": url, "results": json.loads(raw)})
    if new_searches:
        merged_searches = {x["query"]: x for x in search_results + new_searches}
        searches_path.write_text(json.dumps(list(merged_searches.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"searches": [{"query": x["query"], "repos": [v["id"] for v in x["results"]]} for x in new_searches]}, ensure_ascii=False))
    repos = args.repo or ([] if args.search else REPOS)
    for repo in repos:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or ".." in repo:
            raise ValueError("Expected a public owner/model repository ID")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(inspect, repos))
    catalog_path = ROOT / "catalog.json"
    previous = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
    catalog = {item["repo"]: item for item in previous}
    catalog.update({item["repo"]: item for item in results})
    catalog_path.write_text(json.dumps(list(catalog.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        text = {k: item.get(k) for k in ("repo", "revision", "license", "error")}
        text["weight_files"] = [{"name": f["name"], "size": f.get("size")} for f in item.get("files", [])
                                if f["name"].endswith((".safetensors", ".gguf", ".bin", ".pt", ".tar.gz"))]
        print(json.dumps(text, ensure_ascii=False))


if __name__ == "__main__":
    main()

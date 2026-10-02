"""Small local prompt curriculum, not neural weight training or a video dataset.

Separate composition from movement so related actions reuse one checked image.
The English prompts are intentionally short enough for the model's CLIP encoder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

NEGATIVE = "cartoon, illustration, blurry, extra people, duplicate workers, malformed limbs, warped machine, bent tracks, floating wheels, text, watermark"
EXCAVATOR_IMAGE = "Documentary photo, yellow tracked excavator on the left, two workers wearing white hard hats and orange vests on the right, gravel construction site, full bodies visible, clear space between workers and bucket, wide eye level view, daylight."
LOADER_IMAGE = "Documentary photo, yellow wheel loader on the left, two workers wearing white hard hats and orange vests on the right, gravel construction site, full bodies visible, wide eye level view, daylight."
DUMP_IMAGE = "Documentary photo, dump truck in left background, two workers wearing white hard hats and orange vests on the right, gravel construction site, full bodies visible, clear open lane, wide eye level view, daylight."
FORKLIFT_IMAGE = "Documentary photo, yellow forklift carrying a low pallet on the left, two workers wearing white hard hats and orange vests on the right, industrial yard, full bodies visible, wide eye level view, daylight."
CRANE_IMAGE = "Documentary photo, mobile crane on the left, hook and hanging steel beam low above ground, two workers wearing white hard hats and orange vests on the right holding tag lines, construction site, full bodies visible, wide eye level view, daylight."
ROLLER_IMAGE = "Documentary photo, yellow road roller on the left, two workers wearing white hard hats and orange vests on the right, asphalt road construction, full bodies visible, wide eye level view, daylight."

PROFILES = {
    "rail_lifting_crew": {
        "label": "실사 참고: 굴착기와 레일 작업자", "equipment": "road_rail_excavator",
        "first_frame_prompt": "Documentary construction photo, excavator at a railway work site, nearby workers in red workwear, beneath a concrete overpass.",
        "prompt": "Fixed tripod camera, constant daylight exposure. A stationary yellow excavator gently moves its boom under a concrete bridge. Nearby workers in red overalls make small guiding hand gestures beside a steel railway track. The rails stay rigid on the ground. Stable crew and equipment, continuous documentary footage.",
        "visible_action": "excavator boom moves gently; nearby workers guide with small gestures",
        "prompt_modifier": "",
        "recommended_seed": 43,
        "reference": {
            "relative_path": "datasets/construction-v1/rail_lifting_crew/segment-06.png",
            "sha256": "db3e97a34091767cdf7c200e7d04874ff18f5edcb64dcc3ac1bb5110a027cd07",
            "author": "ŠJů, Wikimedia Commons", "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://commons.wikimedia.org/wiki/File:Rekonstrukce_smy%C4%8Dky_Florenc,_p%C5%99en%C3%A1%C5%A1en%C3%AD_koleje.webm",
            "changes": "Frame extracted, resized and letterboxed; then conditioned a newly generated AI video; original audio omitted.",
        },
    },
    "excavator_spotter": {
        "label": "굴착기와 신호 작업자", "equipment": "excavator", "first_frame_prompt": EXCAVATOR_IMAGE,
        "prompt": "Yellow tracked excavator left, two workers in white hard hats and orange vests right. One worker raises an arm to signal. The excavator bucket slowly lowers toward gravel, tracks stay planted. The other worker watches. Same people and machine throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "bucket lowers; one worker signals; tracks remain planted",
    },
    "excavator_loading": {
        "label": "굴착기 버킷과 작업자", "equipment": "excavator", "first_frame_prompt": EXCAVATOR_IMAGE,
        "prompt": "Yellow tracked excavator left, two workers in white hard hats and orange vests right. The bucket gently curls and lifts gravel, tracks stay planted. One worker points toward the gravel pile while the other watches. Same people and machine throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "bucket curls and lifts gravel; worker points",
    },
    "loader_guidance": {
        "label": "로더 이동과 작업자 유도", "equipment": "loader", "first_frame_prompt": LOADER_IMAGE,
        "prompt": "Yellow wheel loader left, two workers in white hard hats and orange vests right. The loader creeps forward with bucket low, wheels roll on gravel. One worker gives a small guiding hand signal while the other watches from the side. Same people and machine throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "wheels roll; loader creeps forward; guiding signal",
    },
    "dump_unloading": {
        "label": "덤프 하역과 작업자", "equipment": "dump_truck", "first_frame_prompt": DUMP_IMAGE,
        "prompt": "Dump truck left, two workers in white hard hats and orange vests right. The stationary truck bed slowly tilts upward and gravel slides out. One worker signals from beside the open lane. The other watches. Same people and truck throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "truck bed rises; gravel slides; worker signals",
    },
    "forklift_guidance": {
        "label": "지게차와 작업자", "equipment": "forklift", "first_frame_prompt": FORKLIFT_IMAGE,
        "prompt": "Yellow forklift with low pallet left, two workers in white hard hats and orange vests right. The forklift creeps forward, wheels roll on concrete. One worker gives a small guiding hand signal while the other watches from the side. Same people and forklift throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "forklift wheels roll; pallet remains supported; signal",
    },
    "crane_rigging": {
        "label": "크레인과 인양 작업자", "equipment": "crane", "first_frame_prompt": CRANE_IMAGE,
        "prompt": "Mobile crane left, two workers in white hard hats and orange vests right holding tag lines. The hanging steel beam slowly lifts a short distance, cables stay taut. Workers gently guide the beam with their lines. Same people and crane throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "beam lifts slightly; taut cables; workers guide tag lines",
    },
    "roller_crew": {
        "label": "롤러와 포장 작업자", "equipment": "road_roller", "first_frame_prompt": ROLLER_IMAGE,
        "prompt": "Yellow road roller left, two workers in white hard hats and orange vests right. The roller creeps forward, drum rotates against asphalt. One worker gently moves a rake along the road edge while the other watches. Same people and roller throughout, fixed wide documentary shot, daylight, continuous real time motion.",
        "visible_action": "roller drum turns; rake moves along edge",
    },
}


def revision():
    return hashlib.sha256(json.dumps(PROFILES, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def expand(value, root=None):
    """Explicit presets never discard user-supplied scene prompts."""
    if not value.get("site_template"):
        return dict(value)
    name = value["site_template"]
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError("site_template은 video_site_presets에서 확인한 이름을 사용하세요.")
    if "scenes" in value:
        raise ValueError("site_template과 scenes를 동시에 지정할 수 없습니다. 수정한 장면은 scenes로 전달하세요.")
    item = PROFILES[name]
    scene = {"id": name, "prompt": item["prompt"], "first_frame_prompt": item["first_frame_prompt"],
             "prompt_modifier": item.get("prompt_modifier", ""), "negative_prompt": NEGATIVE}
    text_only = value.get("model_profile") == "wan"
    if text_only:
        # Wan 1.3B cannot condition on a photograph. Use only the prepared text.
        scene.pop("first_frame_prompt")
        scene.pop("prompt_modifier")
    if item.get("reference") and not text_only:
        if root is None:
            raise ValueError("실사 현장 템플릿에는 준비된 참고 자료가 필요합니다.")
        reference = item["reference"]
        image = (root / reference["relative_path"]).resolve()
        if not image.is_relative_to(root.resolve()) or not image.is_file():
            raise ValueError("현장 참고 자료가 없습니다. tools/prepare_site_data.py를 실행하세요.")
        if hashlib.sha256(image.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("현장 참고 화면이 변경됐습니다. 원본에서 다시 준비하세요.")
        scene.update(image_path=str(image), reference_attribution={k: v for k, v in reference.items() if k != "relative_path"})
    return {**value, "continuous": value.get("continuous", True),
            "seed": value.get("seed", item.get("recommended_seed", 42)),
            "duration_seconds": value.get("duration_seconds", 2),
            "scenes": [scene]}


def catalog():
    return {"templates": [{"id": name, "label": item["label"], "recommended_seed": item.get("recommended_seed", 42), "first_image": "licensed_photo_reference" if item.get("reference") else "generated_sd15_image", "quality_review": "draft_only_realism_not_met"} for name, item in PROFILES.items()],
            "default_duration_seconds": 2, "continuous": True,
            "adaptation_method": "local_prompt_and_initial_image_conditioning",
            "weights_finetuned": False, "sora_quality_verified": False,
            "custom_script": "템플릿은 정해진 동작입니다. 다른 동작은 Cursor가 scenes에 짧은 영문 프롬프트로 전달합니다."}

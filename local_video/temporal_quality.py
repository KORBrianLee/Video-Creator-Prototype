"""Cheap continuity signals; they cannot recognize people or judge physics."""
from __future__ import annotations

from statistics import median


def analyze(deltas, brightness=None):
    if not deltas:
        return {"state": "insufficient_frames", "possible_jump_frames": [], "possible_hold_runs": []}
    typical = median(deltas)
    # The absolute floor avoids labelling tiny normal movements as scene cuts.
    threshold = max(12.0, typical * 8.0)
    jumps = [index+1 for index, value in enumerate(deltas) if value > threshold]
    holds, start = [], None
    for index, value in enumerate([*deltas, 1.0]):
        if value <= 0.02 and start is None:
            start = index
        elif value > 0.02 and start is not None:
            if index-start >= 12:
                holds.append({"first_frame": start, "last_frame": index, "frame_pairs": index-start})
            start = None
    darkening = None
    if brightness and len(brightness) >= 8:
        start_level = sum(brightness[:4])/4
        end_level = sum(brightness[-4:])/4
        if start_level-end_level > 15 and end_level < start_level*0.55:
            darkening = {"start_mean_luma": round(start_level, 3), "end_mean_luma": round(end_level, 3),
                         "end_start_ratio": round(end_level/max(start_level, 1), 4)}
    return {"state": "review_required" if jumps or holds or darkening else "no_large_jump_hold_or_darkening_detected",
            "possible_jump_frames": jumps, "possible_hold_runs": holds,
            "possible_darkening": darkening,
            "median_adjacent_luma_difference": round(typical, 5),
            "jump_threshold": round(threshold, 5),
            "limitations": "밝기 변화 검사이며 작업자 수·형태·관절·물리·미세 떨림이나 사실감을 판정하지 않습니다. 의도한 어두워짐도 검토 표시될 수 있습니다."}

from typing import Dict


def get_diff(prev: Dict[str, int], new: [str, int]):
    keys = prev.keys()
    diffs = []
    for key in keys:
        diffs.append(abs(prev[key] - new[key]))

    return sum(diffs) / len(diffs)
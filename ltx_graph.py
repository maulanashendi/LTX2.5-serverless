from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
I2V_TEMPLATE_PATH = ROOT_DIR / "video_ltx2_5_i2v_API.json"
T2V_TEMPLATE_PATH = ROOT_DIR / "video_ltx2_5_t2v_API.json"

MODES = ("i2v", "t2v")

FPS = 24
SECONDS_MIN = 1.0
SECONDS_MAX = 20.0

ASPECT_RATIOS: dict[str, dict[str, int]] = {
    "16:9": {"width": 1280, "height": 720},
    "9:16": {"width": 720, "height": 1280},
    "1:1": {"width": 1024, "height": 1024},
}

DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_SECONDS = 5
DEFAULT_IMAGE_NAME = "source-image.png"

PROMPT_NODE = "398:376"
DURATION_NODE = "398:362"
WIDTH_NODE = "398:372"
HEIGHT_NODE = "398:360"
FPS_NODE = "398:361"
PROMPT_OPTIMIZER_NODE = "398:380"
PROMPT_OPTIMIZER_TOGGLE_NODE = "398:383"
SEED_NODE_1 = "398:338"
SEED_NODE_2 = "398:339"
IMAGE_NODE = "395"

with I2V_TEMPLATE_PATH.open("r", encoding="utf-8") as _template_file:
    _I2V_TEMPLATE: dict[str, Any] = json.load(_template_file)

with T2V_TEMPLATE_PATH.open("r", encoding="utf-8") as _template_file:
    _T2V_TEMPLATE: dict[str, Any] = json.load(_template_file)

_TEMPLATES: dict[str, dict[str, Any]] = {"i2v": _I2V_TEMPLATE, "t2v": _T2V_TEMPLATE}


def seconds_to_frames(seconds: float) -> int:
    if seconds < SECONDS_MIN or seconds > SECONDS_MAX:
        raise ValueError(
            f"Duration must be between {SECONDS_MIN:g} and {SECONDS_MAX:g} seconds."
        )
    if not float(seconds).is_integer():
        raise ValueError("Duration must be a whole number of seconds.")
    return int(round(seconds * FPS)) + 1


def sanitize_image_name(image_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (image_name or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or DEFAULT_IMAGE_NAME


def build_graph(
    *,
    mode: str,
    prompt: str,
    seconds: float = DEFAULT_SECONDS,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    image_name: str | None = None,
    optimize_prompt: bool = True,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    if prompt is None or not str(prompt).strip():
        raise ValueError("Prompt is required.")
    prompt = str(prompt).strip()

    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    if mode == "i2v" and not image_name:
        raise ValueError("Source image name is required for i2v mode.")

    # Validates the range/whole-number rule; the frame count itself is computed
    # inside the graph by node "398:378" from duration x fps, never written here.
    seconds_to_frames(seconds)
    dimensions = ASPECT_RATIOS[aspect_ratio]

    graph: dict[str, Any] = copy.deepcopy(_TEMPLATES[mode])

    if seed is not None:
        seed_pass_1 = int(seed)
        seed_pass_2 = int(seed) + 1
        enhancer_seed = int(seed)
    else:
        random_source = rng or random.SystemRandom()
        enhancer_seed = random_source.randrange(1, 10**9)
        seed_pass_1 = random_source.randrange(1, 10**15)
        seed_pass_2 = random_source.randrange(1, 10**15)

    graph[PROMPT_NODE]["inputs"]["value"] = prompt
    graph[DURATION_NODE]["inputs"]["value"] = int(seconds)
    graph[WIDTH_NODE]["inputs"]["value"] = dimensions["width"]
    graph[HEIGHT_NODE]["inputs"]["value"] = dimensions["height"]
    graph[FPS_NODE]["inputs"]["value"] = FPS
    graph[PROMPT_OPTIMIZER_NODE]["inputs"]["sampling_mode"] = (
        "on" if optimize_prompt else "off"
    )
    graph[PROMPT_OPTIMIZER_NODE]["inputs"]["sampling_mode.seed"] = enhancer_seed
    graph[PROMPT_OPTIMIZER_TOGGLE_NODE]["inputs"]["value"] = bool(optimize_prompt)
    graph[SEED_NODE_1]["inputs"]["noise_seed"] = seed_pass_1
    graph[SEED_NODE_2]["inputs"]["noise_seed"] = seed_pass_2

    if mode == "i2v":
        graph[IMAGE_NODE]["inputs"]["image"] = image_name

    return graph

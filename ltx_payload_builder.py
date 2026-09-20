from __future__ import annotations

import random
from typing import Any

from ltx_graph import (
    ASPECT_RATIOS,
    FPS,
    MODES,
    SECONDS_MAX,
    SECONDS_MIN,
    build_graph,
    sanitize_image_name,
    seconds_to_frames,
)

DEFAULT_MODE = "i2v"
SECONDS_STEP = 1.0


def build_payload(
    *,
    prompt: str,
    seconds: float,
    aspect_ratio: str,
    image_name: str | None = None,
    image_data_url: str | None = None,
    optimize_prompt: bool,
    mode: str = DEFAULT_MODE,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "i2v":
        if not image_data_url or not image_data_url.startswith("data:image/"):
            raise ValueError("Source image must be provided as a data URL.")
        if not image_name or not image_name.strip():
            raise ValueError("Source image name is required.")

    normalized_image_name = sanitize_image_name(image_name) if mode == "i2v" else None

    workflow = build_graph(
        mode=mode,
        prompt=prompt,
        seconds=seconds,
        aspect_ratio=aspect_ratio,
        image_name=normalized_image_name,
        optimize_prompt=optimize_prompt,
        rng=rng,
    )

    if mode == "i2v":
        return {
            "input": {
                "workflow": workflow,
                "images": [
                    {
                        "name": normalized_image_name,
                        "image": image_data_url,
                    }
                ],
            }
        }

    return {"input": {"workflow": workflow}}

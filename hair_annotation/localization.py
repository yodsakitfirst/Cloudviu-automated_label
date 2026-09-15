"""Model-independent geometry helpers for generic product localization."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .config import LocalizationConfig
from .types import Candidate, Tile


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_tiles(
    image_width: int, image_height: int, tile_size: int, overlap: float
) -> list[Tile]:
    """Generate row-major, edge-anchored tiles covering an image."""
    if type(image_width) is not int or image_width <= 0:
        raise ValueError("image_width must be a positive integer")
    if type(image_height) is not int or image_height <= 0:
        raise ValueError("image_height must be a positive integer")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
        raise ValueError("overlap must be a finite number between 0 and 1")
    overlap = float(overlap)
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be a finite number between 0 and 1")

    x_starts = _axis_starts(image_width, tile_size, overlap)
    y_starts = _axis_starts(image_height, tile_size, overlap)
    tiles: list[Tile] = []
    for y in y_starts:
        for x in x_starts:
            tiles.append(
                Tile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=min(tile_size, image_width - x),
                    height=min(tile_size, image_height - y),
                )
            )
    return tiles


def mask_to_box(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return a tight xyxy box for a finite polygon, if it has area."""
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1:] != (2,):
        return None
    if array.shape[0] == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    if not np.isfinite(array).all():
        return None

    x1 = float(np.min(array[:, 0]))
    y1 = float(np.min(array[:, 1]))
    x2 = float(np.max(array[:, 0]))
    y2 = float(np.max(array[:, 1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _record_rejection(rejected: dict[str, int], reason: str) -> None:
    rejected[reason] = rejected.get(reason, 0) + 1


def filter_candidates(
    candidates: Iterable[Candidate],
    image_width: int,
    image_height: int,
    config: LocalizationConfig,
) -> tuple[list[Candidate], dict[str, int]]:
    """Clip candidates to image bounds and apply conservative geometry filters."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    valid: list[Candidate] = []
    rejected: dict[str, int] = {}
    image_area = float(image_width * image_height)
    for candidate in candidates:
        values = (*candidate.xyxy, candidate.localization_confidence)
        try:
            finite = all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            _record_rejection(rejected, "non_finite")
            continue

        x1, y1, x2, y2 = (float(value) for value in candidate.xyxy)
        clipped = (
            min(max(x1, 0.0), float(image_width)),
            min(max(y1, 0.0), float(image_height)),
            min(max(x2, 0.0), float(image_width)),
            min(max(y2, 0.0), float(image_height)),
        )
        clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
        width = clipped_x2 - clipped_x1
        height = clipped_y2 - clipped_y1
        if width <= 0.0 or height <= 0.0:
            _record_rejection(rejected, "zero_area")
            continue
        if width < config.min_side or height < config.min_side:
            _record_rejection(rejected, "too_small")
            continue
        if width * height > config.max_area_ratio * image_area:
            _record_rejection(rejected, "too_large")
            continue
        aspect_ratio = width / height
        if not config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
            _record_rejection(rejected, "aspect_ratio")
            continue

        valid.append(
            Candidate(
                xyxy=clipped,
                localization_confidence=candidate.localization_confidence,
                prompt_name=candidate.prompt_name,
                tile_index=candidate.tile_index,
                mask_used=candidate.mask_used,
            )
        )
    return valid, rejected


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_width = max(0.0, first[2] - first[0])
    first_height = max(0.0, first[3] - first[1])
    second_width = max(0.0, second[2] - second[0])
    second_height = max(0.0, second[3] - second[1])
    first_area = first_width * first_height
    second_area = second_width * second_height
    union = first_area + second_area
    if union <= 0.0:
        return 0.0

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    return intersection / (union - intersection)


def class_agnostic_nms(
    candidates: Iterable[Candidate], iou_threshold: float
) -> tuple[list[Candidate], int]:
    """Suppress overlapping boxes independent of their generic prompt class."""
    if isinstance(iou_threshold, bool) or not math.isfinite(float(iou_threshold)):
        raise ValueError("iou_threshold must be finite")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    ordered = list(candidates)
    ranked = sorted(
        enumerate(ordered),
        key=lambda item: (-item[1].localization_confidence, item[0]),
    )
    kept: list[Candidate] = []
    duplicate_count = 0
    for _, candidate in ranked:
        if any(
            _intersection_over_union(candidate.xyxy, previous.xyxy) > iou_threshold
            for previous in kept
        ):
            duplicate_count += 1
            continue
        kept.append(candidate)
    return kept, duplicate_count

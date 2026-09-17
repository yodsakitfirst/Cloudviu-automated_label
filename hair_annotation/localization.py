"""Model-independent geometry helpers for generic product localization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .config import LocalizationConfig, RecallRetryConfig
from .types import Candidate, Tile


_GEOMETRY_REJECTION_KEYS = (
    "non_finite",
    "zero_area",
    "too_small",
    "too_large",
    "aspect_ratio",
)


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


def load_text_localizer(model_name: str, prompts: Sequence[str]) -> Any:
    """Load a YOLOE segmentation checkpoint and configure text prompts."""
    try:
        from ultralytics import YOLOE

        model = YOLOE(model_name)
        model.set_classes(list(prompts))
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load YOLOE text localizer '{model_name}': {exc}. "
            "Make the offline checkpoint available locally before running."
        ) from exc


def _as_numpy(value: Any) -> np.ndarray:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current)


def _parse_tile_result(
    results: Any,
    tile: Tile,
    prompts: Sequence[str],
) -> tuple[list[Candidate], int, int]:
    if not results:
        raise ValueError("YOLOE returned no result object")
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLOE result is missing boxes")

    xyxy = _as_numpy(boxes.xyxy)
    confidence = _as_numpy(boxes.conf).reshape(-1)
    prompt_indices = _as_numpy(boxes.cls).reshape(-1)
    if xyxy.size == 0:
        xyxy = xyxy.reshape((0, 4))
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("boxes.xyxy must have shape (N, 4)")
    if not (len(xyxy) == len(confidence) == len(prompt_indices)):
        raise ValueError("YOLOE output arrays have unequal lengths")

    mask_polygons = getattr(getattr(result, "masks", None), "xy", None)
    candidates: list[Candidate] = []
    mask_count = 0
    fallback_count = 0
    for index, (box, score, raw_prompt_index) in enumerate(
        zip(xyxy, confidence, prompt_indices)
    ):
        try:
            prompt_value = float(raw_prompt_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"YOLOE returned invalid prompt index {raw_prompt_index!r}"
            ) from exc
        if not math.isfinite(prompt_value) or prompt_value != math.floor(prompt_value):
            raise ValueError(f"YOLOE returned invalid prompt index {prompt_value!r}")
        prompt_index = int(prompt_value)
        if not 0 <= prompt_index < len(prompts):
            raise ValueError(f"YOLOE returned unknown prompt index {prompt_index}")

        mask_box = None
        if mask_polygons is not None:
            try:
                mask_box = mask_to_box(mask_polygons[index])
            except (IndexError, TypeError, ValueError):
                mask_box = None
        if mask_box is None:
            selected_box = tuple(float(value) for value in box)
            mask_used = False
            fallback_count += 1
        else:
            selected_box = mask_box
            mask_used = True
            mask_count += 1

        x1, y1, x2, y2 = selected_box
        candidates.append(
            Candidate(
                xyxy=(x1 + tile.x, y1 + tile.y, x2 + tile.x, y2 + tile.y),
                localization_confidence=float(score),
                prompt_name=prompts[prompt_index],
                tile_index=tile.index,
                mask_used=mask_used,
            )
        )
    return candidates, mask_count, fallback_count


def _infer_tile(
    model: Any,
    image_bgr: np.ndarray,
    tile: Tile,
    prompts: Sequence[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: int | str,
) -> tuple[list[Candidate], int, int]:
    tile_image = image_bgr[
        tile.y : tile.y + tile.height,
        tile.x : tile.x + tile.width,
    ]
    try:
        results = model.predict(
            source=tile_image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        return _parse_tile_result(results, tile, prompts)
    except Exception as exc:
        raise RuntimeError(
            f"YOLOE text localization failed for tile {tile.index}: {exc}"
        ) from exc


def _diagnostics(
    tile_count: int,
    raw_count: int,
    mask_count: int,
    fallback_count: int,
    duplicate_count: int,
    rejected: dict[str, int],
) -> dict[str, int]:
    diagnostics = {
        "tiles": tile_count,
        "raw_candidates": raw_count,
        "mask_boxes": mask_count,
        "fallback_boxes": fallback_count,
        "duplicates_removed": duplicate_count,
    }
    diagnostics.update(
        {reason: rejected.get(reason, 0) for reason in _GEOMETRY_REJECTION_KEYS}
    )
    return diagnostics


def _localize_pass(
    model: Any,
    image_bgr: np.ndarray,
    config: LocalizationConfig | RecallRetryConfig,
    prompts: Sequence[str],
    imgsz: int,
    device: int | str,
) -> tuple[list[Candidate], dict[str, int]]:
    image_height, image_width = image_bgr.shape[:2]
    tiles = generate_tiles(image_width, image_height, config.tile_size, config.overlap)
    raw_candidates: list[Candidate] = []
    mask_count = 0
    fallback_count = 0
    for tile in tiles:
        tile_candidates, tile_masks, tile_fallbacks = _infer_tile(
            model,
            image_bgr,
            tile,
            prompts,
            config.conf,
            config.iou,
            imgsz,
            device,
        )
        raw_candidates.extend(tile_candidates)
        mask_count += tile_masks
        fallback_count += tile_fallbacks

    filtered, rejected = filter_candidates(
        raw_candidates, image_width, image_height, config
    )
    return filtered, _diagnostics(
        len(tiles),
        len(raw_candidates),
        mask_count,
        fallback_count,
        0,
        rejected,
    )


def localize_image(
    model: Any,
    image_bgr: np.ndarray,
    config: LocalizationConfig,
    imgsz: int,
    device: int | str,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Run text-prompted YOLOE inference with an optional recall retry."""
    primary, primary_diagnostics = _localize_pass(
        model, image_bgr, config, config.prompts, imgsz, device
    )
    primary_kept, _ = class_agnostic_nms(primary, config.nms_iou)
    retry = config.recall_retry
    if retry is None:
        kept, duplicate_count = class_agnostic_nms(primary, config.nms_iou)
        primary_diagnostics["duplicates_removed"] = duplicate_count
        return kept, primary_diagnostics

    image_height, image_width = image_bgr.shape[:2]
    megapixels = image_width * image_height / 1_000_000.0
    required_candidates = max(
        1, math.ceil(megapixels * retry.min_candidates_per_megapixel)
    )
    retry_triggered = retry.enabled and len(primary_kept) < required_candidates
    retry_candidates: list[Candidate] = []
    retry_diagnostics = _diagnostics(0, 0, 0, 0, 0, {})
    if retry_triggered:
        retry_candidates, retry_diagnostics = _localize_pass(
            model, image_bgr, retry, config.prompts, imgsz, device
        )

    merged = [*primary, *retry_candidates]
    nms_iou = retry.nms_iou if retry_triggered else config.nms_iou
    kept, duplicate_count = class_agnostic_nms(merged, nms_iou)
    retry_kept, _ = class_agnostic_nms(retry_candidates, retry.nms_iou)
    diagnostics: dict[str, Any] = {
        "tiles": primary_diagnostics["tiles"] + retry_diagnostics["tiles"],
        "raw_candidates": (
            primary_diagnostics["raw_candidates"]
            + retry_diagnostics["raw_candidates"]
        ),
        "mask_boxes": (
            primary_diagnostics["mask_boxes"] + retry_diagnostics["mask_boxes"]
        ),
        "fallback_boxes": (
            primary_diagnostics["fallback_boxes"]
            + retry_diagnostics["fallback_boxes"]
        ),
        "duplicates_removed": duplicate_count,
    }
    diagnostics.update(
        {
            reason: primary_diagnostics[reason] + retry_diagnostics[reason]
            for reason in _GEOMETRY_REJECTION_KEYS
        }
    )
    diagnostics.update(
        {
            "recall_retry_triggered": retry_triggered,
            "recall_retry_reason": (
                f"{len(primary_kept)} primary candidates below "
                f"{required_candidates} required"
                if retry_triggered
                else None
            ),
            "primary_raw_candidates": primary_diagnostics["raw_candidates"],
            "primary_candidates": len(primary_kept),
            "retry_tiles": retry_diagnostics["tiles"],
            "retry_raw_candidates": retry_diagnostics["raw_candidates"],
            "retry_candidates": len(retry_kept),
            "merged_candidates_before_nms": len(merged),
        }
    )
    return kept, diagnostics


def _filter_reference_candidates(
    candidates: Iterable[Candidate], image_width: int, image_height: int
) -> tuple[list[Candidate], dict[str, int]]:
    valid: list[Candidate] = []
    rejected: dict[str, int] = {}
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
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            _record_rejection(rejected, "zero_area")
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


def localize_reference_image(
    model: Any,
    image_bgr: np.ndarray,
    config: LocalizationConfig,
    imgsz: int,
    device: int | str,
) -> tuple[list[Candidate], dict[str, int]]:
    """Localize products in one full reference image without shelf heuristics."""
    image_height, image_width = image_bgr.shape[:2]
    tile = Tile(index=0, x=0, y=0, width=image_width, height=image_height)
    raw_candidates, mask_count, fallback_count = _infer_tile(
        model,
        image_bgr,
        tile,
        config.prompts,
        config.conf,
        config.iou,
        imgsz,
        device,
    )
    filtered, rejected = _filter_reference_candidates(
        raw_candidates, image_width, image_height
    )
    kept, duplicate_count = class_agnostic_nms(filtered, config.nms_iou)
    return kept, _diagnostics(
        1,
        len(raw_candidates),
        mask_count,
        fallback_count,
        duplicate_count,
        rejected,
    )

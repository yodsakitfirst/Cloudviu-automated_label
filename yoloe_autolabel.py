"""Offline-testable YOLOE assisted annotation pipeline.

Ultralytics and OpenCV are imported only at the boundaries that need them so
manifest/configuration validation remains usable without a model or GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hair_annotation.config import BoxFirstConfig


@dataclass(frozen=True)
class Sku:
    class_id: int
    barcode: str
    brand: str
    sku_name: str
    enabled: bool


@dataclass(frozen=True)
class ReferencePrompt:
    class_id: int
    image_path: Path
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class Prediction:
    dataset_class_id: int
    prompt_class_id: int
    barcode: str
    sku_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    prompt_batch: int


@dataclass(frozen=True)
class PromptBatch:
    index: int
    dataset_class_ids: tuple[int, ...]
    prompt_to_dataset_class: dict[int, int]
    canvas_path: Path
    boxes: np.ndarray
    temporary_class_ids: np.ndarray


_REQUIRED_SECTIONS = (
    "project",
    "dataset",
    "yoloe",
    "localization",
    "matching",
    "export",
    "output",
    "pilot",
)
_MANIFEST_COLUMNS = ("class_id", "barcode", "brand", "sku_name", "enabled")
_OFFICIAL_YOLOE_ALIASES = frozenset(
    f"yoloe-{family}{size}-seg.pt"
    for family in ("v8", "11", "26")
    for size in ("n", "s", "m", "l", "x")
)


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _class_id(value: Any, label: str) -> int:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"{label} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    return parsed


def _require_boolean(section: Mapping[str, Any], key: str, label: str) -> bool:
    value = section.get(key)
    if type(value) is not bool:
        raise ValueError(f"{label} must be true or false")
    return value


def _validate_config(data: dict[str, Any]) -> None:
    for section in _REQUIRED_SECTIONS:
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
    BoxFirstConfig.from_mapping(data)
    dataset = data["dataset"]
    yoloe = data["yoloe"]
    output = data["output"]
    pilot = data["pilot"]
    for aliases in (("manifest", "sku_manifest"), ("references", "reference_definitions"), ("images", "shelf_images", "image_root")):
        value = next((dataset[name] for name in aliases if name in dataset), None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Dataset config requires a non-empty path ({', '.join(aliases)})")
    extensions = dataset.get("extensions", dataset.get("image_extensions"))
    if not isinstance(extensions, (list, tuple)) or not extensions or any(not isinstance(ext, str) or not ext.strip() for ext in extensions):
        raise ValueError("dataset image extensions must be a non-empty list of strings")
    model = yoloe.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("yoloe.model must be a non-empty string")
    if "imgsz" not in yoloe:
        raise ValueError("Missing required config value yoloe.imgsz")
    _positive_integer(yoloe["imgsz"], "yoloe.imgsz")
    # Legacy visual-prompt controls are no longer required, but validate them
    # when present so older configs cannot silently carry malformed values.
    for key in ("prompt_batch_size", "canvas_cell_size"):
        if key in yoloe:
            _positive_integer(yoloe[key], f"yoloe.{key}")
    for key in ("conf", "iou"):
        if key in yoloe:
            _probability(yoloe[key], f"yoloe.{key}")
    device = yoloe.get("device")
    if isinstance(device, bool) or not (
        (type(device) is int and device >= 0) or (isinstance(device, str) and bool(device.strip()))
    ):
        raise ValueError("yoloe.device must be a non-negative integer or non-empty string")
    if "canvas_padding" in yoloe:
        padding = yoloe["canvas_padding"]
        if type(padding) is not int or padding < 0:
            raise ValueError("yoloe.canvas_padding must be a non-negative integer")
        if "canvas_cell_size" in yoloe and padding * 2 >= yoloe["canvas_cell_size"]:
            raise ValueError("yoloe.canvas_padding must be smaller than half the cell size")
    root = output.get("root")
    # The effective root can be supplied by --output; when present in the file,
    # it must still be valid at this configuration-only boundary.
    if "root" in output and (not isinstance(root, str) or not root.strip()):
        raise ValueError("output.root must be a non-empty string when provided")
    for key in ("overwrite", "save_metadata", "save_previews"):
        _require_boolean(output, key, f"output.{key}")
    for key in ("low_confidence_threshold", "low_confidence_cutoff"):
        if key in output:
            _probability(output[key], f"output.{key}")
    _require_boolean(pilot, "enabled", "pilot.enabled")
    for key in ("max_images", "max_skus"):
        if key not in pilot:
            raise ValueError(f"Missing required config value pilot.{key}")
        _positive_integer(pilot[key], f"pilot.{key}")
    if "class_ids" in pilot:
        if not isinstance(pilot["class_ids"], (list, tuple)):
            raise ValueError("pilot.class_ids must be a list")
        for index, value in enumerate(pilot["class_ids"]):
            _class_id(value, f"pilot.class_ids[{index}]")


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_reviewed_labels(path: Path) -> None:
    lexical = _absolute_lexical(Path(path))
    if any(part.casefold() == "reviewed_labels" for part in lexical.parts):
        raise ValueError("Automated labeling may never access reviewed_labels")
    resolved = lexical.resolve()
    if any(part.casefold() == "reviewed_labels" for part in resolved.parts):
        raise ValueError("Automated labeling may never access reviewed_labels")


def _assert_input_allowed(path: Path) -> Path:
    _reject_reviewed_labels(Path(path))
    return Path(path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate config; output.root may be omitted for a CLI override."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configuration files") from exc
    path = _assert_input_allowed(Path(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid YAML in config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    _validate_config(data)
    return data


def resolve_config_path(config_path: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured path must be a non-empty string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config_path).resolve().parent / candidate
    _reject_reviewed_labels(candidate)
    return candidate.resolve()


def validate_manifest(skus: Sequence[Sku]) -> None:
    if not skus:
        raise ValueError("SKU manifest is empty")
    ids: set[int] = set()
    barcodes: set[str] = set()
    for item in skus:
        if isinstance(item.class_id, bool) or not isinstance(item.class_id, int) or item.class_id < 0:
            raise ValueError(f"Invalid class_id: {item.class_id!r}")
        if item.class_id in ids:
            raise ValueError(f"Duplicate class_id: {item.class_id}")
        barcode = item.barcode.strip()
        if not barcode:
            raise ValueError("Barcode must not be blank")
        barcode_key = barcode.casefold()
        if barcode_key in barcodes:
            raise ValueError(f"Duplicate barcode: {barcode}")
        if not item.brand.strip():
            raise ValueError(f"Blank brand for class {item.class_id}")
        if not item.sku_name.strip():
            raise ValueError(f"Blank sku_name for class {item.class_id}")
        if not isinstance(item.enabled, bool):
            raise ValueError(f"Invalid enabled flag for class {item.class_id}")
        ids.add(item.class_id)
        barcodes.add(barcode_key)


def load_sku_manifest(path: Path) -> dict[int, Sku]:
    path = _assert_input_allowed(Path(path))
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = set(_MANIFEST_COLUMNS) - fields
            if missing:
                raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
            items: list[Sku] = []
            for line_number, row in enumerate(reader, 2):
                raw_id = (row.get("class_id") or "").strip()
                class_id = _class_id(raw_id, f"class_id on line {line_number}")
                enabled_text = (row.get("enabled") or "").strip().casefold()
                if enabled_text not in {"true", "false"}:
                    raise ValueError(f"Invalid enabled value on line {line_number}: {enabled_text!r}")
                items.append(
                    Sku(
                        class_id=class_id,
                        barcode=(row.get("barcode") or "").strip(),
                        brand=(row.get("brand") or "").strip(),
                        sku_name=(row.get("sku_name") or "").strip(),
                        enabled=enabled_text == "true",
                    )
                )
    except OSError as exc:
        raise ValueError(f"Cannot read manifest {path}: {exc}") from exc
    validate_manifest(items)
    return {item.class_id: item for item in items}


def load_reference_definitions(
    path: Path, skus: Mapping[int, Sku], base_dir: Path | None = None
) -> list[ReferencePrompt]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read reference definitions") from exc
    path = _assert_input_allowed(Path(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"Cannot read references {path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid reference YAML {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("references"), list):
        raise ValueError("Reference YAML must be a mapping containing a references list")
    prompts: list[ReferencePrompt] = []
    for index, entry in enumerate(data["references"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Reference {index} must be a mapping")
        class_id = _class_id(entry.get("class_id"), f"reference class_id at index {index}")
        if class_id not in skus:
            raise ValueError(f"Reference uses unknown class_id {class_id}")
        if not skus[class_id].enabled:
            continue
        image_value = entry.get("image_path", entry.get("image", entry.get("path")))
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"Reference {index} has no image path")
        image_candidate = Path(image_value).expanduser()
        if not image_candidate.is_absolute():
            image_candidate = (Path(base_dir).resolve() if base_dir is not None else path.parent) / image_candidate
        image_path = _assert_input_allowed(image_candidate)
        raw_bbox = entry.get("bbox")
        bbox = None
        if raw_bbox is not None:
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                raise ValueError(f"Reference bbox at index {index} must contain four values")
            try:
                bbox = tuple(float(value) for value in raw_bbox)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Reference bbox at index {index} must be numeric") from exc
        prompts.append(ReferencePrompt(class_id, image_path, bbox))
    return prompts


def validate_reference(prompt: ReferencePrompt, image: np.ndarray, skus: Mapping[int, Sku]) -> tuple[int, int, int, int]:
    if prompt.class_id not in skus:
        raise ValueError(f"Reference uses unknown class_id {prompt.class_id}")
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"Reference image is corrupt or empty: {prompt.image_path}")
    height, width = int(image.shape[0]), int(image.shape[1])
    if prompt.bbox is None:
        return 0, 0, width, height
    if len(prompt.bbox) != 4 or not all(math.isfinite(float(v)) for v in prompt.bbox):
        raise ValueError(f"Reference bbox must contain four finite values: {prompt.image_path}")
    x1, y1, x2, y2 = (float(v) for v in prompt.bbox)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Reference bbox is outside image bounds: {prompt.image_path}")
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def select_active_skus(skus: Mapping[int, Sku], pilot: Mapping[str, Any]) -> list[Sku]:
    enabled = sorted((item for item in skus.values() if item.enabled), key=lambda item: item.class_id)
    if not pilot.get("enabled", False):
        selected = enabled
    elif pilot.get("class_ids"):
        raw_ids = pilot["class_ids"]
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError("pilot.class_ids must be a list")
        requested: set[int] = set()
        for raw_id in raw_ids:
            if isinstance(raw_id, bool):
                raise ValueError("pilot.class_ids must contain integer IDs")
            class_id = _class_id(raw_id, "pilot.class_ids value")
            if class_id not in skus or not skus[class_id].enabled:
                raise ValueError(f"Pilot class_id {class_id} is missing or disabled")
            requested.add(class_id)
        selected = [item for item in enabled if item.class_id in requested]
    else:
        maximum = _positive_integer(pilot.get("max_skus"), "pilot.max_skus")
        selected = enabled[:maximum]
    if not selected:
        raise ValueError("At least one enabled SKU must be selected")
    return selected


def discover_shelf_images(root: Path, extensions: Sequence[str]) -> list[Path]:
    root = _assert_input_allowed(Path(root))
    if not root.is_dir():
        raise ValueError(f"Shelf image root is not a directory: {root}")
    normalized = {str(ext).casefold().lstrip(".") for ext in extensions if str(ext).strip()}
    if not normalized:
        raise ValueError("At least one shelf image extension is required")
    images = [
        _assert_input_allowed(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold().lstrip(".") in normalized
    ]
    images.sort(key=lambda path: (path.name.casefold(), path.name))
    if not images:
        raise ValueError(f"No shelf images found under {root}")
    stems: dict[str, Path] = {}
    for image in images:
        key = image.stem.casefold()
        if key in stems:
            raise ValueError(f"Duplicate image stem (case-insensitive): {stems[key].name} and {image.name}")
        stems[key] = image
    return images


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (opencv-python) is required for image operations") from exc
    return cv2


def _setting(settings: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return settings[name] if name in settings else default


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _reject_output_alias(destination: Path, inputs: Sequence[Path]) -> None:
    resolved_destination = Path(destination).resolve()
    for source in inputs:
        resolved_source = _assert_input_allowed(Path(source))
        if resolved_destination == resolved_source or _same_existing_file(Path(destination), Path(source)):
            raise ValueError(f"Output destination aliases input file: {destination} -> {source}")


def _atomic_write_image(path: Path, image: np.ndarray, cv2: Any, overwrite: bool) -> None:
    destination = _assert_raw_destination(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".png"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=suffix, dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"Failed to write image: {destination}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_prompt_batches(
    references: Sequence[ReferencePrompt],
    active_skus: Sequence[Sku],
    settings: Mapping[str, Any],
    canvas_dir: Path,
) -> list[PromptBatch]:
    batch_size = _positive_integer(_setting(settings, "prompt_batch_size"), "prompt_batch_size")
    cell_size = _positive_integer(_setting(settings, "canvas_cell_size"), "canvas_cell_size")
    padding = int(_setting(settings, "canvas_padding", 0))
    if padding < 0 or padding * 2 >= cell_size:
        raise ValueError("canvas_padding must be non-negative and smaller than half the cell size")
    ordered = sorted(active_skus, key=lambda item: item.class_id)
    active_ids = {item.class_id for item in ordered}
    grouped: dict[int, list[ReferencePrompt]] = {class_id: [] for class_id in active_ids}
    for prompt in references:
        if prompt.class_id in grouped:
            grouped[prompt.class_id].append(prompt)
    missing = [class_id for class_id, prompts in grouped.items() if not prompts]
    if missing:
        raise ValueError(f"Missing references for enabled class IDs: {missing}")

    cv2 = _load_cv2()
    decoded: dict[ReferencePrompt, tuple[np.ndarray, tuple[int, int, int, int]]] = {}
    sku_map = {item.class_id: item for item in ordered}
    for prompt in references:
        if prompt.class_id not in active_ids:
            continue
        _assert_input_allowed(prompt.image_path)
        if not prompt.image_path.is_file():
            raise ValueError(f"Reference image is missing: {prompt.image_path}")
        image = cv2.imread(str(prompt.image_path))
        if image is None:
            raise ValueError(f"Reference image cannot be decoded: {prompt.image_path}")
        decoded[prompt] = (image, validate_reference(prompt, image, sku_map))

    canvas_dir = Path(canvas_dir)
    planned_canvases = [
        canvas_dir / f"batch_{batch_index:03d}.png"
        for batch_index in range(math.ceil(len(ordered) / batch_size))
    ]
    reference_paths = [prompt.image_path for prompt in references if prompt.class_id in active_ids]
    for destination in planned_canvases:
        _assert_raw_destination(destination)
        _reject_output_alias(destination, reference_paths)
    batches: list[PromptBatch] = []
    for batch_index, offset in enumerate(range(0, len(ordered), batch_size)):
        batch_skus = ordered[offset : offset + batch_size]
        mapping = {temporary_id: item.class_id for temporary_id, item in enumerate(batch_skus)}
        views: list[tuple[int, ReferencePrompt]] = []
        for temporary_id, item in enumerate(batch_skus):
            views.extend((temporary_id, prompt) for prompt in grouped[item.class_id])
        columns = max(1, math.ceil(math.sqrt(len(views))))
        rows = math.ceil(len(views) / columns)
        canvas = np.zeros((rows * cell_size, columns * cell_size, 3), dtype=np.uint8)
        boxes: list[list[float]] = []
        temporary_ids: list[int] = []
        for view_index, (temporary_id, prompt) in enumerate(views):
            image, (x1, y1, x2, y2) = decoded[prompt]
            crop = image[y1:y2, x1:x2]
            available = cell_size - 2 * padding
            scale = min(available / crop.shape[1], available / crop.shape[0])
            target_width = max(1, min(available, int(round(crop.shape[1] * scale))))
            target_height = max(1, min(available, int(round(crop.shape[0] * scale))))
            resized = cv2.resize(crop, (target_width, target_height))
            row, column = divmod(view_index, columns)
            left = column * cell_size + padding + (available - target_width) // 2
            top = row * cell_size + padding + (available - target_height) // 2
            canvas[top : top + target_height, left : left + target_width] = resized
            boxes.append([float(left), float(top), float(left + target_width), float(top + target_height)])
            temporary_ids.append(temporary_id)
        canvas_path = planned_canvases[batch_index]
        if not (settings.get("_reuse_existing_canvases", False) and canvas_path.is_file()):
            _atomic_write_image(canvas_path, canvas, cv2, True)
        batches.append(
            PromptBatch(
                index=batch_index,
                dataset_class_ids=tuple(item.class_id for item in batch_skus),
                prompt_to_dataset_class=mapping,
                canvas_path=canvas_path,
                boxes=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
                temporary_class_ids=np.asarray(temporary_ids, dtype=np.int64),
            )
        )
    return batches


def _is_model_path_like(value: str) -> bool:
    return value not in _OFFICIAL_YOLOE_ALIASES


def load_yoloe(model_path: str) -> Any:
    if _is_model_path_like(model_path):
        _reject_reviewed_labels(Path(model_path))
    try:
        from ultralytics import YOLOE

        return YOLOE(model_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load YOLOE model '{model_path}': {exc}") from exc


def _as_numpy(value: Any) -> np.ndarray:
    current = value
    for method in ("detach", "cpu"):
        function = getattr(current, method, None)
        if callable(function):
            current = function()
    if hasattr(current, "numpy") and callable(current.numpy):
        current = current.numpy()
    return np.asarray(current)


def run_yoloe(
    model: Any,
    source: Path,
    batch: PromptBatch,
    settings: Mapping[str, Any],
    skus: Mapping[int, Sku],
) -> list[Prediction]:
    _reject_reviewed_labels(Path(source))
    _reject_reviewed_labels(Path(batch.canvas_path))
    try:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        results = model.predict(
            source=str(source),
            refer_image=str(batch.canvas_path),
            visual_prompts={"bboxes": batch.boxes, "cls": batch.temporary_class_ids},
            predictor=YOLOEVPSegPredictor,
            imgsz=settings["imgsz"],
            conf=settings["conf"],
            iou=settings["iou"],
            device=settings["device"],
        )
        if not results:
            raise ValueError("model returned no result object")
        boxes_object = results[0].boxes
        xyxy = _as_numpy(boxes_object.xyxy)
        confidence = _as_numpy(boxes_object.conf).reshape(-1)
        prompt_ids = _as_numpy(boxes_object.cls).reshape(-1)
        if xyxy.size == 0:
            xyxy = xyxy.reshape((0, 4))
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise ValueError("boxes.xyxy must have shape (N, 4)")
        if not (len(xyxy) == len(confidence) == len(prompt_ids)):
            raise ValueError("YOLOE output arrays have unequal lengths")
        if not np.all(np.isfinite(xyxy)):
            raise ValueError("YOLOE returned non-finite box coordinates")
        if not np.all(np.isfinite(confidence)) or np.any(confidence < 0) or np.any(confidence > 1):
            raise ValueError("YOLOE returned invalid confidence values")
        if not np.all(np.isfinite(prompt_ids)) or not np.all(prompt_ids == np.floor(prompt_ids)):
            raise ValueError("YOLOE returned non-integral prompt IDs")
        predictions: list[Prediction] = []
        for box, score, raw_prompt_id in zip(xyxy, confidence, prompt_ids):
            prompt_id = int(raw_prompt_id)
            if prompt_id not in batch.prompt_to_dataset_class:
                raise ValueError(f"YOLOE returned unknown prompt ID {prompt_id}")
            dataset_id = batch.prompt_to_dataset_class[prompt_id]
            if dataset_id not in skus:
                raise ValueError(f"Prompt mapping refers to unknown dataset class {dataset_id}")
            item = skus[dataset_id]
            predictions.append(
                Prediction(
                    dataset_class_id=dataset_id,
                    prompt_class_id=prompt_id,
                    barcode=item.barcode,
                    sku_name=item.sku_name,
                    confidence=float(score),
                    xyxy=tuple(float(value) for value in box),
                    prompt_batch=batch.index,
                )
            )
        return predictions
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("YOLOE inference failed"):
            raise
        raise RuntimeError(f"YOLOE inference failed for {source} (batch {batch.index}): {exc}") from exc


def convert_xyxy_to_yolo(
    xyxy: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(xyxy) != 4:
        raise ValueError("XYXY box must contain four coordinates")
    coordinates = [float(value) for value in xyxy]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("XYXY coordinates must be finite")
    x1, y1, x2, y2 = coordinates
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box is empty after clipping to image bounds")
    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def _output_layout(output_root: Path) -> dict[str, Path]:
    _reject_reviewed_labels(Path(output_root))
    root = Path(output_root).resolve()
    _reject_reviewed_labels(root)
    raw = root / "raw_predictions"
    paths = {
        "root": root,
        "raw_predictions": raw,
        "labels": raw / "labels",
        "metadata": raw / "metadata",
        "previews": raw / "previews",
        "prompt_canvases": raw / "prompt_canvases",
        "summary_csv": raw / "summary.csv",
        "run_json": raw / "run.json",
        "provenance_json": raw / "provenance.json",
    }
    for key, path in paths.items():
        if key == "root":
            continue
        intended = _absolute_lexical(path)
        resolved = intended.resolve()
        if resolved != intended:
            raise ValueError(f"Output path is redirected by a symlink or junction: {path} -> {resolved}")
        try:
            resolved.relative_to(raw)
        except ValueError as exc:
            raise ValueError(f"Output path escapes selected raw_predictions: {path}") from exc
    return paths


def prepare_output_paths(output_root: Path) -> dict[str, Path]:
    paths = _output_layout(output_root)
    # Validate every destination before the first filesystem mutation.
    for key in ("labels", "metadata", "previews", "prompt_canvases", "summary_csv", "run_json", "provenance_json"):
        _assert_raw_destination(paths[key])
    for key in ("labels", "metadata", "previews", "prompt_canvases"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _assert_raw_destination(path: Path) -> Path:
    lexical = _absolute_lexical(Path(path))
    _reject_reviewed_labels(lexical)
    resolved = lexical.resolve()
    raw_parents = [parent for parent in (resolved.parent, *resolved.parents) if parent.name.casefold() == "raw_predictions"]
    if not raw_parents:
        raise ValueError(f"Output destination is not under raw_predictions: {path}")
    raw = raw_parents[0]
    try:
        resolved.relative_to(raw)
    except ValueError as exc:
        raise ValueError(f"Output destination escapes raw_predictions: {path}") from exc
    return resolved


def should_skip_image(image_stem: str, paths: Mapping[str, Path], overwrite: bool) -> bool:
    label = _assert_raw_destination(Path(paths["labels"]) / f"{image_stem}.txt")
    return label.exists() and not overwrite


def _atomic_write_text(path: Path, text: str, overwrite: bool) -> None:
    destination = _assert_raw_destination(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_yolo_labels(
    path: Path,
    predictions: Sequence[Prediction],
    image_width: int,
    image_height: int,
    overwrite: bool,
) -> None:
    ordered = sorted(predictions, key=lambda p: (p.dataset_class_id, -p.confidence, p.xyxy[0], p.xyxy[1]))
    rows = []
    for prediction in ordered:
        xc, yc, width, height = convert_xyxy_to_yolo(prediction.xyxy, image_width, image_height)
        rows.append(f"{prediction.dataset_class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
    text = "\n".join(rows) + ("\n" if rows else "")
    _atomic_write_text(Path(path), text, overwrite)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_metadata(
    path: Path,
    image_path: Path,
    predictions: Sequence[Prediction],
    run_context: Mapping[str, Any],
    overwrite: bool,
) -> None:
    image_path = _assert_input_allowed(Path(image_path))
    cv2 = _load_cv2()
    image = cv2.imread(str(image_path))
    if image is None or image.ndim < 2:
        raise ValueError(f"Cannot decode source image for metadata: {image_path}")
    payload = {
        "image_path": str(Path(image_path)),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "run_context": _json_ready(run_context),
        "predictions": [_json_ready(asdict(prediction)) for prediction in predictions],
    }
    _atomic_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n", overwrite)


def save_preview(path: Path, image: np.ndarray, predictions: Sequence[Prediction], overwrite: bool) -> None:
    destination = _assert_raw_destination(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Preview source image must have positive dimensions")
    cv2 = _load_cv2()
    preview = image.copy()
    height, width = preview.shape[:2]
    for prediction in sorted(predictions, key=lambda p: (p.dataset_class_id, -p.confidence)):
        x1, y1, x2, y2 = prediction.xyxy
        left = int(min(max(x1, 0), width - 1))
        top = int(min(max(y1, 0), height - 1))
        right = int(min(max(x2, 0), width - 1))
        bottom = int(min(max(y2, 0), height - 1))
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{prediction.dataset_class_id} {prediction.sku_name} {prediction.confidence:.2f}"
        cv2.putText(preview, label, (left, max(12, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    _atomic_write_image(destination, preview, cv2, overwrite)


def save_summary(
    csv_path: Path,
    json_path: Path,
    skus: Sequence[Sku],
    predictions_by_image: Mapping[Path, Sequence[Prediction]],
    counters: Mapping[str, int | float],
) -> None:
    threshold = float(counters.get("low_confidence_threshold", 0.5))
    rows: list[dict[str, Any]] = []
    for item in sorted(skus, key=lambda sku_item: sku_item.class_id):
        matches = [
            prediction
            for predictions in predictions_by_image.values()
            for prediction in predictions
            if prediction.dataset_class_id == item.class_id
        ]
        image_count = sum(any(p.dataset_class_id == item.class_id for p in predictions) for predictions in predictions_by_image.values())
        mean = sum(p.confidence for p in matches) / len(matches) if matches else None
        rows.append(
            {
                "class_id": item.class_id,
                "barcode": item.barcode,
                "brand": item.brand,
                "sku_name": item.sku_name,
                "candidate_count": len(matches),
                "image_count": image_count,
                "mean_confidence": mean,
                "low_confidence": bool(matches and mean is not None and mean < threshold),
                "zero_detection": not matches,
            }
        )
    fields = ["class_id", "barcode", "brand", "sku_name", "candidate_count", "image_count", "mean_confidence", "low_confidence", "zero_detection"]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(Path(csv_path), buffer.getvalue(), True)
    payload = {"classes": rows, "counters": _json_ready(counters)}
    _atomic_write_text(Path(json_path), json.dumps(payload, indent=2, sort_keys=True) + "\n", True)


def _positive_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw YOLO labels with YOLOE visual prompts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-enabled-count", type=_positive_arg)
    return parser.parse_args(argv)


def _dataset_value(dataset: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in dataset:
            return dataset[name]
    raise ValueError(f"Dataset config is missing one of: {', '.join(names)}")


def _resolve_model_value(config_path: Path, value: str) -> tuple[str, Path | None]:
    if not _is_model_path_like(value):
        return value, None
    resolved = _assert_input_allowed(resolve_config_path(config_path, value))
    return str(resolved), resolved


def _file_sha256(path: Path) -> str:
    source = _assert_input_allowed(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot checksum reference image {source}: {exc}") from exc
    return digest.hexdigest()


def _build_provenance(
    model_path: str,
    settings: Mapping[str, Any],
    active_skus: Sequence[Sku],
    references: Sequence[ReferencePrompt],
) -> dict[str, Any]:
    ordered_skus = sorted(active_skus, key=lambda item: item.class_id)
    active_ids = {item.class_id for item in ordered_skus}
    batch_size = _positive_integer(settings["prompt_batch_size"], "yoloe.prompt_batch_size")
    mappings = [
        {
            str(temporary_id): item.class_id
            for temporary_id, item in enumerate(ordered_skus[offset : offset + batch_size])
        }
        for offset in range(0, len(ordered_skus), batch_size)
    ]
    reference_records = [
        {
            "class_id": prompt.class_id,
            "path": str(_assert_input_allowed(prompt.image_path)),
            "bbox": list(prompt.bbox) if prompt.bbox is not None else None,
            "sha256": _file_sha256(prompt.image_path),
        }
        for prompt in references
        if prompt.class_id in active_ids
    ]
    material = {
        "schema_version": 1,
        "model": model_path,
        "settings": {
            key: _json_ready(settings[key])
            for key in ("imgsz", "conf", "iou", "device", "prompt_batch_size", "canvas_cell_size", "canvas_padding")
        },
        "active_class_ids": [item.class_id for item in ordered_skus],
        "batch_mappings": mappings,
        "references": reference_records,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {**material, "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _read_provenance(path: Path) -> dict[str, Any] | None:
    source = _assert_raw_destination(path)
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read existing provenance {source}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("fingerprint"), str):
        raise ValueError(f"Existing provenance is malformed: {source}")
    return data


def _validate_completion_markers(
    markers: Sequence[Path],
    labels_dir: Path,
    raw_root: Path,
    input_files: Sequence[Path],
    allow_link_removal: bool,
) -> None:
    lexical_labels = _absolute_lexical(labels_dir)
    resolved_labels = lexical_labels.resolve()
    resolved_raw = _absolute_lexical(raw_root).resolve()
    try:
        resolved_labels.relative_to(resolved_raw)
    except ValueError as exc:
        raise ValueError(f"Selected labels directory escapes raw_predictions: {labels_dir}") from exc
    safe_inputs = [_assert_input_allowed(path) for path in input_files]
    for marker in markers:
        lexical_marker = _absolute_lexical(marker)
        if lexical_marker.parent != lexical_labels or lexical_marker.parent.resolve() != resolved_labels:
            raise ValueError(f"Completion marker is outside the selected labels directory: {marker}")
        if not os.path.lexists(lexical_marker):
            continue
        if lexical_marker.is_symlink():
            if not allow_link_removal:
                raise ValueError(f"Completion marker is a symbolic link: {marker}")
            # Do not resolve or compare the target: unlinking this lexical entry
            # leaves any external or input target untouched.
            continue
        if any(_same_existing_file(lexical_marker, source) for source in safe_inputs):
            raise ValueError(f"Completion marker aliases an input file: {marker}")


def _invalidate_labels(
    labels: Sequence[Path], labels_dir: Path, raw_root: Path, input_files: Sequence[Path]
) -> None:
    _validate_completion_markers(labels, labels_dir, raw_root, input_files, True)
    for label in labels:
        lexical_marker = _absolute_lexical(label)
        if os.path.lexists(lexical_marker):
            lexical_marker.unlink()


def _all_lexical_label_markers(labels_dir: Path, raw_root: Path) -> list[Path]:
    lexical_labels = _absolute_lexical(labels_dir)
    resolved_labels = lexical_labels.resolve()
    resolved_raw = _absolute_lexical(raw_root).resolve()
    try:
        resolved_labels.relative_to(resolved_raw)
    except ValueError as exc:
        raise ValueError(f"Selected labels directory escapes raw_predictions: {labels_dir}") from exc
    if not lexical_labels.is_dir():
        return []
    markers: list[Path] = []
    with os.scandir(lexical_labels) as entries:
        for entry in entries:
            if not entry.name.casefold().endswith(".txt"):
                continue
            if entry.is_symlink() or entry.is_file(follow_symlinks=False):
                markers.append(lexical_labels / entry.name)
    return sorted(markers, key=lambda path: (path.name.casefold(), path.name))


def _lexical_path_key(path: Path) -> str:
    return os.path.normcase(os.fspath(_absolute_lexical(path)))


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _validate_input_output_separation(
    paths: Mapping[str, Path],
    input_files: Sequence[Path],
    input_directories: Sequence[Path],
    planned_outputs: Sequence[Path],
) -> None:
    output_root = Path(paths["root"]).resolve()
    raw = Path(paths["raw_predictions"]).resolve()
    safe_inputs = [_assert_input_allowed(path) for path in input_files]
    safe_directories = [_assert_input_allowed(path) for path in input_directories]
    for source in safe_inputs:
        if _is_within(source, output_root):
            raise ValueError(f"Input file overlaps the output root: {source}")
    for source_directory in safe_directories:
        if _is_within(output_root, source_directory) or _is_within(source_directory, output_root):
            raise ValueError(f"Input directory overlaps the output root: {source_directory}")
    for destination in planned_outputs:
        resolved = _assert_raw_destination(destination)
        try:
            resolved.relative_to(raw)
        except ValueError as exc:
            raise ValueError(f"Output destination escapes selected raw_predictions: {destination}") from exc
        _reject_output_alias(destination, safe_inputs)


def _validate_references_for_active(
    references: Sequence[ReferencePrompt], active_skus: Sequence[Sku], skus: Mapping[int, Sku]
) -> None:
    cv2 = _load_cv2()
    active_ids = {item.class_id for item in active_skus}
    seen: set[int] = set()
    for prompt in references:
        if prompt.class_id not in active_ids:
            continue
        _assert_input_allowed(prompt.image_path)
        if not prompt.image_path.is_file():
            raise ValueError(f"Reference image is missing: {prompt.image_path}")
        image = cv2.imread(str(prompt.image_path))
        if image is None:
            raise ValueError(f"Reference image cannot be decoded: {prompt.image_path}")
        validate_reference(prompt, image, skus)
        seen.add(prompt.class_id)
    missing = sorted(active_ids - seen)
    if missing:
        raise ValueError(f"Missing valid references for enabled class IDs: {missing}")


def _git_info(config_path: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {"branch": None, "commit": None}
    for key, arguments in (("branch", ["branch", "--show-current"]), ("commit", ["rev-parse", "HEAD"])):
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=config_path.parent, capture_output=True, text=True, timeout=5, check=False
            )
            value = completed.stdout.strip()
            result[key] = value or None
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _ultralytics_version() -> str | None:
    try:
        import ultralytics

        return getattr(ultralytics, "__version__", None)
    except ImportError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = time.monotonic()
    try:
        config_path = _assert_input_allowed(Path(args.config))
        config = load_config(config_path)
        dataset = config["dataset"]
        yoloe_settings = config["yoloe"]
        output_settings = config["output"]
        pilot = config["pilot"]
        overwrite = bool(output_settings.get("overwrite", False)) if args.overwrite is None else bool(args.overwrite)

        manifest_path = resolve_config_path(config_path, _dataset_value(dataset, "manifest", "sku_manifest"))
        references_path = resolve_config_path(config_path, _dataset_value(dataset, "references", "reference_definitions"))
        images_root = resolve_config_path(config_path, _dataset_value(dataset, "images", "shelf_images", "image_root"))
        extensions = dataset.get("extensions", dataset.get("image_extensions", ["jpg", "jpeg", "png"]))
        if not isinstance(extensions, (list, tuple)):
            raise ValueError("dataset.extensions must be a list")
        output_value = args.output if args.output is not None else output_settings.get("root")
        if output_value is None:
            raise ValueError("output.root is required unless --output is supplied")
        output_root = resolve_config_path(config_path, str(output_value))
        paths = _output_layout(output_root)

        skus = load_sku_manifest(manifest_path)
        references = load_reference_definitions(references_path, skus, config_path.parent)
        active_skus = select_active_skus(skus, pilot)
        enabled_count = sum(item.enabled for item in skus.values())
        if args.require_enabled_count is not None and enabled_count != args.require_enabled_count:
            raise ValueError(f"Enabled SKU count is {enabled_count}, expected {args.require_enabled_count}")
        shelf_images = discover_shelf_images(images_root, extensions)
        if pilot.get("enabled", False):
            shelf_images = shelf_images[: _positive_integer(pilot.get("max_images"), "pilot.max_images")]
        configured_model = str(yoloe_settings.get("model", yoloe_settings.get("model_path", "")))
        model_path, model_input = _resolve_model_value(config_path, configured_model)
        batch_count = math.ceil(len(active_skus) / _positive_integer(yoloe_settings["prompt_batch_size"], "yoloe.prompt_batch_size"))
        planned_outputs = [paths["summary_csv"], paths["run_json"], paths["provenance_json"]]
        planned_outputs.extend(paths["prompt_canvases"] / f"batch_{index:03d}.png" for index in range(batch_count))
        completion_markers: list[Path] = []
        for image_path in shelf_images:
            completion_markers.append(paths["labels"] / f"{image_path.stem}.txt")
            planned_outputs.extend(
                (
                    paths["metadata"] / f"{image_path.stem}.json",
                    paths["previews"] / f"{image_path.stem}{image_path.suffix}",
                )
            )
        input_files = [config_path, manifest_path, references_path, *[prompt.image_path for prompt in references], *shelf_images]
        if model_input is not None:
            input_files.append(model_input)
        _validate_input_output_separation(paths, input_files, [images_root], planned_outputs)
        _validate_completion_markers(
            completion_markers,
            paths["labels"],
            paths["raw_predictions"],
            input_files,
            overwrite,
        )
        _validate_references_for_active(references, active_skus, skus)
        provenance = _build_provenance(model_path, yoloe_settings, active_skus, references)
        completed_labels = [marker for marker in completion_markers if os.path.lexists(marker)]
        all_label_markers = _all_lexical_label_markers(paths["labels"], paths["raw_predictions"])
        selected_marker_keys = {_lexical_path_key(marker) for marker in completion_markers}
        retained_labels = [
            marker
            for marker in all_label_markers
            if not (overwrite and _lexical_path_key(marker) in selected_marker_keys)
        ]
        existing_provenance = _read_provenance(paths["provenance_json"])
        matching_resume = bool(retained_labels)
        if matching_resume and (
            existing_provenance is None
            or existing_provenance.get("fingerprint") != provenance["fingerprint"]
        ):
            raise ValueError(
                "Existing labels were produced with missing or incompatible provenance; "
                "use --overwrite to regenerate them"
            )
        if args.validate_only:
            print(f"Validation successful: {len(active_skus)} SKU(s), {len(shelf_images)} image(s)")
            return 0

        paths = prepare_output_paths(output_root)
        if overwrite:
            _invalidate_labels(completed_labels, paths["labels"], paths["raw_predictions"], input_files)
        if not matching_resume:
            _atomic_write_text(
                paths["provenance_json"],
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                True,
            )
        batch_settings = dict(yoloe_settings)
        batch_settings["_reuse_existing_canvases"] = matching_resume
        batches = build_prompt_batches(references, active_skus, batch_settings, paths["prompt_canvases"])
        model = load_yoloe(model_path)
        save_metadata_enabled = bool(output_settings.get("save_metadata", True))
        save_previews_enabled = bool(output_settings.get("save_previews", True))
        run_context = {
            "model": model_path,
            "ultralytics_version": _ultralytics_version(),
            **{key: yoloe_settings.get(key) for key in ("imgsz", "conf", "iou", "device")},
        }
        counters: dict[str, int | float] = {"processed": 0, "skipped": 0, "errors": 0, "no_detection_images": 0}
        error_details: list[dict[str, str]] = []
        predictions_by_image: dict[Path, Sequence[Prediction]] = {}
        cv2 = _load_cv2()
        for image_path in shelf_images:
            if should_skip_image(image_path.stem, paths, overwrite):
                counters["skipped"] = int(counters["skipped"]) + 1
                continue
            try:
                image = cv2.imread(str(image_path))
                if image is None or image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
                    raise ValueError(f"Cannot decode shelf image: {image_path}")
                predictions = [prediction for batch in batches for prediction in run_yoloe(model, image_path, batch, yoloe_settings, skus)]
                if not predictions:
                    counters["no_detection_images"] = int(counters["no_detection_images"]) + 1
                if save_metadata_enabled:
                    save_metadata(paths["metadata"] / f"{image_path.stem}.json", image_path, predictions, run_context, True)
                if save_previews_enabled:
                    save_preview(paths["previews"] / f"{image_path.stem}{image_path.suffix}", image, predictions, True)
                save_yolo_labels(paths["labels"] / f"{image_path.stem}.txt", predictions, image.shape[1], image.shape[0], overwrite)
                predictions_by_image[image_path] = predictions
                counters["processed"] = int(counters["processed"]) + 1
            except Exception as exc:
                counters["errors"] = int(counters["errors"]) + 1
                error_details.append({"image_path": str(image_path), "error": str(exc)})
                print(f"ERROR {image_path}: {exc}", file=sys.stderr)
        counters["low_confidence_threshold"] = float(
            output_settings.get(
                "low_confidence_threshold",
                output_settings.get("low_confidence_cutoff", yoloe_settings.get("conf", 0.5)),
            )
        )
        save_summary(paths["summary_csv"], paths["run_json"], active_skus, predictions_by_image, counters)
        run_payload = json.loads(paths["run_json"].read_text(encoding="utf-8"))
        run_payload.update(
            {
                "command": [sys.executable, *sys.argv] if argv is None else [sys.executable, *argv],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git": _git_info(config_path),
                "config": _json_ready(config),
                "batches": [
                    {
                        "index": batch.index,
                        "dataset_class_ids": list(batch.dataset_class_ids),
                        "prompt_to_dataset_class": batch.prompt_to_dataset_class,
                        "canvas_path": str(batch.canvas_path),
                    }
                    for batch in batches
                ],
                "settings": _json_ready(run_context),
                "effective_paths": {
                    "config": str(config_path),
                    "manifest": str(manifest_path),
                    "references": str(references_path),
                    "shelf_images": str(images_root),
                    "output_root": str(output_root),
                    "model": model_path,
                },
                "cli_overrides": {"output": args.output, "overwrite": bool(args.overwrite)},
                "errors": error_details,
                "elapsed_seconds": max(0.0, time.monotonic() - started_at),
                "provenance_fingerprint": provenance["fingerprint"],
            }
        )
        _atomic_write_text(paths["run_json"], json.dumps(run_payload, indent=2, sort_keys=True) + "\n", True)
        return 0 if counters["errors"] == 0 else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

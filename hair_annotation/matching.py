"""Reference-view derivation and conservative OpenCLIP SKU matching."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import MatchingConfig
from .types import Candidate, ClassifiedCandidate, RankedSku


class EmbeddingBackend(Protocol):
    """Minimal embedding interface used by prototype and candidate matching."""

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class PrototypeBank:
    """Ordered permanent SKU identities and their normalized prototypes."""

    class_ids: tuple[int, ...]
    barcodes: tuple[str, ...]
    sku_names: tuple[str, ...]
    visual_embeddings: np.ndarray
    text_embeddings: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.class_ids)
        if len(self.barcodes) != count or len(self.sku_names) != count:
            raise ValueError("PrototypeBank identity fields must have equal lengths")
        visual = np.array(self.visual_embeddings, dtype=np.float64, copy=True)
        text = np.array(self.text_embeddings, dtype=np.float64, copy=True)
        if visual.ndim != 2 or text.ndim != 2:
            raise ValueError(
                "PrototypeBank embeddings must be two-dimensional matrices"
            )
        if visual.shape[0] != count or text.shape[0] != count:
            raise ValueError("PrototypeBank embeddings must have one row per SKU")
        if visual.shape[1] != text.shape[1]:
            raise ValueError("PrototypeBank has inconsistent embedding dimensions")
        visual.flags.writeable = False
        text.flags.writeable = False
        object.__setattr__(self, "visual_embeddings", visual)
        object.__setattr__(self, "text_embeddings", text)


class OpenClipBackend:
    """Thin, lazily imported OpenCLIP embedding boundary."""

    def __init__(self, model_name: str, pretrained: str, device: str | int):
        import open_clip
        import torch
        from PIL import Image

        self._torch = torch
        self._image_type = Image
        self._device = f"cuda:{device}" if isinstance(device, int) else str(device)
        if self._device.startswith("cuda:"):
            cuda = torch.cuda
            available = bool(cuda.is_available())
            index_text = self._device.split(":", 1)[1]
            try:
                index = int(index_text)
            except ValueError as exc:
                raise RuntimeError(
                    f"CUDA device '{self._device}' is invalid; expected cuda:N"
                ) from exc
            device_count = getattr(cuda, "device_count", None)
            in_range = not callable(device_count) or 0 <= index < device_count()
            if not available or not in_range:
                raise RuntimeError(
                    f"CUDA device '{self._device}' is unavailable; choose an available "
                    "CUDA device or use device='cpu'."
                )

        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load OpenCLIP model '{model_name}' with pretrained weights "
                f"'{pretrained}': {exc}. Make the model and weights available locally "
                "or allow their download before running."
            ) from exc

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if not images_rgb:
            raise ValueError("images_rgb must contain at least one image")
        tensors = [
            self._preprocess(self._image_type.fromarray(np.asarray(image)))
            for image in images_rgb
        ]
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.inference_mode():
            encoded = self._model.encode_image(batch)
        return _cpu_float64_array(encoded)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must contain at least one prompt")
        tokens = self._tokenizer(list(texts)).to(self._device)
        with self._torch.inference_mode():
            encoded = self._model.encode_text(tokens)
        return _cpu_float64_array(encoded)


def _cpu_float64_array(value: Any) -> np.ndarray:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current, dtype=np.float64)


def _validate_image(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{label} must be an HxWx3 BGR image")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{label} must have positive dimensions")
    return array


def _finite_box(candidate: Candidate) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(value) for value in candidate.xyxy)
        confidence = float(candidate.localization_confidence)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*values, confidence)):
        return None
    return values


def _pixel_box(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy
    clipped_x1 = max(0, min(width, math.floor(x1)))
    clipped_y1 = max(0, min(height, math.floor(y1)))
    clipped_x2 = max(0, min(width, math.ceil(x2)))
    clipped_y2 = max(0, min(height, math.ceil(y2)))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _reference_rank_key(
    candidate: Candidate, image_width: int, image_height: int
) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in candidate.xyxy)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    normalized_distance = math.hypot(
        center_x - image_width / 2.0,
        center_y - image_height / 2.0,
    ) / math.hypot(image_width, image_height)
    return (
        -float(candidate.localization_confidence),
        normalized_distance,
        x1,
        y1,
        x2,
        y2,
    )


def derive_reference_views(
    reference_images_bgr: Mapping[int, np.ndarray],
    reference_candidates: Mapping[int, Sequence[Candidate]],
    max_views: int,
) -> tuple[dict[int, list[np.ndarray]], list[dict[str, Any]]]:
    """Select deterministic product crops, falling back to a copied full image."""
    if type(max_views) is not int or max_views <= 0:
        raise ValueError("max_views must be a positive integer")

    views: dict[int, list[np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    for class_id in sorted(reference_images_bgr):
        image = _validate_image(
            reference_images_bgr[class_id], f"reference image for class {class_id}"
        )
        height, width = image.shape[:2]
        valid_candidates = [
            candidate
            for candidate in reference_candidates.get(class_id, ())
            if _finite_box(candidate) is not None
        ]
        valid_candidates.sort(
            key=lambda candidate: _reference_rank_key(candidate, width, height)
        )

        class_views: list[np.ndarray] = []
        for candidate in valid_candidates:
            finite_box = _finite_box(candidate)
            assert finite_box is not None
            box = _pixel_box(finite_box, width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            class_views.append(image[y1:y2, x1:x2, ::-1].copy())
            if len(class_views) == max_views:
                break

        derived_count = len(class_views)
        fallback = derived_count == 0
        if fallback:
            class_views = [image[:, :, ::-1].copy()]
        views[class_id] = class_views
        diagnostics.append(
            {
                "class_id": class_id,
                "derived_views": derived_count,
                "fallback_full_image": fallback,
            }
        )
    return views, diagnostics


def _normalized_rows(value: Any, label: str, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ValueError(
            f"{label} must be a 2D matrix with {expected_rows} embedding rows"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite {label}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"zero-norm {label}")
    return matrix / norms


def _sku_identity(row: Mapping[str, Any]) -> tuple[int, str, str]:
    try:
        class_id = row["class_id"]
        barcode = row["barcode"]
        sku_name = row["sku_name"]
    except KeyError as exc:
        raise ValueError(f"SKU row is missing required field {exc.args[0]!r}") from exc
    if type(class_id) is not int:
        raise ValueError("SKU class_id must be an integer")
    if not isinstance(barcode, str) or not barcode.strip():
        raise ValueError(f"SKU {class_id} barcode must be a non-empty string")
    if not isinstance(sku_name, str) or not sku_name.strip():
        raise ValueError(f"SKU {class_id} sku_name must be a non-empty string")
    return class_id, barcode.strip(), sku_name.strip()


def build_prototypes(
    backend: EmbeddingBackend,
    sku_rows: Sequence[Mapping[str, Any]],
    views_rgb: Mapping[int, Sequence[np.ndarray]],
    config: MatchingConfig,
) -> PrototypeBank:
    """Build one normalized visual and text prototype per permanent SKU."""
    identities = sorted(
        (_sku_identity(row) for row in sku_rows), key=lambda item: item[0]
    )
    class_ids = [identity[0] for identity in identities]
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("duplicate class ID in SKU rows")
    if config.needs_review_class_id in class_ids:
        raise ValueError(
            f"reserved class ID {config.needs_review_class_id} cannot be a "
            "permanent SKU"
        )
    if not identities:
        raise ValueError("at least one permanent SKU is required")

    all_views: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for class_id in class_ids:
        sku_views = list(views_rgb.get(class_id, ()))
        if not sku_views:
            raise ValueError(f"missing reference views for class ID {class_id}")
        start = len(all_views)
        all_views.extend(sku_views)
        spans.append((start, len(all_views)))

    normalized_views = _normalized_rows(
        backend.encode_images(all_views), "image embeddings", len(all_views)
    )
    visual_rows: list[np.ndarray] = []
    for start, end in spans:
        mean = normalized_views[start:end].mean(axis=0, keepdims=True)
        visual_rows.append(_normalized_rows(mean, "visual prototypes", 1)[0])
    visual_embeddings = np.stack(visual_rows)

    texts = [
        config.text_template.format(english_sku_name=sku_name)
        for _, _, sku_name in identities
    ]
    text_embeddings = _normalized_rows(
        backend.encode_texts(texts), "text embeddings", len(identities)
    )
    if visual_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "inconsistent embedding dimensions between image and text embeddings"
        )

    return PrototypeBank(
        class_ids=tuple(class_ids),
        barcodes=tuple(identity[1] for identity in identities),
        sku_names=tuple(identity[2] for identity in identities),
        visual_embeddings=visual_embeddings,
        text_embeddings=text_embeddings,
    )


def _candidate_crop(image_bgr: np.ndarray, candidate: Candidate) -> np.ndarray:
    finite_box = _finite_box(candidate)
    if finite_box is None:
        raise ValueError("candidate box and localization confidence must be finite")
    height, width = image_bgr.shape[:2]
    box = _pixel_box(finite_box, width, height)
    if box is None:
        raise ValueError(
            f"candidate box {candidate.xyxy!r} has no area inside the image"
        )
    x1, y1, x2, y2 = box
    return image_bgr[y1:y2, x1:x2, ::-1].copy()


def classify_candidates(
    backend: EmbeddingBackend,
    image_bgr: np.ndarray,
    candidates: Sequence[Candidate],
    bank: PrototypeBank,
    config: MatchingConfig,
) -> list[ClassifiedCandidate]:
    """Rank SKU prototypes and conservatively assign or defer each candidate."""
    image = _validate_image(image_bgr, "image_bgr")
    if not candidates:
        return []
    if not bank.class_ids:
        raise ValueError("PrototypeBank must contain at least one SKU")

    crops = [_candidate_crop(image, candidate) for candidate in candidates]
    query_embeddings = _normalized_rows(
        backend.encode_images(crops), "candidate image embeddings", len(candidates)
    )
    if query_embeddings.shape[1] != bank.visual_embeddings.shape[1]:
        raise ValueError("inconsistent embedding dimensions for candidate images")

    visual_similarities = query_embeddings @ bank.visual_embeddings.T
    text_similarities = query_embeddings @ bank.text_embeddings.T
    scores = (
        config.visual_weight * visual_similarities
        + config.text_weight * text_similarities
    )

    classified: list[ClassifiedCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        order = sorted(
            range(len(bank.class_ids)),
            key=lambda sku_index: (
                -float(scores[candidate_index, sku_index]),
                bank.class_ids[sku_index],
            ),
        )
        ranking_values = tuple(
            RankedSku(
                class_id=bank.class_ids[sku_index],
                barcode=bank.barcodes[sku_index],
                sku_name=bank.sku_names[sku_index],
                score=float(scores[candidate_index, sku_index]),
                visual_similarity=float(
                    visual_similarities[candidate_index, sku_index]
                ),
                text_similarity=float(text_similarities[candidate_index, sku_index]),
            )
            for sku_index in order[:3]
        )
        top_index = order[0]
        top_score = float(scores[candidate_index, top_index])
        margin = (
            top_score - float(scores[candidate_index, order[1]])
            if len(order) > 1
            else math.inf
        )
        if top_score < config.min_score:
            accepted = False
            reason = "score_below_threshold"
        elif margin < config.min_margin:
            accepted = False
            reason = "top_one_margin_below_threshold"
        else:
            accepted = True
            reason = "accepted"

        classified.append(
            ClassifiedCandidate(
                candidate=candidate,
                assigned_class_id=(
                    bank.class_ids[top_index]
                    if accepted
                    else config.needs_review_class_id
                ),
                assigned_name=bank.sku_names[top_index] if accepted else "Needs Review",
                accepted=accepted,
                review_reason=reason,
                rankings=ranking_values,
            )
        )
    return classified

# Ultralytics Platform Box-First Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the direct 89-class visual-prompt run with a runtime-local, box-first annotation assistant that exports tight product candidates and conservative English SKU suggestions for review in Ultralytics Platform.

**Architecture:** Keep `yoloe_autolabel.py` as the CLI and safety/orchestration boundary. Put tile geometry and YOLOE text localization in `hair_annotation/localization.py`, and reference-view/OpenCLIP matching in `hair_annotation/matching.py`; keep archive construction in `hair_annotation/export.py`. The CLI loads the existing manifest and references, runs generic localization before SKU matching, writes review artifacts, and delegates strict Platform packaging to `colab_runtime.py`.

**Tech Stack:** Python 3.12, Ultralytics `8.4.149`, YOLOE-26L segmentation, OpenCLIP `3.3.0`, PyTorch, NumPy, OpenCV, Pillow, PyYAML, pytest, YOLO detection labels.

**Spec:** `docs/superpowers/specs/2026-09-15-platform-box-first-annotation-design.md`

## Global Constraints

- Run entirely under Colab Enterprise runtime storage; never mount Google Drive or require Cloud Storage/bucket access.
- Keep `yoloe_autolabel.py` as the executable entrypoint and preserve `--config`, `--validate-only`, `--overwrite`, `--output`, and `--require-enabled-count`.
- Treat the workbook, 89 product references, 632 shelf images, and reviewed labels as read-only inputs; automated code must never access a path component named `reviewed_labels`.
- Use permanent class IDs 0 through 88 from `sku_manifest.csv`; use class ID 89 only as temporary `Needs Review`.
- Pilot matching uses all 89 enabled SKUs and only limits the shelf-image count to 10.
- Default localization uses `yoloe-26l-seg.pt`, 1024-pixel tiles, 20 percent overlap, and the five approved English generic prompts.
- Default matching uses OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k`, visual weight 0.80, text weight 0.20, minimum score 0.24, and minimum top-one margin 0.02.
- Preserve original shelf-image bytes and filenames in the Platform archive.
- Produce a deterministic 90/10 train/validation split and a strict YOLO archive containing only `data.yaml`, `images/`, and `labels/`.
- Keep review CSV, JSON, previews, summaries, and provenance in a separate review archive.
- Use atomic file/archive replacement and require explicit overwrite for an existing destination.
- Work on `main` only; do not create a worktree.

## File Structure

- Create `hair_annotation/__init__.py`: stable exports for the new annotation package.
- Create `hair_annotation/types.py`: immutable tile, localization candidate, ranked SKU, and classified-candidate records.
- Create `hair_annotation/config.py`: validated box-first configuration dataclasses and defaults.
- Create `hair_annotation/localization.py`: tiling, mask/box conversion, geometry filtering, NMS, and the YOLOE text-prompt adapter.
- Create `hair_annotation/matching.py`: automatic reference-view derivation, OpenCLIP embedding adapter, prototype construction, ranking, and uncertainty gating.
- Create `hair_annotation/export.py`: deterministic split calculation, `data.yaml`, strict Platform ZIP, review ZIP, and archive validation.
- Modify `yoloe_autolabel.py`: keep existing input/output safety, replace the visual-prompt execution path with box-first orchestration, and write review diagnostics.
- Modify `colab_runtime.py`: generate a 10-image/all-SKU pilot config and package both result archives.
- Modify `colab/config.yaml`: replace prompt-canvas controls with localization, matching, and export settings.
- Modify `requirements.txt`: pin OpenCLIP and Pillow while retaining the working Ultralytics pin.
- Modify `prepare_hair_colab.py`: include the new package and tests in `hair_colab_runtime.zip`.
- Modify `colab/hair_colab_enterprise.ipynb`: keep the upload/extract/validate/pilot/full-run flow and expose both downloads.
- Modify `colab/README.md` and `README.md`: document the two-stage behavior and Platform review workflow.
- Create `tests/test_box_first_config.py`, `tests/test_localization.py`, `tests/test_matching.py`, and `tests/test_platform_export.py`.
- Modify `tests/test_yoloe_autolabel.py`, `tests/test_colab_assets.py`, and `tests/test_prepare_hair_colab.py` for CLI and package integration.

---

### Task 1: Box-First Types and Configuration

**Files:**
- Create: `hair_annotation/__init__.py`
- Create: `hair_annotation/types.py`
- Create: `hair_annotation/config.py`
- Create: `tests/test_box_first_config.py`
- Modify: `yoloe_autolabel.py:65-163`

**Interfaces:**
- Consumes: the `localization`, `matching`, and `export` mappings from loaded YAML.
- Produces: `BoxFirstConfig.from_mapping(data: Mapping[str, Any]) -> BoxFirstConfig`; `Tile`, `Candidate`, `RankedSku`, and `ClassifiedCandidate` records used by every later task.

- [ ] **Step 1: Write failing configuration and record tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from hair_annotation.config import BoxFirstConfig
from hair_annotation.types import Candidate, Tile


def valid_sections():
    return {
        "localization": {
            "tile_size": 1024,
            "overlap": 0.20,
            "prompts": [
                "shampoo bottle",
                "conditioner bottle",
                "hair treatment pouch",
                "boxed hair product",
                "hair care multipack",
            ],
            "conf": 0.10,
            "iou": 0.50,
            "min_side": 12,
            "max_area_ratio": 0.10,
            "min_aspect_ratio": 0.15,
            "max_aspect_ratio": 4.0,
            "nms_iou": 0.50,
        },
        "matching": {
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "visual_weight": 0.80,
            "text_weight": 0.20,
            "min_score": 0.24,
            "min_margin": 0.02,
            "max_reference_views": 3,
            "needs_review_class_id": 89,
            "text_template": "a retail hair-care product package of {english_sku_name}",
        },
        "export": {"train_fraction": 0.90, "split_seed": "hair-osa-v1"},
    }


def test_box_first_defaults_are_parsed_and_immutable():
    config = BoxFirstConfig.from_mapping(valid_sections())
    assert config.localization.tile_size == 1024
    assert config.localization.prompts[0] == "shampoo bottle"
    assert config.matching.needs_review_class_id == 89
    assert config.export.train_fraction == 0.90
    with pytest.raises(FrozenInstanceError):
        config.localization.tile_size = 640


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("localization", "overlap", 1.0, "localization.overlap"),
        ("localization", "max_area_ratio", 0.0, "localization.max_area_ratio"),
        ("matching", "visual_weight", 0.7, "sum to 1"),
        ("matching", "needs_review_class_id", 88, "must be 89"),
        ("matching", "text_template", "hair product", "english_sku_name"),
        ("export", "train_fraction", 1.0, "export.train_fraction"),
    ],
)
def test_invalid_box_first_config_is_rejected(section, key, value, message):
    data = valid_sections()
    data[section][key] = value
    with pytest.raises(ValueError, match=message):
        BoxFirstConfig.from_mapping(data)


def test_candidate_records_original_coordinates():
    tile = Tile(index=2, x=820, y=0, width=1024, height=900)
    candidate = Candidate((900.0, 30.0, 980.0, 240.0), 0.71, "shampoo bottle", 2, True)
    assert tile.x == 820
    assert candidate.xyxy == (900.0, 30.0, 980.0, 240.0)
```

- [ ] **Step 2: Run the new tests and verify the package is absent**

Run: `.venv/bin/pytest -q tests/test_box_first_config.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'hair_annotation'`.

- [ ] **Step 3: Add immutable records and strict configuration parsing**

```python
# hair_annotation/types.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    xyxy: tuple[float, float, float, float]
    localization_confidence: float
    prompt_name: str
    tile_index: int
    mask_used: bool


@dataclass(frozen=True)
class RankedSku:
    class_id: int
    barcode: str
    sku_name: str
    score: float
    visual_similarity: float
    text_similarity: float


@dataclass(frozen=True)
class ClassifiedCandidate:
    candidate: Candidate
    assigned_class_id: int
    assigned_name: str
    accepted: bool
    review_reason: str
    rankings: tuple[RankedSku, ...]
```

Implement frozen `LocalizationConfig`, `MatchingConfig`, `ExportConfig`, and `BoxFirstConfig` dataclasses in `hair_annotation/config.py`. `from_mapping` must require every key shown in `valid_sections()`, reject booleans as numbers, require `0 <= overlap < 1`, require all probability-like values to be finite and within their stated open/closed intervals, require positive integer sizes, require a non-empty sequence of unique non-blank prompt strings, require `visual_weight + text_weight == 1.0` within `1e-9`, require `needs_review_class_id == 89`, and require `{english_sku_name}` in the text template. Export these records from `hair_annotation/__init__.py`.

Update `_REQUIRED_SECTIONS` and `_validate_config` in `yoloe_autolabel.py` so loading `colab/config.yaml` also calls `BoxFirstConfig.from_mapping(data)`. Retain project, dataset, output, and pilot safety checks, but change YOLOE validation to require only non-empty `model`, positive integer `imgsz`, and a non-negative integer or non-empty string `device`; `conf` and `iou` now belong to `localization`, and the visual-prompt canvas keys are no longer required.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/test_box_first_config.py tests/test_yoloe_autolabel.py::ConfigAndReferenceTests`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the configuration boundary**

```bash
git add hair_annotation/__init__.py hair_annotation/types.py hair_annotation/config.py tests/test_box_first_config.py yoloe_autolabel.py
git commit -m "feat: add box-first annotation configuration"
```

### Task 2: Tile Geometry, Mask Boxes, Filters, and NMS

**Files:**
- Create: `hair_annotation/localization.py`
- Create: `tests/test_localization.py`

**Interfaces:**
- Consumes: `Tile`, `Candidate`, and `LocalizationConfig` from Task 1.
- Produces: `generate_tiles(image_width: int, image_height: int, tile_size: int, overlap: float) -> list[Tile]`; `mask_to_box(points: np.ndarray) -> tuple[float, float, float, float] | None`; `filter_candidates(candidates, image_width, image_height, config) -> tuple[list[Candidate], dict[str, int]]`; `class_agnostic_nms(candidates, iou_threshold) -> tuple[list[Candidate], int]`.

- [ ] **Step 1: Write failing deterministic geometry tests**

```python
import numpy as np

from hair_annotation.config import LocalizationConfig
from hair_annotation.localization import (
    class_agnostic_nms,
    filter_candidates,
    generate_tiles,
    mask_to_box,
)
from hair_annotation.types import Candidate


def localization_config():
    return LocalizationConfig(
        tile_size=1024,
        overlap=0.20,
        prompts=("shampoo bottle", "conditioner bottle", "hair treatment pouch", "boxed hair product", "hair care multipack"),
        conf=0.10,
        iou=0.50,
        min_side=12,
        max_area_ratio=0.10,
        min_aspect_ratio=0.15,
        max_aspect_ratio=4.0,
        nms_iou=0.50,
    )


def test_tiles_cover_right_and_bottom_edges_without_duplicates():
    tiles = generate_tiles(1600, 1200, 1024, 0.20)
    assert [(t.x, t.y, t.width, t.height) for t in tiles] == [
        (0, 0, 1024, 1024),
        (576, 0, 1024, 1024),
        (0, 176, 1024, 1024),
        (576, 176, 1024, 1024),
    ]


def test_small_image_produces_one_exact_tile():
    tiles = generate_tiles(640, 480, 1024, 0.20)
    assert [(t.x, t.y, t.width, t.height) for t in tiles] == [(0, 0, 640, 480)]


def test_mask_to_box_uses_tight_polygon_extent():
    assert mask_to_box(np.array([[5.0, 7.0], [19.0, 8.0], [18.0, 31.0]])) == (5.0, 7.0, 19.0, 31.0)
    assert mask_to_box(np.empty((0, 2))) is None


def test_geometry_reasons_and_class_agnostic_nms_are_reported():
    candidates = [
        Candidate((10, 10, 110, 310), 0.90, "shampoo bottle", 0, True),
        Candidate((12, 12, 108, 306), 0.80, "conditioner bottle", 1, False),
        Candidate((0, 0, 8, 50), 0.70, "shampoo bottle", 0, False),
        Candidate((0, 0, 800, 600), 0.60, "boxed hair product", 0, False),
    ]
    valid, rejected = filter_candidates(candidates, 1600, 1200, localization_config())
    kept, duplicate_count = class_agnostic_nms(valid, 0.50)
    assert rejected == {"too_small": 1, "too_large": 1}
    assert kept == [candidates[0]]
    assert duplicate_count == 1
```

- [ ] **Step 2: Run the geometry tests and verify missing functions**

Run: `.venv/bin/pytest -q tests/test_localization.py`

Expected: collection fails because the localization functions are not defined.

- [ ] **Step 3: Implement edge-anchored tiling and geometry utilities**

Use `stride = max(1, round(tile_size * (1 - overlap)))`. For each axis, emit `0` for images no larger than a tile; otherwise emit regular stride positions and append `length - tile_size` if it is not already present. Form tiles in row-major order and clip tile width/height to the image.

`mask_to_box` must accept only finite `N x 2` points and return min/max extents when both dimensions are positive. `filter_candidates` must clip boxes to image bounds, then count rejection reasons using these exact keys: `non_finite`, `zero_area`, `too_small`, `too_large`, and `aspect_ratio`. `class_agnostic_nms` must sort by descending localization confidence with original input position as the tie-breaker, compute standard IoU, and suppress candidates whose IoU is strictly greater than the threshold regardless of generic prompt class.

```python
def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts
```

- [ ] **Step 4: Run geometry tests and regression tests**

Run: `.venv/bin/pytest -q tests/test_localization.py tests/test_yoloe_autolabel.py::ConversionAndOutputTests`

Expected: all selected tests pass.

- [ ] **Step 5: Commit deterministic localization geometry**

```bash
git add hair_annotation/localization.py tests/test_localization.py
git commit -m "feat: add tiled localization geometry"
```

### Task 3: YOLOE Generic Text Localizer

**Files:**
- Modify: `hair_annotation/localization.py`
- Modify: `tests/test_localization.py`

**Interfaces:**
- Consumes: geometry functions from Task 2 and a YOLOE-compatible object with `set_classes(list[str])` plus a keyword-only `predict` call using `source`, `imgsz`, `conf`, `iou`, `device`, and `verbose`.
- Produces: `load_text_localizer(model_name: str, prompts: Sequence[str]) -> Any`; `localize_image(model, image_bgr: np.ndarray, config: LocalizationConfig, imgsz: int, device: int | str) -> tuple[list[Candidate], dict[str, int]]`; `localize_reference_image(model, image_bgr: np.ndarray, config: LocalizationConfig, imgsz: int, device: int | str) -> tuple[list[Candidate], dict[str, int]]`.

- [ ] **Step 1: Add failing adapter tests with boxes and masks**

```python
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hair_annotation.localization import load_text_localizer, localize_image


def test_text_localizer_sets_generic_classes_once():
    model = mock.Mock()
    constructor = mock.Mock(return_value=model)
    prompts = ["shampoo bottle", "conditioner bottle"]
    with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLOE=constructor)}):
        actual = load_text_localizer("yoloe-26l-seg.pt", prompts)
    assert actual is model
    constructor.assert_called_once_with("yoloe-26l-seg.pt")
    model.set_classes.assert_called_once_with(prompts)


def test_localize_image_prefers_mask_extent_and_maps_tile_offsets():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 120.0, 330.0]]),
        conf=np.array([0.81]),
        cls=np.array([0.0]),
    )
    masks = SimpleNamespace(xy=[np.array([[25.0, 40.0], [100.0, 40.0], [100.0, 300.0], [25.0, 300.0]])])
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=masks)]
    image = np.zeros((900, 1300, 3), dtype=np.uint8)
    candidates, diagnostics = localize_image(model, image, localization_config(), imgsz=1280, device=0)
    assert candidates[0].xyxy == (25.0, 40.0, 100.0, 300.0)
    assert candidates[-1].xyxy[0] >= 276.0
    assert diagnostics["tiles"] == 2
    assert diagnostics["mask_boxes"] == 2
```

- [ ] **Step 2: Run the adapter tests and verify failure**

Run: `.venv/bin/pytest -q tests/test_localization.py -k 'text_localizer or localize_image'`

Expected: tests fail because the adapter functions are missing.

- [ ] **Step 3: Implement lazy YOLOE loading and per-tile inference**

`load_text_localizer` must import `YOLOE` lazily, construct the configured `*-seg.pt` checkpoint, call `set_classes` exactly once with the five generic prompts, and wrap import/download errors with the model name and the offline-checkpoint instruction.

`localize_image` must crop each `Tile` directly from the decoded BGR image, call `model.predict` once per tile, validate equal box/confidence/class lengths, use `result.masks.xy[index]` when it yields a valid box, fall back to `result.boxes.xyxy[index]`, add tile offsets, and validate prompt indices. After all tiles, call `filter_candidates` and `class_agnostic_nms`. Return diagnostics with `tiles`, `raw_candidates`, `mask_boxes`, `fallback_boxes`, `duplicates_removed`, and every geometry rejection key.

`localize_reference_image` must reuse the same result parser on exactly one full-image tile and run class-agnostic NMS, but it must not apply shelf-size, full-image-area, or aspect-ratio filters. It rejects only non-finite, clipped zero-area boxes so a product occupying most of a small reference remains eligible. Add a test where a mask covering 80 percent of the reference is retained here but rejected by `localize_image` under `max_area_ratio=0.10`.

```python
results = model.predict(
    source=tile_image,
    imgsz=imgsz,
    conf=config.conf,
    iou=config.iou,
    device=device,
    verbose=False,
)
```

Do not import `YOLOEVPSegPredictor`, construct visual prompt canvases, or send `refer_image`/`visual_prompts` in this path.

- [ ] **Step 4: Run all localization tests**

Run: `.venv/bin/pytest -q tests/test_localization.py`

Expected: all localization tests pass, including fallback behavior for `masks=None` and malformed-output errors.

- [ ] **Step 5: Commit the generic localizer**

```bash
git add hair_annotation/localization.py tests/test_localization.py
git commit -m "feat: add YOLOE text-prompt localizer"
```

### Task 4: Automatic Reference Views and OpenCLIP SKU Matching

**Files:**
- Create: `hair_annotation/matching.py`
- Create: `tests/test_matching.py`

**Interfaces:**
- Consumes: `Candidate` records, decoded BGR reference/shelf arrays, SKU mappings containing `class_id`, `barcode`, and `sku_name`, and an injectable embedding backend.
- Produces: `OpenClipBackend(model_name: str, pretrained: str, device: str | int)` with `encode_images(images_rgb: Sequence[np.ndarray]) -> np.ndarray` and `encode_texts(texts: Sequence[str]) -> np.ndarray`; `derive_reference_views(reference_images_bgr: Mapping[int, np.ndarray], reference_candidates: Mapping[int, Sequence[Candidate]], max_views: int) -> tuple[dict[int, list[np.ndarray]], list[dict[str, Any]]]`; `build_prototypes(backend: EmbeddingBackend, sku_rows: Sequence[Mapping[str, Any]], views_rgb: Mapping[int, Sequence[np.ndarray]], config: MatchingConfig) -> PrototypeBank`; `classify_candidates(backend: EmbeddingBackend, image_bgr: np.ndarray, candidates: Sequence[Candidate], bank: PrototypeBank, config: MatchingConfig) -> list[ClassifiedCandidate]`. `PrototypeBank` is an immutable record in `matching.py` containing ordered SKU identities plus `visual_embeddings` and `text_embeddings` NumPy matrices.

- [ ] **Step 1: Write failing reference and uncertainty tests**

```python
import numpy as np

from hair_annotation.config import MatchingConfig
from hair_annotation.matching import build_prototypes, classify_candidates, derive_reference_views
from hair_annotation.types import Candidate


class FakeBackend:
    def encode_images(self, images_rgb):
        return np.stack([image[0, 0, :2].astype(float) for image in images_rgb])

    def encode_texts(self, texts):
        vectors = {"a retail hair-care product package of Dove Blue 450 ml": [1.0, 0.0], "a retail hair-care product package of Sunsilk Pink 450 ml": [0.0, 1.0]}
        return np.array([vectors[text] for text in texts], dtype=float)


def matching_config(min_score=0.24, min_margin=0.02):
    return MatchingConfig("ViT-B-32", "laion2b_s34b_b79k", 0.80, 0.20, min_score, min_margin, 3, 89, "a retail hair-care product package of {english_sku_name}")


def test_reference_fallback_preserves_source_and_marks_low_quality():
    source = np.full((20, 10, 3), 7, dtype=np.uint8)
    before = source.copy()
    views, diagnostics = derive_reference_views({0: source}, {0: []}, max_views=3)
    assert np.array_equal(source, before)
    assert np.array_equal(views[0][0], source[:, :, ::-1])
    assert diagnostics == [{"class_id": 0, "derived_views": 0, "fallback_full_image": True}]


def test_matching_accepts_clear_winner_and_rejects_small_margin():
    sku_rows = [
        {"class_id": 0, "barcode": "111", "sku_name": "Dove Blue 450 ml"},
        {"class_id": 1, "barcode": "222", "sku_name": "Sunsilk Pink 450 ml"},
    ]
    views = {0: [np.array([[[1, 0, 0]]], dtype=np.uint8)], 1: [np.array([[[0, 1, 0]]], dtype=np.uint8)]}
    bank = build_prototypes(FakeBackend(), sku_rows, views, matching_config())
    candidates = [Candidate((0, 0, 10, 10), 0.8, "shampoo bottle", 0, True)]
    clear_image_bgr = np.array([[[0.0, 0.0, 1.0]]])
    accepted = classify_candidates(FakeBackend(), clear_image_bgr, candidates, bank, matching_config())[0]
    assert accepted.assigned_class_id == 0
    assert accepted.assigned_name == "Dove Blue 450 ml"
    assert accepted.accepted is True
    ambiguous_image_bgr = np.array([[[0.0, 1.0, 1.0]]])
    ambiguous = classify_candidates(FakeBackend(), ambiguous_image_bgr, candidates, bank, matching_config(min_margin=0.20))[0]
    assert ambiguous.assigned_class_id == 89
    assert ambiguous.assigned_name == "Needs Review"
    assert ambiguous.review_reason == "top_one_margin_below_threshold"
    assert len(ambiguous.rankings) == 2
```

- [ ] **Step 2: Run matching tests and verify failure**

Run: `.venv/bin/pytest -q tests/test_matching.py`

Expected: collection fails because `hair_annotation.matching` is absent.

- [ ] **Step 3: Implement reference derivation and normalized prototypes**

For each reference, sort raw generic candidates by this exact key: descending localization confidence, ascending normalized distance between box center and image center, then ascending box coordinates. Keep at most `max_reference_views` positive-area crops, convert OpenCV BGR to RGB, and never modify the source array. If none remains, use a copied full RGB image and write `fallback_full_image: true` in diagnostics.

`build_prototypes` must L2-normalize every backend row, average and renormalize each SKU's visual rows, render one text template per SKU, and retain class ID/barcode/English name in ascending class-ID order. Reject missing views, non-finite embeddings, zero-norm rows, inconsistent dimensions, duplicate class IDs, and class ID 89 in permanent SKUs.

- [ ] **Step 4: Implement OpenCLIP and conservative ranking**

```python
class OpenClipBackend:
    def __init__(self, model_name, pretrained, device):
        import open_clip
        import torch
        from PIL import Image

        self._torch = torch
        self._image_type = Image
        self._device = f"cuda:{device}" if isinstance(device, int) else str(device)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
```

Both encode methods must use `torch.inference_mode()`, batch inputs, return CPU `float64` NumPy arrays, and surface a message naming the missing model/weights when download fails. `classify_candidates` must clip each candidate box, crop it from `image_bgr`, convert BGR to RGB, then call the backend. It computes `0.80 * visual cosine + 0.20 * text cosine`, uses stable descending score/class-ID ordering, retains at most three `RankedSku` values, and applies reasons in this order: `score_below_threshold`, `top_one_margin_below_threshold`, `accepted`. An uncertain candidate gets class ID 89 and display name `Needs Review` without discarding its ranked suggestions.

When `device` is an integer, map it to `cuda:{device}` for OpenCLIP. When it is `cpu`, `mps`, or an explicit `cuda:N` string, pass it through unchanged. Report a clear error before inference if the requested CUDA device is unavailable.

- [ ] **Step 5: Run matching tests**

Run: `.venv/bin/pytest -q tests/test_matching.py`

Expected: all matching tests pass using `FakeBackend`; no model weights are downloaded.

- [ ] **Step 6: Commit SKU matching**

```bash
git add hair_annotation/matching.py tests/test_matching.py
git commit -m "feat: add conservative OpenCLIP SKU matching"
```

### Task 5: Integrate the Two Stages into the Existing CLI

**Files:**
- Modify: `yoloe_autolabel.py:28-62,629-821,869-1240`
- Modify: `tests/test_yoloe_autolabel.py`

**Interfaces:**
- Consumes: `BoxFirstConfig`, `load_text_localizer`, `localize_reference_image`, `localize_image`, `derive_reference_views`, `OpenClipBackend`, `build_prototypes`, and `classify_candidates`.
- Produces: `build_runtime_matcher(active_skus: Sequence[Sku], references: Sequence[ReferencePrompt], decoded_references: Mapping[Path, np.ndarray], localizer: Any, localization_config: LocalizationConfig, matching_config: MatchingConfig, imgsz: int, device: int | str) -> tuple[OpenClipBackend, PrototypeBank, list[dict[str, Any]]]`; `classify_runtime_candidates(backend: OpenClipBackend, image_bgr: np.ndarray, candidates: Sequence[Candidate], bank: PrototypeBank, matching_config: MatchingConfig) -> list[ClassifiedCandidate]`; the existing CLI exit-code contract; one numeric YOLO label per successfully processed shelf image; per-image JSON; English preview overlays; `review_queue.csv`; `reference_diagnostics.json`; expanded `run.json`; provenance fingerprints covering both model stages.

- [ ] **Step 1: Replace visual-prompt CLI mocks with failing box-first integration tests**

```python
def write_box_first_project(tmp_path, sku_count=2, image_count=1):
    root = tmp_path / "project"
    root.mkdir()
    config = valid_config(pilot={"enabled": False, "class_ids": [], "max_images": image_count, "max_skus": sku_count})
    config_path = write_project(root, config)
    rows = []
    references = []
    for class_id in range(sku_count):
        barcode = str(111 + class_id)
        name = "Dove Blue 450 ml" if class_id == 0 else "Sunsilk Pink 450 ml"
        rows.append({"class_id": str(class_id), "barcode": barcode, "brand": "Brand", "sku_name": name, "sku_name_th": f"Thai {class_id}", "enabled": "true"})
        reference_name = f"ref_{class_id}.jpg"
        (root / reference_name).write_bytes(f"reference-{class_id}".encode("ascii"))
        references.append({"class_id": class_id, "image": reference_name})
    ManifestTests().write_manifest(root, rows, columns=["class_id", "barcode", "brand", "sku_name", "sku_name_th", "enabled"])
    (root / "refs.yaml").write_text(json.dumps({"references": references}), encoding="utf-8")
    for index in range(image_count):
        (root / "images" / f"shelf_{index}.jpg").write_bytes(f"shelf-{index}".encode("ascii"))
    (root / "images" / "shelf.jpg").unlink(missing_ok=True)
    return root, config_path


def run_box_first_cli_with_fakes(config_path, classified):
    root = config_path.parent
    decoded = {path: np.ones((80, 60, 3), dtype=np.uint8) for path in root.glob("ref_*.jpg")}
    decoded.update({path: np.ones((100, 100, 3), dtype=np.uint8) for path in (root / "images").glob("*.jpg")})
    localized = [item.candidate for item in classified]
    with mock.patch.object(app, "_load_cv2", return_value=FakeCv2(decoded)), \
         mock.patch.object(app, "load_text_localizer", return_value=mock.Mock()), \
         mock.patch.object(app, "localize_image", return_value=(localized, {"tiles": 1})), \
         mock.patch.object(app, "build_runtime_matcher", return_value=(mock.Mock(), mock.Mock(), [])) as build_matcher, \
         mock.patch.object(app, "classify_runtime_candidates", return_value=classified):
        result = app.main(["--config", str(config_path), "--require-enabled-count", "2"])
    return result, build_matcher


def test_cli_runs_localization_then_matching_for_all_enabled_skus(tmp_path):
    root, config_path = write_box_first_project(tmp_path, sku_count=2, image_count=1)
    localized = [Candidate((10, 10, 30, 50), 0.8, "shampoo bottle", 0, True)]
    ranked = (RankedSku(1, "112", "Sunsilk Pink 450 ml", 0.55, 0.58, 0.43),)
    classified = [ClassifiedCandidate(localized[0], 1, "Sunsilk Pink 450 ml", True, "accepted", ranked)]
    result, build_matcher = run_box_first_cli_with_fakes(config_path, classified)
    assert result == 0
    assert [sku.class_id for sku in build_matcher.call_args.kwargs["active_skus"]] == [0, 1]
    label = (root / "output/raw_predictions/labels/shelf_0.txt").read_text(encoding="utf-8")
    assert label.startswith("1 ")


def test_uncertain_preview_and_review_queue_use_english_suggestion(tmp_path):
    root, config_path = write_box_first_project(tmp_path, sku_count=2, image_count=1)
    ranked = (RankedSku(0, "111", "Dove Blue 450 ml", 0.23, 0.24, 0.19),)
    candidate = Candidate((10, 10, 30, 50), 0.8, "shampoo bottle", 0, True)
    classified = [ClassifiedCandidate(candidate, 89, "Needs Review", False, "score_below_threshold", ranked)]
    result, _ = run_box_first_cli_with_fakes(config_path, classified)
    assert result == 0
    rows = list(csv.DictReader((root / "output/raw_predictions/review_queue.csv").open(encoding="utf-8")))
    assert rows[0]["assigned_class_id"] == "89"
    assert rows[0]["top1_name"] == "Dove Blue 450 ml"
    assert rows[0]["review_reason"] == "score_below_threshold"
```

Before these helpers, import `Candidate`, `ClassifiedCandidate`, and `RankedSku` from `hair_annotation.types`. Update the existing `valid_config()` helper with the complete Task 1 sections and update `sku()` plus manifest fixtures to include English `sku_name` and Thai `sku_name_th`.

- [ ] **Step 2: Run the focused CLI tests and verify the old visual path is still invoked**

Run: `.venv/bin/pytest -q tests/test_yoloe_autolabel.py -k 'localization_then_matching or uncertain_preview'`

Expected: both tests fail because the box-first integration helpers do not exist.

- [ ] **Step 3: Extend manifest and output records without changing permanent IDs**

Append `sku_name_th: str = ""` as the final `Sku` dataclass field so existing positional callers do not silently reorder `enabled`. Require a non-blank `sku_name_th` column when loading the generated runtime manifest, and include Thai name/barcode only in metadata and review CSV. Convert `ClassifiedCandidate` to numeric YOLO rows using `assigned_class_id`; a permanent class must exist in the active manifest, and the only non-manifest class allowed is 89.

Update preview text to use:

```python
if item.accepted:
    label = f"{item.assigned_name} {item.rankings[0].score:.2f}"
else:
    suggestion = item.rankings[0].sku_name if item.rankings else "no suggestion"
    label = f"Needs Review | {suggestion}"
```

Write `review_queue.csv` atomically with one row per box and these exact columns: `image`, `box_index`, `x1`, `y1`, `x2`, `y2`, `box_width`, `box_height`, `area_ratio`, `aspect_ratio`, `tile_index`, `localization_confidence`, `generic_prompt`, `mask_used`, `assigned_class_id`, `assigned_barcode`, `assigned_name`, `assigned_name_th`, `accepted`, `review_reason`, `top1_class_id`, `top1_barcode`, `top1_name`, `top1_score`, `top2_class_id`, `top2_barcode`, `top2_name`, `top2_score`, `top3_class_id`, `top3_barcode`, `top3_name`, `top3_score`. Leave unavailable assigned/top-rank cells empty rather than writing sentinel IDs.

- [ ] **Step 4: Replace the main inference loop with box-first orchestration**

Validation-only mode must validate all configuration, manifest/reference counts, file safety, output separation, class 89 availability, and checkpoint-path syntax without importing Ultralytics/OpenCLIP or creating output.

For a real run:

1. Decode and validate every active reference.
2. Load one text-prompted YOLOE model, call `localize_reference_image` once per reference, and derive reference views.
3. Load one OpenCLIP backend and build all 89 prototypes once.
4. For each shelf image, localize, crop/classify, write JSON/preview/YOLO label, and append review rows.
5. Preserve the existing per-image error policy and return exit code 1 when any shelf image fails.

Remove visual prompt canvas creation and prompt-batch execution from `main`. Pure legacy helpers may remain only when a focused regression test still calls them; delete unreferenced visual-only helpers and their tests in the same change. Update provenance to hash both model identifiers, local checkpoint bytes when present, generic prompts, geometry settings, matching settings, active SKU rows, reference bytes, dependency versions, and available model-weight fingerprints.

Write `reference_diagnostics.json` immediately after prototype construction. Each per-image JSON must include original dimensions, localization diagnostics, final box geometry, assigned numeric class, English display name, acceptance/review reason, and all available top-three class IDs, barcodes, English names, component similarities, and combined scores. For each processed shelf image, record `candidate_count`, geometry rejections, duplicates removed, assigned-permanent count, `Needs Review` count, and `image_status`, using `no_candidates_needs_review` when localization returns no candidates. Aggregate `run.json` with totals plus top-one/top-two score summaries containing `count`, `minimum`, `p25`, `median`, `p75`, and `maximum`; use `null` for every statistic when the score list is empty.

On a compatible resume, load existing `review_queue.csv` and per-image metadata before processing. Preserve rows for skipped images, remove old rows only for images selected for overwrite, and atomically rewrite the combined queue in deterministic image/box order. Never mark a failed image complete and never create an empty label as a substitute for an inference error.

- [ ] **Step 5: Run CLI, safety, and resume tests**

Run: `.venv/bin/pytest -q tests/test_yoloe_autolabel.py`

Expected: all CLI tests pass; fake inference verifies localization precedes matching, overwrite remains explicit, incompatible provenance refuses resume, and validation-only imports neither model.

- [ ] **Step 6: Commit CLI integration**

```bash
git add yoloe_autolabel.py tests/test_yoloe_autolabel.py
git commit -m "feat: run box-first SKU annotation pipeline"
```

### Task 6: Strict Ultralytics Platform and Review Archives

**Files:**
- Create: `hair_annotation/export.py`
- Create: `tests/test_platform_export.py`
- Modify: `colab_runtime.py:94-132`
- Modify: `tests/test_colab_assets.py:57-84`

**Interfaces:**
- Consumes: full-run shelf images, labels, SKU manifest, review CSV, metadata, previews, summaries, and provenance.
- Produces: `assign_splits(image_names: Sequence[str], train_fraction: float, seed: str) -> dict[str, str]`; `package_platform_dataset(project: Path, destination: Path, config: ExportConfig) -> Path`; `package_review_bundle(project: Path, destination: Path) -> Path`; `colab_runtime.package_results(project: Path, platform_destination: Path, review_destination: Path) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing exact-layout and split tests**

```python
import csv
import zipfile
from pathlib import Path

import pytest
import yaml

from hair_annotation.config import ExportConfig
from hair_annotation.export import assign_splits, package_platform_dataset


def export_config():
    return ExportConfig(train_fraction=0.90, split_seed="hair-osa-v1")


@pytest.fixture
def platform_project(tmp_path):
    project = tmp_path / "hair_colab"
    images = project / "shelf_images"
    labels = project / "output" / "raw_predictions" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(10):
        (images / f"shelf_{index}.jpg").write_bytes(f"original-{index}".encode("ascii"))
        (labels / f"shelf_{index}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    with (project / "sku_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "barcode", "brand", "sku_name", "sku_name_th", "enabled"])
        writer.writeheader()
        for class_id in range(89):
            writer.writerow({
                "class_id": class_id,
                "barcode": str(100000 + class_id),
                "brand": "Brand",
                "sku_name": "Dove Blue 450 ml" if class_id == 0 else f"English SKU {class_id}",
                "sku_name_th": f"Thai SKU {class_id}",
                "enabled": "true",
            })
    return project


def test_split_is_deterministic_disjoint_and_exact_for_ten_images():
    names = [f"shelf_{index}.jpg" for index in range(10)]
    first = assign_splits(names, 0.90, "hair-osa-v1")
    second = assign_splits(list(reversed(names)), 0.90, "hair-osa-v1")
    assert first == second
    assert list(first.values()).count("train") == 9
    assert list(first.values()).count("val") == 1


def test_platform_archive_contains_only_standard_yolo_members(platform_project, tmp_path):
    destination = tmp_path / "hair_ultralytics_platform.zip"
    package_platform_dataset(platform_project, destination, export_config())
    with zipfile.ZipFile(destination) as archive:
        names = sorted(archive.namelist())
        assert "data.yaml" in names
        assert "images/train/shelf_0.jpg" in names or "images/val/shelf_0.jpg" in names
        assert "labels/train/shelf_0.txt" in names or "labels/val/shelf_0.txt" in names
        assert not any(name.startswith("metadata/") for name in names)
        data = yaml.safe_load(archive.read("data.yaml"))
    assert data["path"] == "."
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"][0] == "Dove Blue 450 ml"
    assert data["names"][89] == "Needs Review"
```

Also test missing image/label pairs, class IDs outside 0-89, non-finite or non-normalized coordinates, duplicate case-insensitive stems, an existing destination, atomic cleanup after injected ZIP failure, and preservation of original image bytes.

- [ ] **Step 2: Run export tests and verify failure**

Run: `.venv/bin/pytest -q tests/test_platform_export.py tests/test_colab_assets.py -k 'package or split'`

Expected: collection fails because `hair_annotation.export` is absent.

- [ ] **Step 3: Implement deterministic exact-count splitting and `data.yaml`**

Sort unique image names by `sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()` with the filename as a tie-breaker. For two or more images, compute `val_count = min(len(names) - 1, max(1, round(len(names) * (1.0 - train_fraction))))`; assign the first `val_count` hashes to validation and the remainder to train. Reject a dataset with fewer than two successful images at packaging time.

Generate `data.yaml` with `path: .`, relative split paths, and an integer-keyed `names` mapping for 0-88 in manifest order plus `89: Needs Review`. Reject missing/gapped permanent IDs and any manifest row already using 89.

- [ ] **Step 4: Implement atomic archives and exhaustive preflight**

The Platform archive must contain only the five documented path families: `data.yaml`, `images/train/`, `images/val/`, `labels/train/`, and `labels/val/`. Validate every numeric YOLO row has five fields, class 0-89, and four finite values in `[0, 1]`. Require one label file per image, including an empty file for an image with zero candidates.

The review archive must contain `sku_manifest.csv`, `review_queue.csv`, `metadata/`, `previews/`, `summary.csv`, `run.json`, `provenance.json`, `reference_diagnostics.json`, and `README.txt`. The README must state that the Platform labels are machine suggestions requiring human review and that all `Needs Review` annotations must be reassigned before class 89 is deleted.

Both packagers must write to a temporary ZIP beside the destination, reopen it with `ZipFile.testzip()`, and call `os.replace` only after successful validation. `colab_runtime.package_results` must build both files and return their resolved paths.

- [ ] **Step 5: Run export and runtime-helper tests**

Run: `.venv/bin/pytest -q tests/test_platform_export.py tests/test_colab_assets.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit both result packagers**

```bash
git add hair_annotation/export.py tests/test_platform_export.py colab_runtime.py tests/test_colab_assets.py
git commit -m "feat: package Platform and review archives"
```

### Task 7: Runtime Configuration and Dependency Pins

**Files:**
- Modify: `colab/config.yaml`
- Modify: `requirements.txt`
- Modify: `tests/test_colab_assets.py`

**Interfaces:**
- Consumes: the exact keys validated by `BoxFirstConfig`.
- Produces: a full-run runtime config and an all-SKU/10-image pilot config accepted by `yoloe_autolabel.py`.

- [ ] **Step 1: Update asset tests to assert all approved defaults**

```python
def test_colab_config_has_approved_box_first_defaults():
    config = yoloe_autolabel.load_config(Path("colab/config.yaml"))
    assert config["yoloe"] == {"model": "yoloe-26l-seg.pt", "imgsz": 1280, "device": 0}
    assert config["localization"]["tile_size"] == 1024
    assert config["localization"]["overlap"] == 0.20
    assert config["matching"]["model"] == "ViT-B-32"
    assert config["matching"]["pretrained"] == "laion2b_s34b_b79k"
    assert config["matching"]["needs_review_class_id"] == 89
    assert config["export"] == {"train_fraction": 0.90, "split_seed": "hair-osa-v1"}


def test_pilot_limits_images_but_not_skus(tmp_path):
    project = tmp_path / "hair_colab"
    project.mkdir()
    shutil.copy2(Path("colab/config.yaml"), project / "config.yaml")
    pilot_path = colab_runtime.write_pilot_config(project, max_images=10)
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))
    assert pilot["pilot"] == {"enabled": True, "class_ids": [], "max_skus": 89, "max_images": 10}
    assert pilot["output"]["root"] == "pilot_output"
```

Add `import shutil` to `tests/test_colab_assets.py` with the existing standard-library imports.

- [ ] **Step 2: Run asset tests and verify old defaults fail**

Run: `.venv/bin/pytest -q tests/test_colab_assets.py -k 'defaults or pilot'`

Expected: tests fail because prompt-canvas settings and five-SKU pilot settings are still present.

- [ ] **Step 3: Replace the runtime YAML controls**

Keep `project`, `dataset`, `output`, and full-run `pilot.enabled: false`. Reduce `yoloe` to model, inference size, and device. Add every localization/matching/export value shown in Task 1. Keep `max_skus: 89` for backward-compatible pilot validation, but the box-first CLI must use all enabled SKUs for prototype construction regardless of this value.

Pin dependencies as:

```text
ultralytics==8.4.149
open-clip-torch==3.3.0
numpy>=2.0,<3
pandas>=2.2,<3
openpyxl>=3.1,<4
PyYAML>=6.0,<7
opencv-python>=4.10,<5
Pillow>=10,<13
pytest>=8,<9
```

- [ ] **Step 4: Run config and validation-only tests**

Run: `.venv/bin/pytest -q tests/test_box_first_config.py tests/test_colab_assets.py tests/test_yoloe_autolabel.py -k 'config or validate_only or enabled_count'`

Expected: all selected tests pass without downloading YOLOE or OpenCLIP weights.

- [ ] **Step 5: Commit runtime defaults**

```bash
git add colab/config.yaml requirements.txt tests/test_colab_assets.py colab_runtime.py
git commit -m "chore: configure box-first Colab runtime"
```

### Task 8: Runtime Package Builder Includes the New Implementation

**Files:**
- Modify: `prepare_hair_colab.py:333-394,457-504`
- Modify: `tests/test_prepare_hair_colab.py:242-284,327-367`

**Interfaces:**
- Consumes: all source modules/tests/config/notebook required by an offline extracted runtime.
- Produces: `dist/hair_colab_runtime.zip` with ASCII-safe paths, original 89 references and 632 shelf images, the new annotation modules/tests, and unchanged image bytes.

- [ ] **Step 1: Add failing archive-member assertions**

```python
def test_runtime_archive_contains_box_first_modules_and_tests(package_fixture, tmp_path):
    archive_path = tmp_path / "hair_colab_runtime.zip"
    prep.build_runtime_package(package_fixture, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "hair_colab/hair_annotation/__init__.py" in names
    assert "hair_colab/hair_annotation/config.py" in names
    assert "hair_colab/hair_annotation/localization.py" in names
    assert "hair_colab/hair_annotation/matching.py" in names
    assert "hair_colab/hair_annotation/export.py" in names
    assert "hair_colab/tests/test_box_first_config.py" in names
    assert "hair_colab/tests/test_localization.py" in names
    assert "hair_colab/tests/test_matching.py" in names
    assert "hair_colab/tests/test_platform_export.py" in names
```

- [ ] **Step 2: Run the package test and verify missing members**

Run: `.venv/bin/pytest -q tests/test_prepare_hair_colab.py -k 'box_first_modules'`

Expected: the test fails because `_required_repo_files` only includes the legacy script and one test.

- [ ] **Step 3: Expand the explicit package allowlist**

Keep `_required_repo_files` as an explicit source-to-archive mapping rather than copying the repository recursively. Add every `hair_annotation/*.py` file and the four new model-free test files alongside the already packaged `tests/test_yoloe_autolabel.py`. Continue excluding `.git`, `.DS_Store`, source workbook, source Thai filenames, local environments, cached bytecode, and generated documentation.

Update package tests so extracted `pytest -q` uses fake backends only. Assert every archive path is ASCII, there are exactly 89 `references/` images and 632 `shelf_images/` images, and the copied bytes match the source checksum.

- [ ] **Step 4: Run all builder tests**

Run: `.venv/bin/pytest -q tests/test_prepare_hair_colab.py`

Expected: all builder tests pass, including failure atomicity and exact registry metadata export.

- [ ] **Step 5: Commit the package allowlist**

```bash
git add prepare_hair_colab.py tests/test_prepare_hair_colab.py
git commit -m "build: include box-first modules in runtime ZIP"
```

### Task 9: Colab Notebook and Human Review Documentation

**Files:**
- Modify: `colab/hair_colab_enterprise.ipynb`
- Modify: `colab/README.md`
- Modify: `README.md`
- Modify: `tests/test_colab_assets.py`

**Interfaces:**
- Consumes: the unchanged CLI command and new two-archive `colab_runtime.package_results` call.
- Produces: a linear Colab Enterprise workflow that validates 89 SKUs/632 images, runs a 10-image pilot, shows review previews/diagnostics, runs all images, and downloads both ZIPs.

- [ ] **Step 1: Strengthen notebook-content tests before editing the notebook**

```python
def test_notebook_uses_runtime_only_box_first_workflow():
    notebook = json.loads(Path("colab/hair_colab_enterprise.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "drive.mount" not in source
    assert "google.cloud.storage" not in source
    assert "max_images=10" in source
    assert '"--require-enabled-count", "89"' in source
    assert "hair_ultralytics_platform.zip" in source
    assert "hair_annotation_review.zip" in source
    assert "package_results" in source
    assert "Needs Review" in source
```

- [ ] **Step 2: Run notebook tests and verify missing workflow text**

Run: `.venv/bin/pytest -q tests/test_colab_assets.py -k 'notebook'`

Expected: the content test fails because the notebook still packages only `hair_label_results.zip`.

- [ ] **Step 3: Update notebook cells in their existing order**

Keep the upload and extraction cells. The dependency cell runs `pip install -r requirements.txt`; the offline test cell runs `pytest -q`. The validation cell remains:

```python
subprocess.run(
    [sys.executable, "yoloe_autolabel.py", "--config", "config.yaml", "--validate-only", "--require-enabled-count", "89"],
    cwd=project,
    check=True,
)
```

Create the pilot with `write_pilot_config(project, max_images=10)`, run it against all 89 SKUs, print `pilot_output/raw_predictions/run.json`, show a grid of generated previews, and explain that the user should inspect box tightness plus `Needs Review` rate before continuing. Keep full-run execution in a separately marked cell.

The packaging cell uses:

```python
platform_zip, review_zip = colab_runtime.package_results(
    project,
    Path("/content/hair_ultralytics_platform.zip"),
    Path("/content/hair_annotation_review.zip"),
)
print(platform_zip)
print(review_zip)
```

Add a final Markdown cell stating that `hair_ultralytics_platform.zip` goes to Ultralytics Platform, `hair_annotation_review.zip` stays beside the reviewer, every box requires review, and class 89 must be empty before it is deleted.

- [ ] **Step 4: Update both READMEs with the exact operator sequence**

Document runtime ephemerality, first-use downloads (`yoloe-26l-seg.pt`, YOLOE text encoder, and OpenCLIP weights), manual checkpoint upload fallback, all-89-SKU pilot behavior, the two ZIP purposes, the numeric-label/English-name relationship, and the six-step Platform review order from the approved spec. Remove instructions referring to prompt canvases or `hair_label_results.zip`.

- [ ] **Step 5: Run notebook and documentation checks**

Run: `.venv/bin/pytest -q tests/test_colab_assets.py`

Run: `rg -n "drive\.mount|google\.cloud\.storage|hair_label_results\.zip|prompt canvas|5-SKU" colab README.md`

Expected: tests pass and `rg` returns no matches.

- [ ] **Step 6: Commit the operator workflow**

```bash
git add colab/hair_colab_enterprise.ipynb colab/README.md README.md tests/test_colab_assets.py
git commit -m "docs: update Colab Platform review workflow"
```

### Task 10: Full Verification and Runtime Artifact Rebuild

**Files:**
- Verify: all tracked Python, YAML, notebook, tests, and documentation from Tasks 1-9.
- Generate: `dist/hair_colab_runtime.zip` and `colab/generated/sku_manifest.csv` using the existing source inputs; generated files remain ignored unless repository policy changes.

**Interfaces:**
- Consumes: the completed implementation and the user-provided workbook/product/shelf directories.
- Produces: a locally verified runtime ZIP ready to upload to Colab Enterprise.

- [ ] **Step 1: Run syntax, whitespace, and complete model-free tests**

Run: `git diff --check`

Run: `.venv/bin/python -m compileall -q hair_annotation yoloe_autolabel.py colab_runtime.py prepare_hair_colab.py`

Run: `.venv/bin/pytest -q`

Expected: no diff errors, compilation exits 0, and all tests pass without downloading model weights.

- [ ] **Step 2: Rebuild the runtime ZIP from the exact user inputs**

Run:

```bash
.venv/bin/python prepare_hair_colab.py \
  --workbook "/Users/me/Downloads/Cloudviu-data/HAIR SKU list (with Account list).xlsx" \
  --product-images "/Users/me/Downloads/Cloudviu-data/HAIR_product_images" \
  --shelf-images "/Users/me/Downloads/Cloudviu-data/HAIR_shelf_images" \
  --overrides colab/reference_name_overrides.yaml \
  --translations colab/sku_name_translations.yaml \
  --output dist/hair_colab_runtime.zip \
  --metadata-output colab/generated \
  --overwrite
```

Expected: success reports exactly 89 SKUs, 89 references, and 632 shelf images.

- [ ] **Step 3: Extract to a temporary directory and validate the packaged runtime**

Run:

```bash
runtime_check=$(mktemp -d /private/tmp/hair-colab-check.XXXXXX)
unzip -q dist/hair_colab_runtime.zip -d "$runtime_check"
.venv/bin/python "$runtime_check/hair_colab/yoloe_autolabel.py" \
  --config "$runtime_check/hair_colab/config.yaml" \
  --validate-only \
  --require-enabled-count 89
```

Expected: `Validation successful: 89 SKU(s), 632 image(s)` and no model download.

- [ ] **Step 4: Inspect archive boundaries and class names**

Run:

```bash
unzip -Z1 dist/hair_colab_runtime.zip | LC_ALL=C sort > /private/tmp/hair-colab-members.txt
rg -n "(^|/)\.DS_Store$|reviewed_labels|HAIR SKU list" /private/tmp/hair-colab-members.txt
```

Expected: `rg` returns no matches. Confirm `hair_annotation/`, four offline test files, `sku_manifest.csv`, `reference_prompts.yaml`, 89 references, and 632 shelf images are present.

- [ ] **Step 5: Review the final tracked diff and status**

Run: `git status --short`

Run: `git log --oneline -10`

Expected: only the pre-existing untracked `.DS_Store` may remain; generated runtime artifacts are ignored; every implementation task has its own commit on `main`.

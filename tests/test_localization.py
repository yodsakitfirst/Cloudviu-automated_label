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
        prompts=(
            "shampoo bottle",
            "conditioner bottle",
            "hair treatment pouch",
            "boxed hair product",
            "hair care multipack",
        ),
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
    assert mask_to_box(np.array([[5.0, 7.0], [19.0, 8.0], [18.0, 31.0]])) == (
        5.0,
        7.0,
        19.0,
        31.0,
    )
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


def test_filter_clips_boxes_before_geometry_checks():
    candidates = [Candidate((-10, -5, 20, 25), 0.5, "bottle", 0, False)]
    valid, rejected = filter_candidates(candidates, 100, 100, localization_config())
    assert valid == [Candidate((0.0, 0.0, 20.0, 25.0), 0.5, "bottle", 0, False)]
    assert rejected == {}


def test_filter_reports_non_finite_zero_area_and_aspect_ratio():
    candidates = [
        Candidate((np.nan, 0, 10, 10), 0.5, "bottle", 0, False),
        Candidate((5, 5, 5, 10), 0.5, "bottle", 0, False),
        Candidate((0, 0, 12, 100), 0.5, "bottle", 0, False),
    ]
    valid, rejected = filter_candidates(candidates, 200, 200, localization_config())
    assert valid == []
    assert rejected == {"non_finite": 1, "zero_area": 1, "aspect_ratio": 1}


def test_nms_uses_input_order_for_equal_confidence_and_strict_threshold():
    first = Candidate((0, 0, 10, 10), 0.5, "bottle", 0, False)
    second = Candidate((5, 0, 15, 10), 0.5, "pouch", 1, False)
    tied = [second, first]
    kept, duplicate_count = class_agnostic_nms(tied, iou_threshold=1 / 3)
    assert kept == tied
    assert duplicate_count == 0

    kept, duplicate_count = class_agnostic_nms(tied, iou_threshold=0.2)
    assert kept == [second]
    assert duplicate_count == 1

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


def test_recall_retry_config_is_parsed():
    data = valid_sections()
    data["localization"]["recall_retry"] = {
        "enabled": True,
        "min_candidates_per_megapixel": 10.0,
        "tile_size": 640,
        "overlap": 0.30,
        "conf": 0.02,
        "iou": 0.60,
        "min_side": 6,
        "max_area_ratio": 0.20,
        "min_aspect_ratio": 0.08,
        "max_aspect_ratio": 8.0,
        "nms_iou": 0.70,
    }

    config = BoxFirstConfig.from_mapping(data)

    assert config.localization.recall_retry is not None
    assert config.localization.recall_retry.enabled is True
    assert config.localization.recall_retry.tile_size == 640
    assert config.localization.recall_retry.conf == 0.02


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", "yes"),
        ("min_candidates_per_megapixel", 0),
        ("tile_size", 0),
        ("overlap", 1.0),
        ("conf", -0.01),
        ("min_side", 0),
        ("max_area_ratio", 1.1),
        ("min_aspect_ratio", 0),
        ("max_aspect_ratio", 0.01),
        ("nms_iou", 1.1),
    ],
)
def test_invalid_recall_retry_config_is_rejected(key, value):
    data = valid_sections()
    retry = {
        "enabled": True,
        "min_candidates_per_megapixel": 10.0,
        "tile_size": 640,
        "overlap": 0.30,
        "conf": 0.02,
        "iou": 0.60,
        "min_side": 6,
        "max_area_ratio": 0.20,
        "min_aspect_ratio": 0.08,
        "max_aspect_ratio": 8.0,
        "nms_iou": 0.70,
    }
    retry[key] = value
    data["localization"]["recall_retry"] = retry

    with pytest.raises(ValueError, match=f"localization.recall_retry.{key}"):
        BoxFirstConfig.from_mapping(data)


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

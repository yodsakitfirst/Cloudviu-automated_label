from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def _required(section: Mapping[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required config value {section_name}.{key}")
    return section[key]


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _closed_unit_interval(value: Any, label: str) -> float:
    parsed = _finite_number(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return parsed


def _open_unit_interval(value: Any, label: str) -> float:
    parsed = _finite_number(value, label)
    if not 0.0 < parsed < 1.0:
        raise ValueError(f"{label} must be greater than 0 and less than 1")
    return parsed


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class LocalizationConfig:
    tile_size: int
    overlap: float
    prompts: tuple[str, ...]
    conf: float
    iou: float
    min_side: int
    max_area_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    nms_iou: float


@dataclass(frozen=True)
class MatchingConfig:
    model: str
    pretrained: str
    visual_weight: float
    text_weight: float
    min_score: float
    min_margin: float
    max_reference_views: int
    needs_review_class_id: int
    text_template: str


@dataclass(frozen=True)
class ExportConfig:
    train_fraction: float
    split_seed: str


@dataclass(frozen=True)
class BoxFirstConfig:
    localization: LocalizationConfig
    matching: MatchingConfig
    export: ExportConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BoxFirstConfig:
        if not isinstance(data, Mapping):
            raise ValueError("Config must be a mapping")

        localization_data = _section(data, "localization")
        matching_data = _section(data, "matching")
        export_data = _section(data, "export")

        prompt_value = _required(localization_data, "localization", "prompts")
        if (
            not isinstance(prompt_value, Sequence)
            or isinstance(prompt_value, (str, bytes))
            or not prompt_value
        ):
            raise ValueError("localization.prompts must be a non-empty sequence of strings")
        prompts: list[str] = []
        seen_prompts: set[str] = set()
        for prompt in prompt_value:
            normalized = _non_empty_string(prompt, "localization.prompts")
            if normalized in seen_prompts:
                raise ValueError("localization.prompts must contain unique strings")
            seen_prompts.add(normalized)
            prompts.append(normalized)

        overlap = _finite_number(
            _required(localization_data, "localization", "overlap"),
            "localization.overlap",
        )
        if not 0.0 <= overlap < 1.0:
            raise ValueError("localization.overlap must be at least 0 and less than 1")

        max_area_ratio = _finite_number(
            _required(localization_data, "localization", "max_area_ratio"),
            "localization.max_area_ratio",
        )
        if not 0.0 < max_area_ratio <= 1.0:
            raise ValueError("localization.max_area_ratio must be greater than 0 and at most 1")

        min_aspect_ratio = _finite_number(
            _required(localization_data, "localization", "min_aspect_ratio"),
            "localization.min_aspect_ratio",
        )
        max_aspect_ratio = _finite_number(
            _required(localization_data, "localization", "max_aspect_ratio"),
            "localization.max_aspect_ratio",
        )
        if min_aspect_ratio <= 0.0:
            raise ValueError("localization.min_aspect_ratio must be greater than 0")
        if max_aspect_ratio < min_aspect_ratio:
            raise ValueError(
                "localization.max_aspect_ratio must be at least localization.min_aspect_ratio"
            )

        localization = LocalizationConfig(
            tile_size=_positive_integer(
                _required(localization_data, "localization", "tile_size"),
                "localization.tile_size",
            ),
            overlap=overlap,
            prompts=tuple(prompts),
            conf=_closed_unit_interval(
                _required(localization_data, "localization", "conf"),
                "localization.conf",
            ),
            iou=_closed_unit_interval(
                _required(localization_data, "localization", "iou"),
                "localization.iou",
            ),
            min_side=_positive_integer(
                _required(localization_data, "localization", "min_side"),
                "localization.min_side",
            ),
            max_area_ratio=max_area_ratio,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            nms_iou=_closed_unit_interval(
                _required(localization_data, "localization", "nms_iou"),
                "localization.nms_iou",
            ),
        )

        visual_weight = _closed_unit_interval(
            _required(matching_data, "matching", "visual_weight"),
            "matching.visual_weight",
        )
        text_weight = _closed_unit_interval(
            _required(matching_data, "matching", "text_weight"),
            "matching.text_weight",
        )
        if not math.isclose(visual_weight + text_weight, 1.0, abs_tol=1e-9):
            raise ValueError("matching.visual_weight and matching.text_weight must sum to 1")

        needs_review_class_id = _required(
            matching_data, "matching", "needs_review_class_id"
        )
        if type(needs_review_class_id) is not int or needs_review_class_id != 89:
            raise ValueError("matching.needs_review_class_id must be 89")

        text_template = _non_empty_string(
            _required(matching_data, "matching", "text_template"),
            "matching.text_template",
        )
        if "{english_sku_name}" not in text_template:
            raise ValueError("matching.text_template must contain {english_sku_name}")

        matching = MatchingConfig(
            model=_non_empty_string(
                _required(matching_data, "matching", "model"), "matching.model"
            ),
            pretrained=_non_empty_string(
                _required(matching_data, "matching", "pretrained"),
                "matching.pretrained",
            ),
            visual_weight=visual_weight,
            text_weight=text_weight,
            min_score=_closed_unit_interval(
                _required(matching_data, "matching", "min_score"),
                "matching.min_score",
            ),
            min_margin=_closed_unit_interval(
                _required(matching_data, "matching", "min_margin"),
                "matching.min_margin",
            ),
            max_reference_views=_positive_integer(
                _required(matching_data, "matching", "max_reference_views"),
                "matching.max_reference_views",
            ),
            needs_review_class_id=needs_review_class_id,
            text_template=text_template,
        )

        export = ExportConfig(
            train_fraction=_open_unit_interval(
                _required(export_data, "export", "train_fraction"),
                "export.train_fraction",
            ),
            split_seed=_non_empty_string(
                _required(export_data, "export", "split_seed"), "export.split_seed"
            ),
        )
        return cls(localization=localization, matching=matching, export=export)

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from hair_annotation.config import LocalizationConfig
from hair_annotation.localization import (
    class_agnostic_nms,
    filter_candidates,
    generate_tiles,
    load_text_localizer,
    localize_image,
    localize_reference_image,
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


def test_text_localizer_sets_generic_classes_once():
    model = mock.Mock()
    constructor = mock.Mock(return_value=model)
    prompts = ["shampoo bottle", "conditioner bottle"]
    with mock.patch.dict(
        "sys.modules", {"ultralytics": SimpleNamespace(YOLOE=constructor)}
    ):
        actual = load_text_localizer("yoloe-26l-seg.pt", prompts)
    assert actual is model
    constructor.assert_called_once_with("yoloe-26l-seg.pt")
    model.set_classes.assert_called_once_with(prompts)


def test_text_localizer_load_error_names_model_and_offline_checkpoint_action():
    constructor = mock.Mock(side_effect=OSError("download unavailable"))
    with mock.patch.dict(
        "sys.modules", {"ultralytics": SimpleNamespace(YOLOE=constructor)}
    ):
        with pytest.raises(RuntimeError) as error:
            load_text_localizer("missing-seg.pt", ["bottle"])
    message = str(error.value)
    assert "missing-seg.pt" in message
    assert "offline checkpoint" in message.lower()


def test_localize_image_prefers_mask_extent_and_maps_tile_offsets():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 120.0, 330.0]]),
        conf=np.array([0.81]),
        cls=np.array([0.0]),
    )
    masks = SimpleNamespace(
        xy=[
            np.array(
                [[25.0, 40.0], [100.0, 40.0], [100.0, 300.0], [25.0, 300.0]]
            )
        ]
    )
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=masks)]
    image = np.zeros((900, 1300, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(1300) % 251

    candidates, diagnostics = localize_image(
        model, image, localization_config(), imgsz=1280, device=0
    )

    assert candidates[0].xyxy == (25.0, 40.0, 100.0, 300.0)
    assert candidates[-1].xyxy == (301.0, 40.0, 376.0, 300.0)
    assert diagnostics == {
        "tiles": 2,
        "raw_candidates": 2,
        "mask_boxes": 2,
        "fallback_boxes": 0,
        "duplicates_removed": 0,
        "non_finite": 0,
        "zero_area": 0,
        "too_small": 0,
        "too_large": 0,
        "aspect_ratio": 0,
    }
    assert model.predict.call_count == 2
    first_call, second_call = model.predict.call_args_list
    assert first_call.kwargs["source"].shape == (900, 1024, 3)
    assert second_call.kwargs["source"].shape == (900, 1024, 3)
    assert np.shares_memory(first_call.kwargs["source"], image)
    assert np.shares_memory(second_call.kwargs["source"], image)
    np.testing.assert_array_equal(first_call.kwargs["source"], image[:, :1024])
    np.testing.assert_array_equal(second_call.kwargs["source"], image[:, 276:])
    for call in (first_call, second_call):
        assert set(call.kwargs) == {
            "source",
            "imgsz",
            "conf",
            "iou",
            "device",
            "verbose",
        }
        assert call.kwargs["imgsz"] == 1280
        assert call.kwargs["conf"] == 0.10
        assert call.kwargs["iou"] == 0.50
        assert call.kwargs["device"] == 0
        assert call.kwargs["verbose"] is False


def test_localize_image_falls_back_to_box_when_masks_are_unavailable():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 100.0, 230.0]]),
        conf=np.array([0.81]),
        cls=np.array([1.0]),
    )
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=None)]

    candidates, diagnostics = localize_image(
        model,
        np.zeros((500, 500, 3), dtype=np.uint8),
        localization_config(),
        imgsz=640,
        device="cpu",
    )

    assert candidates == [
        Candidate((20.0, 30.0, 100.0, 230.0), 0.81, "conditioner bottle", 0, False)
    ]
    assert diagnostics["mask_boxes"] == 0
    assert diagnostics["fallback_boxes"] == 1


def test_localize_image_falls_back_when_mask_polygon_cannot_form_a_box():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 100.0, 230.0]]),
        conf=np.array([0.81]),
        cls=np.array([0.0]),
    )
    masks = SimpleNamespace(xy=[[[20.0, 30.0], [100.0]]])
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=masks)]

    candidates, diagnostics = localize_image(
        model,
        np.zeros((500, 500, 3), dtype=np.uint8),
        localization_config(),
        imgsz=640,
        device="cpu",
    )

    assert candidates[0].xyxy == (20.0, 30.0, 100.0, 230.0)
    assert candidates[0].mask_used is False
    assert diagnostics["fallback_boxes"] == 1


def test_localize_image_rejects_unequal_output_lengths():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 120.0, 330.0]]),
        conf=np.array([]),
        cls=np.array([0.0]),
    )
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=None)]

    with pytest.raises(RuntimeError, match="unequal lengths"):
        localize_image(
            model,
            np.zeros((500, 500, 3), dtype=np.uint8),
            localization_config(),
            imgsz=640,
            device="cpu",
        )


def test_localize_image_rejects_invalid_prompt_index():
    boxes = SimpleNamespace(
        xyxy=np.array([[20.0, 30.0, 120.0, 330.0]]),
        conf=np.array([0.81]),
        cls=np.array([5.0]),
    )
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=None)]

    with pytest.raises(RuntimeError, match="unknown prompt index 5"):
        localize_image(
            model,
            np.zeros((500, 500, 3), dtype=np.uint8),
            localization_config(),
            imgsz=640,
            device="cpu",
        )


def test_reference_localization_retains_large_product_rejected_on_shelf():
    boxes = SimpleNamespace(
        xyxy=np.array([[0.0, 0.0, 80.0, 100.0]]),
        conf=np.array([0.9]),
        cls=np.array([0.0]),
    )
    masks = SimpleNamespace(
        xy=[np.array([[0.0, 0.0], [80.0, 0.0], [80.0, 100.0], [0.0, 100.0]])]
    )
    model = mock.Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, masks=masks)]
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    shelf_candidates, shelf_diagnostics = localize_image(
        model, image, localization_config(), imgsz=640, device="cpu"
    )
    reference_candidates, reference_diagnostics = localize_reference_image(
        model, image, localization_config(), imgsz=640, device="cpu"
    )

    assert shelf_candidates == []
    assert shelf_diagnostics["too_large"] == 1
    assert reference_candidates == [
        Candidate((0.0, 0.0, 80.0, 100.0), 0.9, "shampoo bottle", 0, True)
    ]
    assert reference_diagnostics["too_large"] == 0

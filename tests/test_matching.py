import sys
import types

import numpy as np
import pytest

from hair_annotation.config import MatchingConfig
from hair_annotation.matching import (
    OpenClipBackend,
    PrototypeBank,
    build_prototypes,
    classify_candidates,
    derive_reference_views,
)
from hair_annotation.types import Candidate


class FakeBackend:
    def encode_images(self, images_rgb):
        return np.stack([image[0, 0, :2].astype(float) for image in images_rgb])

    def encode_texts(self, texts):
        vectors = {
            "a retail hair-care product package of Dove Blue 450 ml": [1.0, 0.0],
            "a retail hair-care product package of Sunsilk Pink 450 ml": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts], dtype=float)


def matching_config(min_score=0.24, min_margin=0.02):
    return MatchingConfig(
        "ViT-B-32",
        "laion2b_s34b_b79k",
        0.80,
        0.20,
        min_score,
        min_margin,
        3,
        89,
        "a retail hair-care product package of {english_sku_name}",
    )


def test_reference_fallback_preserves_source_and_marks_low_quality():
    source = np.full((20, 10, 3), 7, dtype=np.uint8)
    before = source.copy()
    views, diagnostics = derive_reference_views({0: source}, {0: []}, max_views=3)
    assert np.array_equal(source, before)
    assert np.array_equal(views[0][0], source[:, :, ::-1])
    assert diagnostics == [
        {"class_id": 0, "derived_views": 0, "fallback_full_image": True}
    ]


def test_reference_views_prioritize_confidence_then_center_then_coordinates():
    source = np.zeros((10, 10, 3), dtype=np.uint8)
    source[4:6, 4:6] = [10, 20, 30]
    source[0:2, 0:2] = [40, 50, 60]
    source[8:10, 8:10] = [70, 80, 90]
    candidates = [
        Candidate((8, 8, 10, 10), 0.8, "bottle", 0, False),
        Candidate((0, 0, 2, 2), 0.9, "bottle", 0, False),
        Candidate((4, 4, 6, 6), 0.9, "bottle", 0, False),
    ]

    views, diagnostics = derive_reference_views(
        {4: source}, {4: candidates}, max_views=2
    )

    assert np.array_equal(views[4][0], np.full((2, 2, 3), [30, 20, 10]))
    assert np.array_equal(views[4][1], np.full((2, 2, 3), [60, 50, 40]))
    assert not np.shares_memory(views[4][0], source)
    assert diagnostics == [
        {"class_id": 4, "derived_views": 2, "fallback_full_image": False}
    ]


def test_reference_views_skip_clipped_zero_area_boxes():
    source = np.full((4, 4, 3), 9, dtype=np.uint8)
    candidates = [
        Candidate((-5, 0, -1, 3), 0.99, "bottle", 0, False),
        Candidate((1, 1, 3, 4), 0.8, "bottle", 0, False),
    ]

    views, diagnostics = derive_reference_views(
        {2: source}, {2: candidates}, max_views=3
    )

    assert views[2][0].shape == (3, 2, 3)
    assert diagnostics[0]["derived_views"] == 1
    assert diagnostics[0]["fallback_full_image"] is False


def test_reference_center_distance_is_normalized_by_image_diagonal():
    source = np.zeros((10, 100, 3), dtype=np.uint8)
    source[8:10, 49:51] = [10, 20, 30]
    source[4:6, 69:71] = [40, 50, 60]
    candidates = [
        Candidate((69, 4, 71, 6), 0.9, "bottle", 0, False),
        Candidate((49, 8, 51, 10), 0.9, "bottle", 0, False),
    ]

    views, _ = derive_reference_views({3: source}, {3: candidates}, max_views=1)

    assert np.array_equal(views[3][0], np.full((2, 2, 3), [30, 20, 10]))


def test_matching_accepts_clear_winner_and_rejects_small_margin():
    sku_rows = [
        {"class_id": 0, "barcode": "111", "sku_name": "Dove Blue 450 ml"},
        {"class_id": 1, "barcode": "222", "sku_name": "Sunsilk Pink 450 ml"},
    ]
    views = {
        0: [np.array([[[1, 0, 0]]], dtype=np.uint8)],
        1: [np.array([[[0, 1, 0]]], dtype=np.uint8)],
    }
    bank = build_prototypes(FakeBackend(), sku_rows, views, matching_config())
    candidates = [Candidate((0, 0, 10, 10), 0.8, "shampoo bottle", 0, True)]
    clear_image_bgr = np.array([[[0.0, 0.0, 1.0]]])
    accepted = classify_candidates(
        FakeBackend(), clear_image_bgr, candidates, bank, matching_config()
    )[0]
    assert accepted.assigned_class_id == 0
    assert accepted.assigned_name == "Dove Blue 450 ml"
    assert accepted.accepted is True
    ambiguous_image_bgr = np.array([[[0.0, 1.0, 1.0]]])
    ambiguous = classify_candidates(
        FakeBackend(),
        ambiguous_image_bgr,
        candidates,
        bank,
        matching_config(min_margin=0.20),
    )[0]
    assert ambiguous.assigned_class_id == 89
    assert ambiguous.assigned_name == "Needs Review"
    assert ambiguous.review_reason == "top_one_margin_below_threshold"
    assert len(ambiguous.rankings) == 2


def test_build_prototypes_normalizes_each_view_before_averaging():
    rows = [{"class_id": 7, "barcode": "777", "sku_name": "Dove Blue 450 ml"}]
    views = {
        7: [
            np.array([[[3.0, 0.0, 0.0]]]),
            np.array([[[0.0, 4.0, 0.0]]]),
        ]
    }

    bank = build_prototypes(FakeBackend(), rows, views, matching_config())

    expected = np.array([[2**-0.5, 2**-0.5]])
    assert bank.class_ids == (7,)
    assert bank.barcodes == ("777",)
    assert bank.sku_names == ("Dove Blue 450 ml",)
    assert np.allclose(bank.visual_embeddings, expected)
    assert np.array_equal(bank.text_embeddings, np.array([[1.0, 0.0]]))
    assert bank.visual_embeddings.flags.writeable is False
    assert bank.text_embeddings.flags.writeable is False


@pytest.mark.parametrize(
    ("rows", "views", "message"),
    [
        (
            [{"class_id": 1, "barcode": "1", "sku_name": "Dove Blue 450 ml"}],
            {},
            "missing reference views",
        ),
        (
            [
                {"class_id": 1, "barcode": "1", "sku_name": "Dove Blue 450 ml"},
                {"class_id": 1, "barcode": "2", "sku_name": "Sunsilk Pink 450 ml"},
            ],
            {1: [np.array([[[1.0, 0.0, 0.0]]])]},
            "duplicate class ID",
        ),
        (
            [{"class_id": 89, "barcode": "1", "sku_name": "Dove Blue 450 ml"}],
            {89: [np.array([[[1.0, 0.0, 0.0]]])]},
            "reserved class ID 89",
        ),
    ],
)
def test_build_prototypes_rejects_invalid_sku_identity_sets(rows, views, message):
    with pytest.raises(ValueError, match=message):
        build_prototypes(FakeBackend(), rows, views, matching_config())


class InvalidEmbeddingBackend(FakeBackend):
    def __init__(self, image_embeddings=None, text_embeddings=None):
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def encode_images(self, images_rgb):
        if self.image_embeddings is None:
            return super().encode_images(images_rgb)
        return np.asarray(self.image_embeddings, dtype=float)

    def encode_texts(self, texts):
        if self.text_embeddings is None:
            return super().encode_texts(texts)
        return np.asarray(self.text_embeddings, dtype=float)


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (InvalidEmbeddingBackend([[0.0, 0.0]]), "zero-norm image embeddings"),
        (InvalidEmbeddingBackend([[np.nan, 1.0]]), "non-finite image embeddings"),
        (
            InvalidEmbeddingBackend([[1.0, 0.0]], [[1.0, 0.0, 0.0]]),
            "inconsistent embedding dimensions",
        ),
    ],
)
def test_build_prototypes_rejects_invalid_backend_embeddings(backend, message):
    rows = [{"class_id": 0, "barcode": "111", "sku_name": "Dove Blue 450 ml"}]
    views = {0: [np.array([[[1.0, 0.0, 0.0]]])]}

    with pytest.raises(ValueError, match=message):
        build_prototypes(backend, rows, views, matching_config())


class CapturingBackend:
    def __init__(self):
        self.images = None

    def encode_images(self, images_rgb):
        self.images = images_rgb
        return np.array([[1.0, 0.0]])

    def encode_texts(self, texts):
        raise AssertionError("classification must use the prototype text embeddings")


def test_classification_clips_box_converts_crop_and_rejects_low_score():
    backend = CapturingBackend()
    image_bgr = np.zeros((3, 4, 3), dtype=float)
    image_bgr[:2, :2] = [0.0, -1.0, -1.0]
    bank = PrototypeBank(
        class_ids=(0, 1),
        barcodes=("111", "222"),
        sku_names=("Dove Blue 450 ml", "Sunsilk Pink 450 ml"),
        visual_embeddings=np.array([[-1.0, 0.0], [0.0, -1.0]]),
        text_embeddings=np.array([[-1.0, 0.0], [0.0, -1.0]]),
    )
    candidate = Candidate((-4, -3, 2, 2), 0.8, "bottle", 0, False)

    result = classify_candidates(
        backend, image_bgr, [candidate], bank, matching_config(min_score=0.24)
    )[0]

    assert np.array_equal(backend.images[0], image_bgr[:2, :2, ::-1])
    assert result.assigned_class_id == 89
    assert result.review_reason == "score_below_threshold"
    assert result.accepted is False


def test_classification_keeps_top_three_with_stable_score_and_class_id_order():
    backend = InvalidEmbeddingBackend(image_embeddings=[[1.0, 0.0]])
    bank = PrototypeBank(
        class_ids=(9, 2, 5, 3),
        barcodes=("9", "2", "5", "3"),
        sku_names=("nine", "two", "five", "three"),
        visual_embeddings=np.array([[1.0, 0.0]] * 4),
        text_embeddings=np.array([[1.0, 0.0]] * 4),
    )
    candidate = Candidate((0, 0, 1, 1), 0.8, "bottle", 0, False)

    result = classify_candidates(
        backend,
        np.array([[[0.0, 0.0, 1.0]]]),
        [candidate],
        bank,
        matching_config(min_score=0.0, min_margin=0.1),
    )[0]

    assert [ranking.class_id for ranking in result.rankings] == [2, 3, 5]
    assert result.review_reason == "top_one_margin_below_threshold"


def test_openclip_loader_names_model_and_weights_when_loading_fails(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("not cached")
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    with pytest.raises(RuntimeError, match="ViT-Z-99.*unavailable-weights"):
        OpenClipBackend("ViT-Z-99", "unavailable-weights", "cpu")


def test_openclip_loader_rejects_unavailable_cuda_before_model_load(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=lambda *args, **kwargs: pytest.fail(
            "model loading should not start"
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    with pytest.raises(RuntimeError, match="CUDA device 'cuda:2'.*unavailable"):
        OpenClipBackend("ViT-B-32", "weights", 2)


class FakeTensor:
    def __init__(self, value, events):
        self.value = np.asarray(value)
        self.events = events

    def to(self, device):
        self.events.append(("to", device))
        return self

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        return self

    def numpy(self):
        return self.value


class FakeInferenceMode:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("inference_enter")

    def __exit__(self, *args):
        self.events.append("inference_exit")


def uninitialized_openclip_backend():
    events = []
    backend = object.__new__(OpenClipBackend)
    backend._device = "mps"
    backend._image_type = types.SimpleNamespace(
        fromarray=lambda image: events.append(("fromarray", image.copy())) or image
    )
    backend._preprocess = lambda image: FakeTensor(image, events)
    backend._torch = types.SimpleNamespace(
        stack=lambda tensors: FakeTensor(
            np.stack([tensor.value for tensor in tensors]), events
        ),
        inference_mode=lambda: FakeInferenceMode(events),
    )
    backend._model = types.SimpleNamespace(
        encode_image=lambda batch: events.append(("image_batch", batch.value.shape))
        or FakeTensor([[1, 2]], events),
        encode_text=lambda tokens: events.append(("text_batch", tokens.value.shape))
        or FakeTensor([[3, 4], [5, 6]], events),
    )
    backend._tokenizer = lambda texts: events.append(("texts", texts)) or FakeTensor(
        [[1], [2]], events
    )
    return backend, events


def test_openclip_image_encoding_batches_under_inference_mode_and_returns_float64():
    backend, events = uninitialized_openclip_backend()
    image = np.ones((2, 3, 3), dtype=np.uint8)

    result = backend.encode_images([image])

    assert result.dtype == np.float64
    assert np.array_equal(result, np.array([[1.0, 2.0]]))
    assert ("image_batch", (1, 2, 3, 3)) in events
    assert events.index("inference_enter") < events.index(("image_batch", (1, 2, 3, 3)))
    assert ("to", "mps") in events
    assert "cpu" in events


def test_openclip_text_encoding_batches_under_inference_mode_and_returns_float64():
    backend, events = uninitialized_openclip_backend()

    result = backend.encode_texts(["first", "second"])

    assert result.dtype == np.float64
    assert np.array_equal(result, np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert ("texts", ["first", "second"]) in events
    assert ("text_batch", (2, 1)) in events
    assert events.index("inference_enter") < events.index(("text_batch", (2, 1)))
    assert ("to", "mps") in events
    assert "cpu" in events

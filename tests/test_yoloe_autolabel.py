import csv
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import yoloe_autolabel as app


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, images=None):
        self.images = {str(Path(k).resolve()): v for k, v in (images or {}).items()}
        self.writes = {}

    def imread(self, path):
        image = self.images.get(str(Path(path).resolve()))
        return None if image is None else image.copy()

    def imwrite(self, path, image):
        self.writes[str(Path(path))] = image.copy()
        Path(path).write_bytes(b"image")
        return True

    def resize(self, image, size):
        width, height = size
        return np.resize(image, (height, width, image.shape[2]))

    def rectangle(self, image, pt1, pt2, color, thickness):
        image[max(0, pt1[1]) : min(image.shape[0], pt1[1] + 1)] = color

    def putText(self, image, text, origin, font, scale, color, thickness, line):
        image[max(0, origin[1] - 1) : min(image.shape[0], origin[1])] = color


def sku(class_id, enabled=True, barcode=None):
    return app.Sku(class_id, barcode or f"00{class_id}", "Brand", f"SKU {class_id}", enabled)


def valid_config(**overrides):
    config = {
        "project": {},
        "dataset": {"manifest": "sku.csv", "references": "refs.yaml", "images": "images", "image_extensions": ["jpg"]},
        "yoloe": {"model": "yoloe-26l-seg.pt", "imgsz": 640, "conf": .25, "iou": .7, "device": "cpu", "prompt_batch_size": 1, "canvas_cell_size": 32, "canvas_padding": 2},
        "localization": {
            "tile_size": 1024,
            "overlap": 0.20,
            "prompts": ["shampoo bottle", "conditioner bottle", "hair treatment pouch", "boxed hair product", "hair care multipack"],
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
        "output": {"root": "output", "overwrite": False, "save_metadata": True, "save_previews": True, "low_confidence_cutoff": .2},
        "pilot": {"enabled": False, "class_ids": [], "max_images": 1, "max_skus": 1},
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def write_project(root, config=None):
    config = config or valid_config()
    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (root / "refs.yaml").write_text(json.dumps({"references": [{"class_id": 2, "image_path": "ref.jpg"}]}), encoding="utf-8")
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / "shelf.jpg").write_bytes(b"shelf-source")
    (root / "ref.jpg").write_bytes(b"reference-source")
    ManifestTests().write_manifest(root, [{"class_id": "2", "barcode": "002", "brand": "B", "sku_name": "N", "enabled": "true"}])
    return config_path


def fake_inference_environment(root, predictions=None):
    class Boxes:
        xyxy = np.empty((0, 4), dtype=float)
        conf = np.array([], dtype=float)
        cls = np.array([], dtype=float)

    if predictions:
        Boxes.xyxy = np.asarray([p[0] for p in predictions], dtype=float)
        Boxes.conf = np.asarray([p[1] for p in predictions], dtype=float)
        Boxes.cls = np.asarray([p[2] for p in predictions], dtype=float)
    model = mock.Mock()
    model.predict.return_value = [types.SimpleNamespace(boxes=Boxes())]
    cv2 = FakeCv2({root / "ref.jpg": np.ones((8, 8, 3), np.uint8), root / "images" / "shelf.jpg": np.zeros((20, 30, 3), np.uint8)})
    predictor_module = types.SimpleNamespace(YOLOEVPSegPredictor=object())
    return model, cv2, predictor_module


class ManifestTests(unittest.TestCase):
    def write_manifest(self, root, rows, columns=None):
        path = root / "sku.csv"
        columns = columns or ["class_id", "barcode", "brand", "sku_name", "enabled"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_manifest_preserves_barcode_and_trims_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.write_manifest(Path(td), [{"class_id": " 7 ", "barcode": " 001234 ", "brand": " Acme ", "sku_name": " Tea ", "enabled": " TRUE "}])
            actual = app.load_sku_manifest(path)
        self.assertEqual(actual[7], app.Sku(7, "001234", "Acme", "Tea", True))

    def test_manifest_rejects_invalid_shapes_and_values(self):
        invalid_rows = [
            [],
            [{"class_id": "x", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "true"}],
            [{"class_id": "-1", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "true"}],
            [{"class_id": "1", "barcode": "", "brand": "B", "sku_name": "N", "enabled": "true"}],
            [{"class_id": "1", "barcode": "1", "brand": "", "sku_name": "N", "enabled": "true"}],
            [{"class_id": "1", "barcode": "1", "brand": "B", "sku_name": "", "enabled": "true"}],
            [{"class_id": "1", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "yes"}],
            [
                {"class_id": "1", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "true"},
                {"class_id": "1", "barcode": "2", "brand": "B", "sku_name": "N", "enabled": "true"},
            ],
            [
                {"class_id": "1", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "true"},
                {"class_id": "2", "barcode": "1", "brand": "B", "sku_name": "N", "enabled": "false"},
            ],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as td:
                path = self.write_manifest(Path(td), rows)
                with self.assertRaises(ValueError):
                    app.load_sku_manifest(path)

    def test_manifest_requires_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.write_manifest(Path(td), [], ["class_id", "barcode"])
            with self.assertRaises(ValueError):
                app.load_sku_manifest(path)


class ConfigAndReferenceTests(unittest.TestCase):
    def test_load_config_validates_sections_and_positive_values(self):
        valid = valid_config(yoloe={"prompt_batch_size": 2, "canvas_cell_size": 100}, pilot={"max_images": 3, "max_skus": 2})
        fake_yaml = types.SimpleNamespace(safe_load=lambda _: valid)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"yaml": fake_yaml}):
            path = Path(td) / "config.yaml"
            path.write_text("ignored", encoding="utf-8")
            self.assertEqual(app.load_config(path), valid)
            for section, key in [("yoloe", "imgsz"), ("yoloe", "prompt_batch_size"), ("yoloe", "canvas_cell_size"), ("pilot", "max_images"), ("pilot", "max_skus")]:
                broken = json.loads(json.dumps(valid))
                broken[section][key] = 0
                fake_yaml.safe_load = lambda _, value=broken: value
                with self.assertRaises(ValueError):
                    app.load_config(path)

    def test_config_path_resolves_relative_to_config(self):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "conf" / "run.yaml"
            self.assertEqual(app.resolve_config_path(config, "../data/a.csv"), (config.parent / "../data/a.csv").resolve())

    def test_config_rejects_invalid_operational_values_before_validate_only(self):
        mutations = [
            ("yoloe", "imgsz", 640.0),
            ("yoloe", "model", ""),
            ("yoloe", "conf", math.nan),
            ("yoloe", "conf", 1.1),
            ("yoloe", "iou", -0.1),
            ("yoloe", "device", True),
            ("yoloe", "canvas_padding", 16),
            ("output", "overwrite", "false"),
            ("output", "save_metadata", 1),
            ("output", "save_previews", None),
            ("output", "root", None),
            ("pilot", "enabled", "true"),
            ("dataset", "image_extensions", "jpg"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            for section, key, value in mutations:
                with self.subTest(section=section, key=key, value=value):
                    config = valid_config()
                    config[section][key] = value
                    path.write_text(json.dumps(config, allow_nan=True), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        app.load_config(path)

    def test_fractional_reference_and_pilot_class_ids_are_rejected(self):
        fake_yaml = types.SimpleNamespace(safe_load=lambda _: {"references": [{"class_id": 4.5, "image": "x.jpg"}]})
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"yaml": fake_yaml}):
            path = Path(td) / "refs.yaml"
            path.write_text("ignored", encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_reference_definitions(path, {4: sku(4)})
        with self.assertRaises(ValueError):
            app.select_active_skus({4: sku(4)}, {"enabled": True, "class_ids": [4.5], "max_skus": 1})

    def test_reviewed_labels_inputs_are_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protected = root / "reviewed_labels"
            protected.mkdir()
            config_path = protected / "config.yaml"
            config_path.write_text(json.dumps(valid_config()), encoding="utf-8")
            manifest_path = protected / "sku.csv"
            ManifestTests().write_manifest(protected, [{"class_id": "2", "barcode": "002", "brand": "B", "sku_name": "N", "enabled": "true"}])
            reference_yaml = protected / "refs.yaml"
            reference_yaml.write_text(json.dumps({"references": []}), encoding="utf-8")
            (protected / "shelf.jpg").write_bytes(b"source")
            for operation in (
                lambda: app.load_config(config_path),
                lambda: app.load_sku_manifest(manifest_path),
                lambda: app.load_reference_definitions(reference_yaml, {2: sku(2)}),
                lambda: app.discover_shelf_images(protected, ["jpg"]),
            ):
                with self.subTest(operation=operation), self.assertRaises(ValueError):
                    operation()
            outside_yaml = root / "refs.yaml"
            outside_yaml.write_text(json.dumps({"references": [{"class_id": 2, "image": "reviewed_labels/shelf.jpg"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_reference_definitions(outside_yaml, {2: sku(2)})

    def test_reference_validation_full_image_and_bounds(self):
        skus = {4: sku(4)}
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.assertEqual(app.validate_reference(app.ReferencePrompt(4, Path("a.jpg"), None), image, skus), (0, 0, 30, 20))
        self.assertEqual(app.validate_reference(app.ReferencePrompt(4, Path("a.jpg"), (1, 2, 29, 19)), image, skus), (1, 2, 29, 19))
        for bbox in [(-1, 0, 2, 2), (0, 0, 31, 2), (2, 2, 2, 3), (0, 0, math.inf, 2)]:
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                app.validate_reference(app.ReferencePrompt(4, Path("a.jpg"), bbox), image, skus)
        with self.assertRaises(ValueError):
            app.validate_reference(app.ReferencePrompt(99, Path("a.jpg"), None), image, skus)
        with self.assertRaises(ValueError):
            app.validate_reference(app.ReferencePrompt(4, Path("a.jpg"), None), np.zeros((0, 1, 3)), skus)

    def test_reference_yaml_rejects_unknown_and_keeps_multiple_views(self):
        payload = {"references": [
            {"class_id": 4, "image_path": "one.jpg"},
            {"class_id": 4, "image_path": "two.jpg", "bbox": [1, 2, 3, 4]},
            {"class_id": 8, "image_path": "disabled.jpg"},
        ]}
        fake_yaml = types.SimpleNamespace(safe_load=lambda _: payload)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"yaml": fake_yaml}):
            path = Path(td) / "refs.yaml"
            path.write_text("ignored", encoding="utf-8")
            refs = app.load_reference_definitions(path, {4: sku(4), 8: sku(8, False)})
            self.assertEqual([r.class_id for r in refs], [4, 4])
            self.assertEqual(refs[1].bbox, (1.0, 2.0, 3.0, 4.0))
            fake_yaml.safe_load = lambda _: {"references": [{"class_id": 99, "image_path": "x.jpg"}]}
            with self.assertRaises(ValueError):
                app.load_reference_definitions(path, {4: sku(4)})


class SelectionAndDiscoveryTests(unittest.TestCase):
    def test_pilot_selection_sorts_and_validates(self):
        skus = {9: sku(9), 2: sku(2), 5: sku(5, False), 7: sku(7)}
        self.assertEqual([s.class_id for s in app.select_active_skus(skus, {"enabled": False, "max_skus": 1})], [2, 7, 9])
        self.assertEqual([s.class_id for s in app.select_active_skus(skus, {"enabled": True, "class_ids": [9, 2], "max_skus": 1})], [2, 9])
        self.assertEqual([s.class_id for s in app.select_active_skus(skus, {"enabled": True, "max_skus": 2})], [2, 7])
        self.assertEqual([s.class_id for s in app.select_active_skus(skus, {"enabled": True, "class_ids": [], "max_skus": 2})], [2, 7])
        with self.assertRaises(ValueError):
            app.select_active_skus(skus, {"enabled": True, "class_ids": [5], "max_skus": 2})
        with self.assertRaises(ValueError):
            app.select_active_skus({5: sku(5, False)}, {"enabled": False, "max_skus": 2})

    def test_discovery_is_case_insensitive_sorted_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b.JPG").write_bytes(b"")
            (root / "A.jpg").write_bytes(b"")
            (root / "ignore.png").write_bytes(b"")
            self.assertEqual([p.name for p in app.discover_shelf_images(root, [".jpg"])], ["A.jpg", "b.JPG"])
            (root / "a.jpeg").write_bytes(b"")
            with self.assertRaises(ValueError):
                app.discover_shelf_images(root, ["jpg", "jpeg"])
        with self.assertRaises(ValueError):
            app.discover_shelf_images(Path("missing-directory"), ["jpg"])


class BatchAndInferenceTests(unittest.TestCase):
    def test_batches_map_noncontiguous_ids_and_repeat_temporary_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [root / f"r{i}.jpg" for i in range(3)]
            images = {paths[0]: np.ones((10, 20, 3), np.uint8), paths[1]: np.ones((8, 12, 3), np.uint8), paths[2]: np.ones((9, 9, 3), np.uint8)}
            for path in paths:
                path.write_bytes(b"source")
            refs = [app.ReferencePrompt(2, paths[0], None), app.ReferencePrompt(2, paths[1], None), app.ReferencePrompt(9, paths[2], None)]
            cv2 = FakeCv2(images)
            with mock.patch.object(app, "_load_cv2", return_value=cv2):
                batches = app.build_prompt_batches(refs, [sku(9), sku(2)], {"prompt_batch_size": 2, "canvas_cell_size": 40, "canvas_padding": 4}, root / "output" / "raw_predictions" / "prompt_canvases")
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0].dataset_class_ids, (2, 9))
            self.assertEqual(batches[0].prompt_to_dataset_class, {0: 2, 1: 9})
            np.testing.assert_array_equal(batches[0].temporary_class_ids, np.array([0, 0, 1]))
            self.assertEqual(batches[0].canvas_path.name, "batch_000.png")
            self.assertTrue(batches[0].canvas_path.exists())

    def test_batches_validate_every_reference_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good, missing = root / "good.jpg", root / "missing.jpg"
            good.write_bytes(b"source")
            cv2 = FakeCv2({good: np.ones((5, 5, 3), np.uint8)})
            refs = [app.ReferencePrompt(2, good, None), app.ReferencePrompt(2, missing, None)]
            with mock.patch.object(app, "_load_cv2", return_value=cv2), self.assertRaises(ValueError):
                app.build_prompt_batches(refs, [sku(2)], {"prompt_batch_size": 1, "canvas_cell_size": 20, "canvas_padding": 2}, root / "canvases")
            self.assertFalse((root / "canvases").exists())

    def test_canvas_directory_must_be_guarded_and_existing_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reference = root / "ref.jpg"
            reference.write_bytes(b"reference-source")
            cv2 = FakeCv2({reference: np.ones((5, 5, 3), np.uint8)})
            prompt = app.ReferencePrompt(2, reference, None)
            settings = {"prompt_batch_size": 1, "canvas_cell_size": 20, "canvas_padding": 2}
            with mock.patch.object(app, "_load_cv2", return_value=cv2), self.assertRaises(ValueError):
                app.build_prompt_batches([prompt], [sku(2)], settings, root / "unguarded")
            canvas_dir = root / "output" / "raw_predictions" / "prompt_canvases"
            canvas_dir.mkdir(parents=True)
            os.link(reference, canvas_dir / "batch_000.png")
            with mock.patch.object(app, "_load_cv2", return_value=cv2), self.assertRaises(ValueError):
                app.build_prompt_batches([prompt], [sku(2)], settings, canvas_dir)
            self.assertEqual(reference.read_bytes(), b"reference-source")

    def test_run_yoloe_translates_ids_and_uses_pinned_call_shape(self):
        class Boxes:
            xyxy = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)
            conf = np.array([0.9, 0.5], float)
            cls = np.array([0, 1], float)
        model = mock.Mock()
        model.predict.return_value = [types.SimpleNamespace(boxes=Boxes())]
        predictor = object()
        fake_module = types.SimpleNamespace(YOLOEVPSegPredictor=predictor)
        batch = app.PromptBatch(3, (2, 9), {0: 2, 1: 9}, Path("canvas.jpg"), np.array([[0, 0, 2, 2]]), np.array([0]))
        with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": fake_module}):
            predictions = app.run_yoloe(model, Path("shelf.jpg"), batch, {"imgsz": 640, "conf": .2, "iou": .7, "device": "cpu"}, {2: sku(2), 9: sku(9)})
        self.assertEqual([p.dataset_class_id for p in predictions], [2, 9])
        self.assertEqual(predictions[0].prompt_batch, 3)
        model.predict.assert_called_once_with(source="shelf.jpg", refer_image="canvas.jpg", visual_prompts={"bboxes": batch.boxes, "cls": batch.temporary_class_ids}, predictor=predictor, imgsz=640, conf=.2, iou=.7, device="cpu")

    def test_run_yoloe_empty_and_malformed_outputs(self):
        fake_module = types.SimpleNamespace(YOLOEVPSegPredictor=object())
        batch = app.PromptBatch(0, (2,), {0: 2}, Path("c.jpg"), np.empty((0, 4)), np.empty((0,)))
        with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": fake_module}):
            empty = mock.Mock()
            empty.predict.return_value = [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=np.empty((0, 4)), conf=np.array([]), cls=np.array([])))]
            self.assertEqual(app.run_yoloe(empty, Path("x.jpg"), batch, {"imgsz": 1, "conf": 0, "iou": 0, "device": "cpu"}, {2: sku(2)}), [])
            for cls in [np.array([8.0]), np.array([0.5]), np.array([math.nan])]:
                bad = mock.Mock()
                bad.predict.return_value = [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=np.array([[1, 2, 3, 4]], float), conf=np.array([.5]), cls=cls))]
                with self.assertRaises(RuntimeError):
                    app.run_yoloe(bad, Path("x.jpg"), batch, {"imgsz": 1, "conf": 0, "iou": 0, "device": "cpu"}, {2: sku(2)})

    def test_run_yoloe_rejects_unequal_lengths_bad_confidence_and_coordinates(self):
        fake_module = types.SimpleNamespace(YOLOEVPSegPredictor=object())
        batch = app.PromptBatch(0, (2,), {0: 2}, Path("c.jpg"), np.empty((0, 4)), np.empty((0,)))
        cases = [
            (np.array([[1, 2, 3, 4]], float), np.array([], float), np.array([0], float)),
            (np.array([[1, 2, math.inf, 4]], float), np.array([.5], float), np.array([0], float)),
            (np.array([[1, 2, 3, 4]], float), np.array([1.1], float), np.array([0], float)),
            (np.array([[1, 2, 3, 4]], float), np.array([math.nan], float), np.array([0], float)),
        ]
        with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": fake_module}):
            for xyxy, conf, cls in cases:
                with self.subTest(xyxy=xyxy, conf=conf):
                    model = mock.Mock()
                    model.predict.return_value = [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=xyxy, conf=conf, cls=cls))]
                    with self.assertRaises(RuntimeError):
                        app.run_yoloe(model, Path("x.jpg"), batch, {"imgsz": 1, "conf": 0, "iou": 0, "device": "cpu"}, {2: sku(2)})

    def test_run_yoloe_rejects_protected_source_and_canvas_before_predict(self):
        model = mock.Mock()
        batch = app.PromptBatch(0, (2,), {0: 2}, Path("safe") / "raw_predictions" / "prompt_canvases" / "batch_000.png", np.empty((0, 4)), np.empty((0,)))
        settings = {"imgsz": 640, "conf": .2, "iou": .7, "device": "cpu"}
        with self.assertRaises(ValueError):
            app.run_yoloe(model, Path("reviewed_labels") / "shelf.jpg", batch, settings, {2: sku(2)})
        protected_batch = app.PromptBatch(0, (2,), {0: 2}, Path("reviewed_labels") / "batch.png", np.empty((0, 4)), np.empty((0,)))
        with self.assertRaises(ValueError):
            app.run_yoloe(model, Path("shelf.jpg"), protected_batch, settings, {2: sku(2)})
        model.predict.assert_not_called()

    def test_load_yoloe_rejects_protected_local_path_before_constructor_but_allows_alias(self):
        constructor = mock.Mock(return_value=object())
        fake_ultralytics = types.SimpleNamespace(YOLOE=constructor)
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with self.assertRaises(ValueError):
                app.load_yoloe(str(Path("reviewed_labels") / "model.pt"))
            constructor.assert_not_called()
            model = app.load_yoloe("yoloe-26l-seg.pt")
        self.assertIsNotNone(model)
        constructor.assert_called_once_with("yoloe-26l-seg.pt")


class ConversionAndOutputTests(unittest.TestCase):
    def test_conversion_exact_clips_and_rejects_invalid(self):
        self.assertEqual(app.convert_xyxy_to_yolo((100, 50, 300, 250), 1000, 500), (.2, .3, .2, .4))
        self.assertEqual(app.convert_xyxy_to_yolo((-10, -5, 20, 10), 100, 50), (.1, .1, .2, .2))
        for box, width, height in [((1, 1, 1, 2), 10, 10), ((1, 1, math.nan, 2), 10, 10), ((1, 1, 2, 2), 0, 10)]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                app.convert_xyxy_to_yolo(box, width, height)

    def test_output_layout_and_containment_reject_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "out"
            paths = app.prepare_output_paths(root)
            self.assertEqual(paths["labels"], root.resolve() / "raw_predictions" / "labels")
            self.assertTrue(paths["prompt_canvases"].is_dir())
            with self.assertRaises(ValueError):
                app.prepare_output_paths(Path(td) / "reviewed_labels" / "run")
            with self.assertRaises(ValueError):
                app.save_yolo_labels(Path(td) / "outside.txt", [], 10, 10, False)

    def test_prepare_output_rejects_raw_predictions_symlink_redirection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            try:
                (output / "raw_predictions").symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                original_resolve = Path.resolve
                outside_resolved = original_resolve(outside)
                redirected_path = Path(os.path.abspath(output / "raw_predictions"))

                def resolve_with_redirect(path, strict=False):
                    if Path(os.path.abspath(path)) == redirected_path:
                        return outside_resolved
                    return original_resolve(path, strict=strict)

                with mock.patch.object(Path, "resolve", resolve_with_redirect), self.assertRaises(ValueError):
                    app.prepare_output_paths(output)
            else:
                with self.assertRaises(ValueError):
                    app.prepare_output_paths(output)
            self.assertEqual(list(outside.iterdir()), [])

    def test_skip_empty_atomic_and_deterministic_labels(self):
        with tempfile.TemporaryDirectory() as td:
            paths = app.prepare_output_paths(Path(td) / "out")
            target = paths["labels"] / "image.txt"
            app.save_yolo_labels(target, [], 10, 10, False)
            self.assertEqual(target.read_text(encoding="utf-8"), "")
            self.assertTrue(app.should_skip_image("image", paths, False))
            self.assertFalse(app.should_skip_image("image", paths, True))
            predictions = [
                app.Prediction(9, 0, "9", "Nine", .3, (5, 5, 8, 8), 0),
                app.Prediction(2, 0, "2", "Two", .4, (4, 4, 8, 8), 0),
                app.Prediction(2, 0, "2", "Two", .9, (1, 1, 3, 3), 0),
            ]
            with self.assertRaises(FileExistsError):
                app.save_yolo_labels(target, predictions, 10, 10, False)
            app.save_yolo_labels(target, predictions, 10, 10, True)
            rows = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual([row.split()[0] for row in rows], ["2", "2", "9"])
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_metadata_is_serializable_and_preview_preserves_source(self):
        prediction = app.Prediction(2, 0, "002", "Tea", .8, (1, 1, 5, 5), 0)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "out"
            paths = app.prepare_output_paths(root)
            image_path = Path(td) / "source.jpg"
            image_path.write_bytes(b"source")
            source = np.zeros((10, 12, 3), np.uint8)
            cv2 = FakeCv2({image_path: source})
            with mock.patch.object(app, "_load_cv2", return_value=cv2):
                app.save_metadata(paths["metadata"] / "source.json", image_path, [prediction], {"model": Path("m.pt"), "imgsz": 640}, False)
                before = source.copy()
                app.save_preview(paths["previews"] / "source.jpg", source, [prediction], False)
            metadata = json.loads((paths["metadata"] / "source.json").read_text(encoding="utf-8"))
            self.assertEqual((metadata["image_width"], metadata["image_height"]), (12, 10))
            self.assertEqual(metadata["predictions"][0]["barcode"], "002")
            self.assertEqual(source.tolist(), before.tolist())
            self.assertEqual(next(iter(cv2.writes.values())).shape, source.shape)

    def test_metadata_rejects_protected_source_before_imread(self):
        cv2 = mock.Mock()
        with tempfile.TemporaryDirectory() as td, mock.patch.object(app, "_load_cv2", return_value=cv2):
            paths = app.prepare_output_paths(Path(td) / "out")
            with self.assertRaises(ValueError):
                app.save_metadata(paths["metadata"] / "x.json", Path(td) / "reviewed_labels" / "x.jpg", [], {}, False)
        cv2.imread.assert_not_called()

    def test_summary_reports_zero_and_low_confidence_classes(self):
        with tempfile.TemporaryDirectory() as td:
            paths = app.prepare_output_paths(Path(td) / "out")
            preds = {Path("a.jpg"): [app.Prediction(2, 0, "002", "Tea", .4, (1, 1, 2, 2), 0)]}
            app.save_summary(paths["summary_csv"], paths["run_json"], [sku(2), sku(9)], preds, {"processed": 1, "skipped": 2, "errors": 0, "no_detection_images": 0, "low_confidence_threshold": .5})
            payload = json.loads(paths["run_json"].read_text(encoding="utf-8"))
            by_id = {row["class_id"]: row for row in payload["classes"]}
            self.assertTrue(by_id[2]["low_confidence"])
            self.assertEqual(by_id[9]["candidate_count"], 0)
            self.assertTrue(by_id[9]["zero_detection"])
            self.assertEqual(payload["counters"]["skipped"], 2)


class CliTests(unittest.TestCase):
    def test_cli_precedence_and_positive_gate(self):
        args = app.parse_args(["--config", "c.yaml", "--output", "override", "--overwrite", "--require-enabled-count", "4"])
        self.assertEqual(args.output, "override")
        self.assertTrue(args.overwrite)
        self.assertEqual(args.require_enabled_count, 4)
        with self.assertRaises(SystemExit):
            app.parse_args(["--config", "c.yaml", "--require-enabled-count", "0"])

    def test_validate_only_never_loads_model_or_creates_output(self):
        config = valid_config(yoloe={"model": "model.pt"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.yaml"
            config_path.write_text("ignored", encoding="utf-8")
            (root / "refs.yaml").write_text("ignored", encoding="utf-8")
            (root / "images").mkdir()
            (root / "images" / "shelf.jpg").write_bytes(b"source")
            (root / "ref.jpg").write_bytes(b"source")
            ManifestTests().write_manifest(root, [{"class_id": "2", "barcode": "002", "brand": "B", "sku_name": "N", "enabled": "true"}])
            fake_yaml = types.SimpleNamespace(safe_load=lambda stream: config if Path(stream.name).name == "config.yaml" else {"references": [{"class_id": 2, "image_path": "ref.jpg"}]})
            cv2 = FakeCv2({root / "ref.jpg": np.ones((5, 5, 3), np.uint8)})
            with mock.patch.dict(sys.modules, {"yaml": fake_yaml}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", side_effect=AssertionError("must not load")):
                result = app.main(["--config", str(config_path), "--validate-only", "--require-enabled-count", "1"])
            self.assertEqual(result, 0)
            self.assertFalse((root / "output").exists())
            self.assertEqual(cv2.writes, {})

    def test_cli_output_supplies_optional_config_root_but_one_effective_root_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = valid_config()
            del config["output"]["root"]
            config_path = write_project(root, config)
            cv2 = FakeCv2({root / "ref.jpg": np.ones((5, 5, 3), np.uint8)})
            self.assertEqual(app.load_config(config_path), config)
            with mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", side_effect=AssertionError("must not load")):
                self.assertEqual(app.main(["--config", str(config_path), "--output", "cli-output", "--validate-only"]), 0)
                self.assertNotEqual(app.main(["--config", str(config_path), "--validate-only"]), 0)
            self.assertFalse((root / "cli-output").exists())

    def test_nested_reference_yaml_resolves_images_from_main_config_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = valid_config(dataset={"references": "definitions/nested/prompts.yaml"})
            config_path = write_project(root, config)
            prompt_file = root / "definitions" / "nested" / "prompts.yaml"
            prompt_file.parent.mkdir(parents=True)
            prompt_file.write_text(json.dumps({"references": [{"class_id": 2, "image": "references/ref.jpg"}]}), encoding="utf-8")
            reference = root / "references" / "ref.jpg"
            reference.parent.mkdir()
            reference.write_bytes(b"reference-source")
            cv2 = FakeCv2({reference: np.ones((5, 5, 3), np.uint8)})
            with mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", side_effect=AssertionError("must not load")):
                result = app.main(["--config", str(config_path), "--validate-only"])
            self.assertEqual(result, 0)

    def test_fake_model_cli_precedence_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = write_project(root)
            model, cv2, predictor_module = fake_inference_environment(root)
            shelf_before = (root / "images" / "shelf.jpg").read_bytes()
            reference_before = (root / "ref.jpg").read_bytes()
            cli_paths = app.prepare_output_paths(root / "cli-output")
            label = cli_paths["labels"] / "shelf.txt"
            label.write_text("sentinel\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model), mock.patch.object(app, "_ultralytics_version", return_value="test"):
                skipped = app.main(["--config", str(config_path), "--output", "cli-output"])
                self.assertEqual(label.read_text(encoding="utf-8"), "sentinel\n")
                result = app.main(["--config", str(config_path), "--output", "cli-output", "--overwrite"])
            self.assertNotEqual(skipped, 0)
            self.assertEqual(result, 0)
            self.assertEqual(label.read_text(encoding="utf-8"), "")
            self.assertEqual(model.predict.call_count, 1)
            self.assertFalse((root / "output").exists())
            self.assertEqual((root / "images" / "shelf.jpg").read_bytes(), shelf_before)
            self.assertEqual((root / "ref.jpg").read_bytes(), reference_before)
            run = json.loads((root / "cli-output" / "raw_predictions" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["effective_paths"]["output_root"], str((root / "cli-output").resolve()))
            self.assertEqual(run["cli_overrides"], {"output": "cli-output", "overwrite": True})
            self.assertEqual(run["errors"], [])
            self.assertGreaterEqual(run["elapsed_seconds"], 0)

    def test_enabled_count_mismatch_precedes_model_load_and_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = write_project(root)
            with mock.patch.object(app, "load_yoloe", side_effect=AssertionError("must not load")):
                result = app.main(["--config", str(config_path), "--require-enabled-count", "2"])
            self.assertNotEqual(result, 0)
            self.assertFalse((root / "output").exists())

    def test_failed_label_leaves_retryable_sidecars_and_label_is_last(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = write_project(root)
            model, cv2, predictor_module = fake_inference_environment(root)
            real_save_labels = app.save_yolo_labels
            environment = mock.patch.multiple(app, _load_cv2=mock.DEFAULT, load_yoloe=mock.DEFAULT, _ultralytics_version=mock.DEFAULT)
            with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), environment as patched:
                patched["_load_cv2"].return_value = cv2
                patched["load_yoloe"].return_value = model
                patched["_ultralytics_version"].return_value = "test"
                with mock.patch.object(app, "save_yolo_labels", side_effect=RuntimeError("simulated label failure")):
                    first = app.main(["--config", str(config_path)])
                raw = root / "output" / "raw_predictions"
                self.assertEqual(first, 1)
                self.assertTrue((raw / "metadata" / "shelf.json").exists())
                self.assertTrue((raw / "previews" / "shelf.jpg").exists())
                self.assertFalse((raw / "labels" / "shelf.txt").exists())
                with mock.patch.object(app, "save_yolo_labels", wraps=real_save_labels):
                    second = app.main(["--config", str(config_path)])
            self.assertEqual(second, 0)
            self.assertTrue((raw / "labels" / "shelf.txt").exists())

    def test_failed_overwrite_invalidates_old_marker_so_ordinary_retry_processes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = write_project(root)
            model, cv2, predictor_module = fake_inference_environment(root)
            real_save_labels = app.save_yolo_labels
            raw = root / "output" / "raw_predictions"
            with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model), mock.patch.object(app, "_ultralytics_version", return_value="test"):
                self.assertEqual(app.main(["--config", str(config_path)]), 0)
                label = raw / "labels" / "shelf.txt"
                self.assertTrue(label.exists())
                stale_label = raw / "labels" / "unselected-stale.txt"
                stale_label.write_text("unrelated\n", encoding="utf-8")
                with mock.patch.object(app, "save_yolo_labels", side_effect=RuntimeError("simulated overwrite label failure")):
                    self.assertEqual(app.main(["--config", str(config_path), "--overwrite"]), 1)
                self.assertFalse(label.exists())
                self.assertEqual(stale_label.read_text(encoding="utf-8"), "unrelated\n")
                with mock.patch.object(app, "save_yolo_labels", wraps=real_save_labels):
                    self.assertEqual(app.main(["--config", str(config_path)]), 0)
            self.assertTrue(label.exists())
            self.assertEqual(model.predict.call_count, 3)

    def test_marker_invalidation_unlinks_selected_symlink_entry_without_resolving_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = app.prepare_output_paths(root / "output")
            selected = paths["labels"] / "shelf.txt"
            stale = paths["labels"] / "unselected.txt"
            reference = root / "ref.jpg"
            selected.write_text("simulated-link-entry", encoding="utf-8")
            stale.write_text("stale", encoding="utf-8")
            reference.write_bytes(b"reference-target")
            original_resolve = Path.resolve
            original_is_symlink = Path.is_symlink
            original_samefile = os.path.samefile

            def guarded_resolve(path, strict=False):
                if Path(os.path.abspath(path)) == Path(os.path.abspath(selected)):
                    raise AssertionError("completion marker itself must never be resolved")
                return original_resolve(path, strict=strict)

            def simulated_symlink(path):
                return Path(os.path.abspath(path)) == Path(os.path.abspath(selected)) or original_is_symlink(path)

            def simulated_samefile(left, right):
                pair = {Path(os.path.abspath(left)), Path(os.path.abspath(right))}
                if pair == {Path(os.path.abspath(selected)), Path(os.path.abspath(reference))}:
                    return True
                return original_samefile(left, right)

            with mock.patch.object(Path, "resolve", guarded_resolve), mock.patch.object(Path, "is_symlink", simulated_symlink), mock.patch("os.path.samefile", simulated_samefile):
                app._invalidate_labels([selected], paths["labels"], paths["raw_predictions"], [reference])
            self.assertFalse(selected.exists())
            self.assertEqual(stale.read_text(encoding="utf-8"), "stale")
            self.assertEqual(reference.read_bytes(), b"reference-target")

    def test_same_provenance_resume_preserves_existing_canvas(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = write_project(root)
            model, cv2, predictor_module = fake_inference_environment(root)
            with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model), mock.patch.object(app, "_ultralytics_version", return_value="test"):
                self.assertEqual(app.main(["--config", str(config_path)]), 0)
                raw = root / "output" / "raw_predictions"
                provenance = raw / "provenance.json"
                canvas = raw / "prompt_canvases" / "batch_000.png"
                self.assertTrue(provenance.exists())
                canvas.write_bytes(b"preserve-existing-canvas")
                self.assertEqual(app.main(["--config", str(config_path)]), 0)
            self.assertEqual(canvas.read_bytes(), b"preserve-existing-canvas")
            self.assertEqual(model.predict.call_count, 1)

    def test_provenance_change_rejects_reference_or_config_before_mutation(self):
        for change in ("reference", "config"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                config_path = write_project(root)
                model, cv2, predictor_module = fake_inference_environment(root)
                with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model), mock.patch.object(app, "_ultralytics_version", return_value="test"):
                    self.assertEqual(app.main(["--config", str(config_path)]), 0)
                    canvas = root / "output" / "raw_predictions" / "prompt_canvases" / "batch_000.png"
                    canvas.write_bytes(b"old-provenance-canvas")
                    if change == "reference":
                        (root / "ref.jpg").write_bytes(b"changed-reference-content")
                    else:
                        config = valid_config(yoloe={"conf": .35})
                        config_path.write_text(json.dumps(config), encoding="utf-8")
                    self.assertNotEqual(app.main(["--config", str(config_path)]), 0)
                self.assertEqual(canvas.read_bytes(), b"old-provenance-canvas")
                self.assertEqual(model.predict.call_count, 1)

    def test_changed_provenance_rejects_new_image_subset_and_preserves_unselected_artifacts(self):
        for overwrite_args in ([], ["--overwrite"]):
            with self.subTest(overwrite=bool(overwrite_args)), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                config_path = write_project(root)
                model, cv2, predictor_module = fake_inference_environment(root)
                with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model), mock.patch.object(app, "_ultralytics_version", return_value="test"):
                    self.assertEqual(app.main(["--config", str(config_path)]), 0)
                    raw = root / "output" / "raw_predictions"
                    old_label = raw / "labels" / "shelf.txt"
                    canvas = raw / "prompt_canvases" / "batch_000.png"
                    run_json = raw / "run.json"
                    before = (old_label.read_bytes(), canvas.read_bytes(), run_json.read_bytes())

                    new_images = root / "new-images"
                    new_images.mkdir()
                    new_image = new_images / "new.jpg"
                    new_image.write_bytes(b"new-shelf-source")
                    cv2.images[str(new_image)] = np.zeros((20, 30, 3), np.uint8)
                    changed = valid_config(dataset={"images": "new-images"}, yoloe={"conf": .35})
                    config_path.write_text(json.dumps(changed), encoding="utf-8")
                    self.assertNotEqual(app.main(["--config", str(config_path), *overwrite_args]), 0)

                self.assertEqual((old_label.read_bytes(), canvas.read_bytes(), run_json.read_bytes()), before)
                self.assertFalse((raw / "labels" / "new.txt").exists())
                self.assertEqual(model.predict.call_count, 1)

    def test_input_output_alias_is_rejected_before_model_load_or_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = valid_config(dataset={"images": "output/raw_predictions/previews"})
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (root / "refs.yaml").write_text(json.dumps({"references": [{"class_id": 2, "image": "ref.jpg"}]}), encoding="utf-8")
            (root / "ref.jpg").write_bytes(b"reference-source")
            ManifestTests().write_manifest(root, [{"class_id": "2", "barcode": "002", "brand": "B", "sku_name": "N", "enabled": "true"}])
            preview_dir = root / "output" / "raw_predictions" / "previews"
            preview_dir.mkdir(parents=True)
            shelf = preview_dir / "shelf.jpg"
            shelf.write_bytes(b"shelf-source")
            before = shelf.read_bytes()
            model, cv2, predictor_module = fake_inference_environment(root)
            cv2.images[str(shelf)] = np.zeros((20, 30, 3), np.uint8)
            with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model) as loader:
                result = app.main(["--config", str(config_path)])
            self.assertNotEqual(result, 0)
            loader.assert_not_called()
            self.assertEqual(shelf.read_bytes(), before)

    def test_path_like_model_is_resolved_but_bare_alias_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for model_value in ("models/custom.pt", "custom.pt", "yoloe-26l-seg.pt"):
                with self.subTest(model=model_value):
                    run_root = root / model_value.replace("/", "_").replace(".", "_")
                    run_root.mkdir()
                    config = valid_config(yoloe={"model": model_value})
                    config_path = write_project(run_root, config)
                    model, cv2, predictor_module = fake_inference_environment(run_root)
                    with mock.patch.dict(sys.modules, {"ultralytics.models.yolo.yoloe": predictor_module}), mock.patch.object(app, "_load_cv2", return_value=cv2), mock.patch.object(app, "load_yoloe", return_value=model) as loader, mock.patch.object(app, "_ultralytics_version", return_value="test"):
                        self.assertEqual(app.main(["--config", str(config_path)]), 0)
                    if model_value == "yoloe-26l-seg.pt":
                        wanted = model_value
                    else:
                        wanted = str((run_root / model_value).resolve())
                    loader.assert_called_once_with(wanted)


if __name__ == "__main__":
    unittest.main()

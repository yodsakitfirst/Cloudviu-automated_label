import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest
import yaml

import colab_runtime
import yoloe_autolabel
from test_platform_export import platform_project


def test_colab_config_is_a_valid_single_registry_full_run():
    config_path = Path("colab/config.yaml")

    config = yoloe_autolabel.load_config(config_path)

    assert config["dataset"] == {
        "sku_manifest": "sku_manifest.csv",
        "references": "reference_prompts.yaml",
        "shelf_images": "shelf_images",
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    }
    assert config["output"]["root"] == "output"
    assert config["pilot"]["enabled"] is False


def test_write_pilot_config_changes_only_pilot_and_output(tmp_path):
    project = tmp_path / "hair_colab"
    project.mkdir()
    source = {
        "project": {"name": "hair_osa_89"},
        "dataset": {"sku_manifest": "sku_manifest.csv"},
        "yoloe": {"model": "yoloe-26l-seg.pt"},
        "output": {"root": "output", "overwrite": False},
        "pilot": {"enabled": False, "class_ids": [], "max_skus": 5, "max_images": 10},
    }
    (project / "config.yaml").write_text(
        yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
    )

    pilot_path = colab_runtime.write_pilot_config(project)

    assert yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8")) == source
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))
    assert pilot["project"] == source["project"]
    assert pilot["dataset"] == source["dataset"]
    assert pilot["yoloe"] == source["yoloe"]
    assert pilot["pilot"] == {
        "enabled": True,
        "class_ids": [],
        "max_skus": 89,
        "max_images": 10,
    }
    assert pilot["output"] == {"root": "pilot_output", "overwrite": False}


def test_package_results_builds_separate_platform_and_review_archives(platform_project, tmp_path):
    platform = tmp_path / "platform.zip"
    review = tmp_path / "review.zip"

    actual = colab_runtime.package_results(platform_project, platform, review)

    assert actual == (platform.resolve(), review.resolve())
    with zipfile.ZipFile(platform) as archive:
        assert "data.yaml" in archive.namelist()
        assert "review_queue.csv" not in archive.namelist()
    with zipfile.ZipFile(review) as archive:
        assert "review_queue.csv" in archive.namelist()
        assert "data.yaml" not in archive.namelist()


def test_package_results_refuses_missing_or_existing_destinations(platform_project, tmp_path):
    with pytest.raises(ValueError):
        colab_runtime.package_results(tmp_path / "missing", tmp_path / "results.zip", tmp_path / "review.zip")
    destination = tmp_path / "results.zip"
    destination.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError):
        colab_runtime.package_results(platform_project, destination, tmp_path / "review.zip")
    assert destination.read_bytes() == b"sentinel"


def test_package_results_preflights_both_destinations(platform_project, tmp_path):
    platform = tmp_path / "platform.zip"
    review = tmp_path / "review.zip"
    review.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError):
        colab_runtime.package_results(platform_project, platform, review)
    assert not platform.exists()
    assert review.read_bytes() == b"sentinel"
    with pytest.raises(ValueError, match="distinct|same|alias"):
        colab_runtime.package_results(platform_project, platform, platform)


def test_package_results_honors_export_config_and_explicit_overwrite(platform_project, tmp_path):
    config = {"dataset": {"sku_manifest": "sku_manifest.csv", "shelf_images": "shelf_images"}, "output": {"root": "output"}, "export": {"train_fraction": 0.5, "split_seed": "configured-seed"}}
    (platform_project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    platform = tmp_path / "platform.zip"
    review = tmp_path / "review.zip"
    for destination in (platform, review):
        destination.write_bytes(b"sentinel")
    colab_runtime.package_results(platform_project, platform, review, overwrite=True)
    with zipfile.ZipFile(platform) as archive:
        assert len([name for name in archive.namelist() if name.startswith("images/train/")]) == 5
        assert len([name for name in archive.namelist() if name.startswith("images/val/")]) == 5


def test_package_results_preflights_review_inputs_before_platform_write(platform_project, tmp_path):
    (platform_project / "output/raw_predictions/review_queue.csv").unlink()
    platform = tmp_path / "platform.zip"
    with pytest.raises(ValueError):
        colab_runtime.package_results(platform_project, platform, tmp_path / "review.zip")
    assert not platform.exists()


def test_notebook_is_valid_json_and_all_code_cells_compile():
    notebook_path = Path("colab/hair_colab_enterprise.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 8
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_colab_config_has_approved_box_first_defaults():
    config = yoloe_autolabel.load_config(Path("colab/config.yaml"))
    assert config["yoloe"] == {"model": "yoloe-26l-seg.pt", "imgsz": 1280, "device": 0}
    assert config["localization"] == {
        "tile_size": 1024, "overlap": 0.20,
        "prompts": [
            "retail personal care product", "hair care product", "shampoo bottle",
            "conditioner bottle", "cosmetic bottle", "pump bottle", "aerosol can",
            "squeeze tube", "hair treatment pouch", "product sachet", "cosmetic jar",
            "boxed hair product", "hair care multipack",
        ],
        "conf": 0.05, "iou": 0.50, "min_side": 12, "max_area_ratio": 0.10,
        "min_aspect_ratio": 0.15, "max_aspect_ratio": 4.0, "nms_iou": 0.50,
        "recall_retry": {
            "enabled": True, "min_candidates_per_megapixel": 10.0,
            "tile_size": 640, "overlap": 0.30, "conf": 0.02, "iou": 0.60,
            "min_side": 6, "max_area_ratio": 0.20, "min_aspect_ratio": 0.08,
            "max_aspect_ratio": 8.0, "nms_iou": 0.70,
        },
    }
    assert config["matching"] == {
        "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k",
        "visual_weight": 0.80, "text_weight": 0.20, "min_score": 0.24,
        "min_margin": 0.02, "max_reference_views": 3, "needs_review_class_id": 89,
        "text_template": "a retail hair-care product package of {english_sku_name}",
    }
    assert config["export"] == {"train_fraction": 0.90, "split_seed": "hair-osa-v1"}
    assert config["pilot"] == {"enabled": False, "class_ids": [], "max_skus": 89, "max_images": 10}


def test_pilot_limits_images_but_not_skus(tmp_path):
    project = tmp_path / "hair_colab"
    project.mkdir()
    shutil.copy2(Path("colab/config.yaml"), project / "config.yaml")
    pilot_path = colab_runtime.write_pilot_config(project, max_images=10)
    pilot = yoloe_autolabel.load_config(pilot_path)
    assert pilot["pilot"] == {"enabled": True, "class_ids": [], "max_skus": 89, "max_images": 10}
    assert pilot["output"]["root"] == "pilot_output"


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


def test_notebook_packages_both_results_beside_workspace(tmp_path):
    notebook = json.loads(Path("colab/hair_colab_enterprise.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "package_results" in "".join(cell["source"]))
    workspace = tmp_path / "workspace"
    project = workspace / "hair_colab"
    observed = []

    def package_results(actual_project, platform, review):
        observed.append((actual_project, platform, review))
        return platform, review

    namespace = {"Path": Path, "workspace": workspace, "project": project,
                 "colab_runtime": types.SimpleNamespace(package_results=package_results)}
    exec("".join(cell["source"]), namespace)
    assert observed == [(project, tmp_path / "hair_ultralytics_platform.zip", tmp_path / "hair_annotation_review.zip")]


def test_notebook_pilot_generates_all_sku_config_and_preserves_validation_gate(tmp_path):
    notebook = json.loads(Path("colab/hair_colab_enterprise.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "write_pilot_config" in "".join(cell["source"]))
    project = tmp_path / "hair_colab"
    project.mkdir()
    shutil.copy2(Path("colab/config.yaml"), project / "config.yaml")
    observed = []
    namespace = {"project": project, "colab_runtime": colab_runtime, "sys": sys,
                 "subprocess": types.SimpleNamespace(run=lambda command, **kwargs: observed.append((command, kwargs)))}
    exec("".join(cell["source"]), namespace)
    pilot = yoloe_autolabel.load_config(project / "pilot_config.yaml")
    assert pilot["pilot"] == {"enabled": True, "class_ids": [], "max_skus": 89, "max_images": 10}
    assert observed == [([sys.executable, "yoloe_autolabel.py", "--config", "pilot_config.yaml", "--require-enabled-count", "89"], {"cwd": project, "check": True})]


@pytest.mark.parametrize("candidate_count, needs_review_count, expected_ratio", [(0, 0, None), (4, 1, 0.25)])
def test_notebook_pilot_diagnostics_report_zero_images_and_review_ratio(tmp_path, monkeypatch, candidate_count, needs_review_count, expected_ratio):
    notebook = json.loads(Path("colab/hair_colab_enterprise.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "pilot_run" in "".join(cell["source"]))
    project = tmp_path / "hair_colab"
    raw = project / "pilot_output" / "raw_predictions"
    previews = raw / "previews"
    previews.mkdir(parents=True)
    payload = {"totals": {"candidate_count": candidate_count, "needs_review_count": needs_review_count,
                          "geometry_rejections": {"too_small": 2}, "duplicates_removed": 3},
               "images": {"empty": {"candidate_count": 0}, "other": {"candidate_count": candidate_count}},
               "score_summaries": {"top1": {"count": candidate_count}}}
    (raw / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    displayed = []
    monkeypatch.setitem(sys.modules, "IPython.display", types.SimpleNamespace(display=displayed.append))
    from PIL import Image
    if candidate_count:
        for index in range(3):
            Image.new("RGB", (10, 20)).save(previews / f"{index}.jpg")
    namespace = {"project": project}
    exec("".join(cell["source"]), namespace)
    assert namespace["pilot_diagnostics"] == {
        "geometry_rejections": {"too_small": 2}, "duplicates_removed": 3,
        "zero_candidate_images": 2 if candidate_count == 0 else 1,
        "needs_review_ratio": expected_ratio, "top_score_distributions": {"top1": {"count": candidate_count}},
    }
    assert [image.size for image in displayed] == ([(1600, 1000)] if candidate_count else [])

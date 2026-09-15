import json
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
        "max_skus": 5,
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

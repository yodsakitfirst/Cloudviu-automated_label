import json
import zipfile
from pathlib import Path

import pytest
import yaml

import colab_runtime
import yoloe_autolabel


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


def test_package_results_archives_only_raw_predictions(tmp_path):
    project = tmp_path / "hair_colab"
    labels = project / "output" / "raw_predictions" / "labels"
    labels.mkdir(parents=True)
    (labels / "shelf.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    destination = tmp_path / "hair_label_results.zip"

    actual = colab_runtime.package_results(project, destination)

    assert actual == destination.resolve()
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["raw_predictions/labels/shelf.txt"]


def test_package_results_refuses_missing_or_existing_destinations(tmp_path):
    project = tmp_path / "hair_colab"
    with pytest.raises(ValueError, match="No full-run raw predictions"):
        colab_runtime.package_results(project, tmp_path / "results.zip")

    labels = project / "output" / "raw_predictions" / "labels"
    labels.mkdir(parents=True)
    (labels / "shelf.txt").write_text("", encoding="utf-8")
    destination = tmp_path / "results.zip"
    destination.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError):
        colab_runtime.package_results(project, destination)
    assert destination.read_bytes() == b"sentinel"


def test_notebook_is_valid_json_and_all_code_cells_compile():
    notebook_path = Path("colab/hair_colab_enterprise.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 8
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")

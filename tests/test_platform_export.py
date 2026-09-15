import csv
import hashlib
import os
import zipfile
from pathlib import Path

import pytest
import yaml

from hair_annotation.config import ExportConfig
from hair_annotation.export import assign_splits, package_platform_dataset, package_review_bundle


def export_config():
    return ExportConfig(train_fraction=0.90, split_seed="hair-osa-v1")


@pytest.fixture
def platform_project(tmp_path):
    project = tmp_path / "hair_colab"
    images = project / "shelf_images"
    raw = project / "output" / "raw_predictions"
    labels = raw / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (raw / "metadata").mkdir()
    (raw / "previews").mkdir()
    for index in range(10):
        (images / f"shelf_{index}.jpg").write_bytes(f"original-{index}".encode("ascii"))
        (labels / f"shelf_{index}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    (raw / "metadata" / "shelf_0.json").write_text("{}", encoding="utf-8")
    (raw / "previews" / "shelf_0.jpg").write_bytes(b"preview")
    for name in ("run.json", "provenance.json", "reference_diagnostics.json"):
        (raw / name).write_text("{}", encoding="utf-8")
    for name in ("summary.csv", "review_queue.csv"):
        (raw / name).write_text("image_name\nshelf_0.jpg\n", encoding="utf-8")
    with (project / "sku_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "barcode", "brand", "sku_name", "sku_name_th", "enabled"])
        writer.writeheader()
        for class_id in range(89):
            writer.writerow({"class_id": class_id, "barcode": str(100000 + class_id), "brand": "Brand", "sku_name": "Dove Blue 450 ml" if class_id == 0 else f"English SKU {class_id}", "sku_name_th": f"Thai SKU {class_id}", "enabled": "true"})
    return project


def test_split_is_deterministic_disjoint_and_exact_for_ten_images():
    names = [f"shelf_{index}.jpg" for index in range(10)]
    first = assign_splits(names, 0.90, "hair-osa-v1")
    assert first == assign_splits(list(reversed(names)), 0.90, "hair-osa-v1")
    assert list(first.values()).count("train") == 9
    assert list(first.values()).count("val") == 1
    ordered = sorted(names, key=lambda name: (hashlib.sha256(f"hair-osa-v1:{name}".encode()).hexdigest(), name))
    assert first[ordered[0]] == "val"
    assert all(first[name] == "train" for name in ordered[1:])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf"), True])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        assign_splits(["a.jpg", "b.jpg"], fraction, "seed")


def test_split_unique_names_and_nonempty_splits():
    assert assign_splits(["a.jpg", "b.jpg", "a.jpg"], 0.99, "seed") == {"a.jpg": "train", "b.jpg": "val"}


def test_platform_archive_contains_only_standard_yolo_members(platform_project, tmp_path):
    destination = tmp_path / "platform.zip"
    assert package_platform_dataset(platform_project, destination, export_config()) == destination.resolve()
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert len(names) == 21
        assert all(name == "data.yaml" or name.startswith(("images/train/", "images/val/", "labels/train/", "labels/val/")) for name in names)
        for index in range(10):
            image = next(name for name in names if name.endswith(f"/shelf_{index}.jpg"))
            label = image.replace("images/", "labels/").replace(".jpg", ".txt")
            assert label in names
            assert archive.read(image) == f"original-{index}".encode("ascii")
        data = yaml.safe_load(archive.read("data.yaml"))
    assert data == {"path": ".", "train": "images/train", "val": "images/val", "names": {**{i: "Dove Blue 450 ml" if i == 0 else f"English SKU {i}" for i in range(89)}, 89: "Needs Review"}}


@pytest.mark.parametrize("row", ["90 0.5 0.5 0.1 0.2", "-1 0.5 0.5 0.1 0.2", "1.5 0.5 0.5 0.1 0.2", "0 nan 0.5 0.1 0.2", "0 0.5 inf 0.1 0.2", "0 -0.1 0.5 0.1 0.2", "0 0.5 1.1 0.1 0.2", "0 0.5 0.5 0.1", "0 0.5 0.5 0.1 0.2 extra", "0 words 0.5 0.1 0.2"])
def test_platform_preflight_rejects_invalid_yolo_rows(platform_project, tmp_path, row):
    (platform_project / "output/raw_predictions/labels/shelf_0.txt").write_text(row, encoding="utf-8")
    destination = tmp_path / "absent" / "bad.zip"
    with pytest.raises(ValueError):
        package_platform_dataset(platform_project, destination, export_config())
    assert not destination.parent.exists()


def test_platform_accepts_empty_labels_and_needs_review(platform_project, tmp_path):
    labels = platform_project / "output/raw_predictions/labels"
    (labels / "shelf_0.txt").write_text("", encoding="utf-8")
    (labels / "shelf_1.txt").write_text("89 0 1 0 1\n", encoding="utf-8")
    destination = package_platform_dataset(platform_project, tmp_path / "ok.zip", export_config())
    with zipfile.ZipFile(destination) as archive:
        assert archive.read(next(n for n in archive.namelist() if n.endswith("/shelf_0.txt"))) == b""


@pytest.mark.parametrize("missing", ["shelf_images/shelf_0.jpg", "output/raw_predictions/labels/shelf_0.txt"])
def test_platform_requires_exact_image_label_pairs(platform_project, tmp_path, missing):
    (platform_project / missing).unlink()
    with pytest.raises(ValueError, match="pair|label|image"):
        package_platform_dataset(platform_project, tmp_path / "bad.zip", export_config())


def test_platform_rejects_duplicate_case_insensitive_stems(platform_project, tmp_path):
    (platform_project / "shelf_images/SHELF_0.png").write_bytes(b"other")
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        package_platform_dataset(platform_project, tmp_path / "bad.zip", export_config())


def test_platform_rejects_fewer_than_two_images(platform_project, tmp_path):
    for index in range(1, 10):
        (platform_project / f"shelf_images/shelf_{index}.jpg").unlink()
        (platform_project / f"output/raw_predictions/labels/shelf_{index}.txt").unlink()
    with pytest.raises(ValueError, match="two|2"):
        package_platform_dataset(platform_project, tmp_path / "bad.zip", export_config())


@pytest.mark.parametrize("mutation", ["gap", "reserved", "duplicate"])
def test_platform_rejects_nonpermanent_manifest(platform_project, tmp_path, mutation):
    manifest = platform_project / "sku_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if mutation == "gap":
        rows.pop(20)
    else:
        rows[-1]["class_id"] = "89" if mutation == "reserved" else "0"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        package_platform_dataset(platform_project, tmp_path / "bad.zip", export_config())


def test_review_archive_exact_contents_and_machine_review_warning(platform_project, tmp_path):
    destination = package_review_bundle(platform_project, tmp_path / "review.zip")
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"sku_manifest.csv", "review_queue.csv", "metadata/", "metadata/shelf_0.json", "previews/", "previews/shelf_0.jpg", "summary.csv", "run.json", "provenance.json", "reference_diagnostics.json", "README.txt"}
        assert archive.read("sku_manifest.csv") == (platform_project / "sku_manifest.csv").read_bytes()
        readme = archive.read("README.txt").decode()
    assert "machine suggestions" in readme and "human review" in readme
    assert "Needs Review" in readme and "reassigned" in readme and "89" in readme and "deleted" in readme


@pytest.mark.parametrize("artifact", ["review_queue.csv", "metadata", "previews", "summary.csv", "run.json", "provenance.json", "reference_diagnostics.json"])
def test_review_requires_every_artifact(platform_project, tmp_path, artifact):
    path = platform_project / "output/raw_predictions" / artifact
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(ValueError):
        package_review_bundle(platform_project, tmp_path / "bad.zip")


@pytest.mark.parametrize("kind", ["platform", "review"])
def test_packagers_require_explicit_overwrite(platform_project, tmp_path, kind):
    destination = tmp_path / "exists.zip"
    destination.write_bytes(b"sentinel")
    def package(**kwargs):
        if kind == "platform":
            return package_platform_dataset(platform_project, destination, export_config(), **kwargs)
        return package_review_bundle(platform_project, destination, **kwargs)
    with pytest.raises(FileExistsError):
        package()
    assert destination.read_bytes() == b"sentinel"
    package(overwrite=True)
    assert zipfile.is_zipfile(destination)


@pytest.mark.parametrize("failure", ["write", "integrity"])
@pytest.mark.parametrize("kind", ["platform", "review"])
def test_archive_failure_is_atomic_and_cleans_temporary_file(platform_project, tmp_path, monkeypatch, kind, failure):
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"sentinel")
    def fail(*args, **kwargs):
        raise OSError("injected ZIP failure")
    monkeypatch.setattr(zipfile.ZipFile, "write" if failure == "write" else "testzip", fail)
    with pytest.raises(OSError, match="injected"):
        if kind == "platform":
            package_platform_dataset(platform_project, destination, export_config(), overwrite=True)
        else:
            package_review_bundle(platform_project, destination, overwrite=True)
    assert destination.read_bytes() == b"sentinel"
    assert not list(tmp_path.glob(".existing*"))


@pytest.mark.parametrize("kind", ["platform", "review"])
def test_packagers_never_write_reviewed_labels_or_input_alias(platform_project, tmp_path, kind):
    source = platform_project / "sku_manifest.csv"
    sentinel = source.read_bytes()
    alias = tmp_path / "alias.zip"
    os.link(source, alias)
    for destination in (source, alias, tmp_path / "reviewed_labels" / "unsafe.zip"):
        with pytest.raises(ValueError):
            if kind == "platform":
                package_platform_dataset(platform_project, destination, export_config(), overwrite=True)
            else:
                package_review_bundle(platform_project, destination, overwrite=True)
    assert source.read_bytes() == sentinel


def test_review_packager_preserves_original_image_hardlink_alias(platform_project, tmp_path):
    source = platform_project / "shelf_images/shelf_0.jpg"
    destination = tmp_path / "image_alias.zip"
    os.link(source, destination)
    with pytest.raises(ValueError, match="alias"):
        package_review_bundle(platform_project, destination, overwrite=True)
    assert source.read_bytes() == b"original-0"


def test_packagers_use_configured_paths(platform_project, tmp_path):
    (platform_project / "shelf_images").rename(platform_project / "configured_images")
    (platform_project / "sku_manifest.csv").rename(platform_project / "configured_manifest.csv")
    (platform_project / "output").rename(platform_project / "configured_output")
    (platform_project / "config.yaml").write_text(yaml.safe_dump({"dataset": {"sku_manifest": "configured_manifest.csv", "shelf_images": "configured_images", "image_extensions": [".jpg"]}, "output": {"root": "configured_output"}, "export": {"train_fraction": 0.9, "split_seed": "hair-osa-v1"}}), encoding="utf-8")
    assert package_platform_dataset(platform_project, tmp_path / "configured.zip", export_config()).exists()
    assert package_review_bundle(platform_project, tmp_path / "review.zip").exists()


@pytest.mark.parametrize("kind", ["platform", "review"])
def test_packagers_reject_symlink_destination(platform_project, tmp_path, kind):
    target = tmp_path / "target.zip"
    target.write_bytes(b"sentinel")
    link = tmp_path / "link.zip"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink|junction|redirect"):
        if kind == "platform":
            package_platform_dataset(platform_project, link, export_config(), overwrite=True)
        else:
            package_review_bundle(platform_project, link, overwrite=True)
    assert target.read_bytes() == b"sentinel"


def configure_reference_inputs(project, reference_entries):
    # Deliberately nest the definition YAML: CLI paths remain project-relative.
    definitions = project / "definitions" / "references.yaml"
    definitions.parent.mkdir(exist_ok=True)
    definitions.write_text(yaml.safe_dump({"references": reference_entries}), encoding="utf-8")
    config = {"dataset": {"sku_manifest": "sku_manifest.csv", "shelf_images": "shelf_images", "references": "definitions/references.yaml"}, "output": {"root": "output"}, "export": {"train_fraction": 0.9, "split_seed": "hair-osa-v1"}}
    (project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.parametrize("kind", ["platform", "review"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_packagers_protect_all_mapped_reference_images(platform_project, tmp_path, kind, hardlink):
    reference = tmp_path / "external_products" / "product.jpg"
    reference.parent.mkdir()
    reference.write_bytes(b"original-product-reference")
    relative_reference = Path(os.path.relpath(reference, platform_project)).as_posix()
    # Include a disabled permanent SKU: safety must not filter mapped files.
    manifest = platform_project / "sku_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[-1]["enabled"] = "false"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    configure_reference_inputs(platform_project, [{"class_id": 88, "image_path": relative_reference}])
    destination = reference
    if hardlink:
        destination = tmp_path / "reference_alias.zip"
        os.link(reference, destination)
    with pytest.raises(ValueError, match="alias"):
        if kind == "platform":
            package_platform_dataset(platform_project, destination, export_config(), overwrite=True)
        else:
            package_review_bundle(platform_project, destination, overwrite=True)
    assert reference.read_bytes() == b"original-product-reference"


@pytest.mark.parametrize("kind", ["platform", "review"])
def test_packagers_reject_lexical_reviewed_labels_reference_mapping(platform_project, tmp_path, kind):
    reviewed = platform_project / "reviewed_labels" / "product.jpg"
    reviewed.parent.mkdir()
    reviewed.write_bytes(b"human-source")
    configure_reference_inputs(platform_project, [{"class_id": 0, "image_path": "reviewed_labels/product.jpg"}])
    destination = tmp_path / "absent" / "archive.zip"
    with pytest.raises(ValueError, match="reviewed_labels"):
        if kind == "platform":
            package_platform_dataset(platform_project, destination, export_config())
        else:
            package_review_bundle(platform_project, destination)
    assert not destination.parent.exists()
    assert reviewed.read_bytes() == b"human-source"

import csv
import io
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from openpyxl import Workbook

import prepare_hair_colab as prep


def write_workbook(path: Path, rows: list[tuple[object, object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    sheet.append([None, None])
    sheet.append(["Barcode", "Product Name"])
    for barcode, name in rows:
        sheet.append([barcode, name])
    workbook.save(path)


def write_image(path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    path.write_bytes(encoded.tobytes())


def test_source_decoder_supports_native_thai_path(tmp_path):
    path = tmp_path / "สินค้า หนึ่ง.png"
    write_image(path)
    assert [entry.name for entry in tmp_path.iterdir()] == ["สินค้า หนึ่ง.png"]
    before = path.read_bytes()
    prep._validate_decodable_image(path, "Product reference")
    assert path.read_bytes() == before


@pytest.fixture
def package_fixture(tmp_path):
    workbook = tmp_path / "sku.xlsx"
    write_workbook(
        workbook,
        [("8851932487177", "สินค้า หนึ่ง"), ("8851932487178", "สินค้า สอง")],
    )
    products = tmp_path / "products"
    products.mkdir()
    write_image(products / "สินค้า หนึ่ง.png")
    write_image(products / "สินค้า สอง.jpg")
    shelves = tmp_path / "shelves"
    shelves.mkdir()
    write_image(shelves / "a.jpg")
    write_image(shelves / "b.JPG")
    (shelves / "index.csv").write_text("ignored", encoding="utf-8")
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("overrides: {}\n", encoding="utf-8")
    translations = tmp_path / "translations.yaml"
    translations.write_text(
        "translations:\n"
        "  '8851932487177': Product One\n"
        "  '8851932487178': Product Two\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "tests").mkdir(parents=True)
    (repo_root / "colab").mkdir()
    (repo_root / "yoloe_autolabel.py").write_bytes(b"engine-bytes")
    (repo_root / "requirements.txt").write_text("dependency==1\n", encoding="utf-8")
    (repo_root / "colab_runtime.py").write_text(
        "def package_results(): pass\n", encoding="utf-8"
    )
    (repo_root / "tests" / "test_yoloe_autolabel.py").write_text(
        "def test_packaged_engine(): pass\n", encoding="utf-8"
    )
    (repo_root / "colab" / "config.yaml").write_text("project: {}\n", encoding="utf-8")
    (repo_root / "colab" / "hair_colab_enterprise.ipynb").write_text(
        '{"cells": []}\n', encoding="utf-8"
    )
    real_repo = Path(__file__).resolve().parents[1]
    (repo_root / "hair_annotation").mkdir()
    for name in ("__init__.py", "config.py", "types.py", "localization.py", "matching.py", "export.py"):
        shutil.copy2(real_repo / "hair_annotation" / name, repo_root / "hair_annotation" / name)
    for name in ("test_box_first_config.py", "test_localization.py", "test_matching.py", "test_platform_export.py"):
        shutil.copy2(real_repo / "tests" / name, repo_root / "tests" / name)
    return prep.PackageInputs(
        workbook=workbook,
        product_images=products,
        shelf_images=shelves,
        repo_root=repo_root,
        overrides=overrides,
        translations=translations,
        expected_skus=2,
        expected_shelves=2,
    )


def test_load_sheet2_skus_preserves_row_order_barcodes_and_thai_names(tmp_path):
    workbook = tmp_path / "sku.xlsx"
    write_workbook(
        workbook,
        [
            ("08851932487177", "สินค้า หนึ่ง"),
            ("18851932355343", "สินค้า สอง"),
        ],
    )

    skus = prep.load_sheet2_skus(workbook, expected_count=2)

    assert skus == [
        prep.HairSku(0, "08851932487177", "สินค้า หนึ่ง"),
        prep.HairSku(1, "18851932355343", "สินค้า สอง"),
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [("8851932487177", "A"), ("8851932487177", "B")],
        [("8851932487177", "A"), ("", "B")],
        [("885193248717X", "A")],
        [("123", "A")],
        [("8851932487177", "A"), ("8851932487178", "A")],
    ],
)
def test_load_sheet2_skus_rejects_invalid_identifiers_and_duplicates(tmp_path, rows):
    workbook = tmp_path / "invalid.xlsx"
    write_workbook(workbook, rows)

    with pytest.raises(ValueError):
        prep.load_sheet2_skus(workbook, expected_count=len(rows))


def test_load_sheet2_skus_rejects_an_unexpected_row_count(tmp_path):
    workbook = tmp_path / "sku.xlsx"
    write_workbook(workbook, [("8851932487177", "A")])

    with pytest.raises(ValueError, match="Expected 2 SKUs, found 1"):
        prep.load_sheet2_skus(workbook, expected_count=2)


def test_normalize_product_name_is_unicode_and_whitespace_stable():
    assert prep.normalize_product_name("  สิ\u0e19ค้า\tหนึ่ง  ") == prep.normalize_product_name(
        "สินค้า หนึ่ง"
    )


def test_match_product_references_uses_exact_names_and_explicit_override(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    exact = products / "สินค้า หนึ่ง.png"
    override = products / "สินค้า-สอง.jpg"
    exact.write_bytes(b"exact")
    override.write_bytes(b"override")
    skus = [
        prep.HairSku(0, "8851932487177", "สินค้า หนึ่ง"),
        prep.HairSku(1, "8851932487178", "สินค้า/สอง"),
    ]

    matched = prep.match_product_references(
        skus,
        products,
        {"สินค้า/สอง": "สินค้า-สอง.jpg"},
    )

    assert matched == {0: exact.resolve(), 1: override.resolve()}
    assert prep.ascii_reference_name(skus[0], exact) == "class_000_8851932487177.png"


def test_match_product_references_rejects_ambiguous_names(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    (products / "A.jpg").write_bytes(b"a")
    (products / "a.png").write_bytes(b"b")

    with pytest.raises(ValueError, match="Ambiguous"):
        prep.match_product_references(
            [prep.HairSku(0, "8851932487177", "A")], products, {}
        )


def test_match_product_references_rejects_missing_and_unused_overrides(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    (products / "A.jpg").write_bytes(b"a")

    with pytest.raises(ValueError, match="Missing reference"):
        prep.match_product_references(
            [prep.HairSku(0, "8851932487177", "B")], products, {}
        )
    with pytest.raises(ValueError, match="Unused override"):
        prep.match_product_references(
            [prep.HairSku(0, "8851932487177", "A")],
            products,
            {"X": "A.jpg"},
        )


def test_load_reference_overrides_requires_a_string_mapping(tmp_path):
    valid = tmp_path / "valid.yaml"
    valid.write_text('overrides:\n  "A/B": "A-B.png"\n', encoding="utf-8")
    assert prep.load_reference_overrides(valid) == {"A/B": "A-B.png"}

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("overrides:\n  A: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="string product names"):
        prep.load_reference_overrides(invalid)


def test_load_sku_translations_requires_an_exact_ascii_barcode_map(tmp_path):
    skus = [
        prep.HairSku(0, "8851932487177", "สินค้า หนึ่ง"),
        prep.HairSku(1, "8851932487178", "สินค้า สอง"),
    ]
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "translations:\n"
        "  '8851932487177': Product One\n"
        "  '8851932487178': Product Two\n",
        encoding="utf-8",
    )
    assert prep.load_sku_translations(valid, skus) == {
        "8851932487177": "Product One",
        "8851932487178": "Product Two",
    }

    missing = tmp_path / "missing.yaml"
    missing.write_text(
        "translations:\n  '8851932487177': Product One\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing barcodes"):
        prep.load_sku_translations(missing, skus)

    thai = tmp_path / "thai.yaml"
    thai.write_text(
        "translations:\n"
        "  '8851932487177': สินค้า One\n"
        "  '8851932487178': Product Two\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ASCII"):
        prep.load_sku_translations(thai, skus)


def test_discover_shelf_images_selects_only_unique_decodable_jpg_files(tmp_path):
    shelves = tmp_path / "shelves"
    shelves.mkdir()
    write_image(shelves / "a.jpg")
    write_image(shelves / "b.JPG")
    (shelves / "index.csv").write_text("ignored", encoding="utf-8")

    assert [
        path.name for path in prep.discover_shelf_images(shelves, expected_count=2)
    ] == ["a.jpg", "b.JPG"]


def test_build_runtime_package_copies_bytes_and_uses_ascii_archive_paths(
    package_fixture, tmp_path
):
    destination = tmp_path / "hair_colab_runtime.zip"

    report = prep.build_runtime_package(package_fixture, destination)

    assert report == prep.PackageReport(destination.resolve(), 2, 2, 2)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert all(name.isascii() for name in names)
        assert len([name for name in names if name.startswith("hair_colab/references/")]) == 2
        assert len([name for name in names if name.startswith("hair_colab/shelf_images/")]) == 2
        for source in (package_fixture.shelf_images / "a.jpg", package_fixture.shelf_images / "b.JPG"):
            assert archive.read(f"hair_colab/shelf_images/{source.name}") == source.read_bytes()
        assert archive.read("hair_colab/yoloe_autolabel.py") == b"engine-bytes"
        assert archive.read("hair_colab/colab_runtime.py") == (package_fixture.repo_root / "colab_runtime.py").read_bytes()
        assert (
            archive.read("hair_colab/references/class_000_8851932487177.png")
            == (package_fixture.product_images / "สินค้า หนึ่ง.png").read_bytes()
        )
        manifest = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("hair_colab/sku_manifest.csv").decode("utf-8")
                )
            )
        )
        assert manifest == [
            {
                "class_id": "0",
                "barcode": "8851932487177",
                "brand": "Unspecified",
                "sku_name": "Product One",
                "sku_name_th": "สินค้า หนึ่ง",
                "enabled": "true",
            },
            {
                "class_id": "1",
                "barcode": "8851932487178",
                "brand": "Unspecified",
                "sku_name": "Product Two",
                "sku_name_th": "สินค้า สอง",
                "enabled": "true",
            },
        ]


def test_build_runtime_package_refuses_to_replace_an_archive(package_fixture, tmp_path):
    destination = tmp_path / "hair_colab_runtime.zip"
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        prep.build_runtime_package(package_fixture, destination)

    assert destination.read_bytes() == b"sentinel"


def test_discover_shelf_images_rejects_corrupt_duplicate_and_non_ascii_inputs(tmp_path):
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "a.jpg").write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="cannot be decoded"):
        prep.discover_shelf_images(corrupt, expected_count=1)

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    write_image(duplicate / "A.jpg")
    write_image(duplicate / "a.jpeg")
    with pytest.raises(ValueError, match="Duplicate shelf image stem"):
        prep.discover_shelf_images(duplicate, expected_count=2)

    non_ascii = tmp_path / "non-ascii"
    non_ascii.mkdir()
    write_image(non_ascii / "ชั้น.jpg")
    with pytest.raises(ValueError, match="not ASCII"):
        prep.discover_shelf_images(non_ascii, expected_count=1)


def test_failed_package_build_leaves_no_final_archive(package_fixture, tmp_path):
    destination = tmp_path / "hair_colab_runtime.zip"
    (package_fixture.repo_root / "colab" / "config.yaml").unlink()

    with pytest.raises(ValueError, match="Required package input is missing"):
        prep.build_runtime_package(package_fixture, destination)

    assert not destination.exists()


def test_cli_metadata_output_contains_only_registry_files(package_fixture, tmp_path):
    metadata = tmp_path / "generated"
    destination = tmp_path / "hair_colab_runtime.zip"

    result = prep.main(
        [
            "--workbook",
            str(package_fixture.workbook),
            "--product-images",
            str(package_fixture.product_images),
            "--shelf-images",
            str(package_fixture.shelf_images),
            "--repo-root",
            str(package_fixture.repo_root),
            "--overrides",
            str(package_fixture.overrides),
            "--translations",
            str(package_fixture.translations),
            "--output",
            str(destination),
            "--metadata-output",
            str(metadata),
            "--expected-skus",
            "2",
            "--expected-shelves",
            "2",
        ]
    )

    assert result == 0
    assert sorted(path.name for path in metadata.iterdir()) == [
        "reference_prompts.yaml",
        "sku_manifest.csv",
    ]
    with zipfile.ZipFile(destination) as archive:
        assert (metadata / "sku_manifest.csv").read_bytes() == archive.read(
            "hair_colab/sku_manifest.csv"
        )
        assert (metadata / "reference_prompts.yaml").read_bytes() == archive.read(
            "hair_colab/reference_prompts.yaml"
        )


def test_runtime_archive_contains_box_first_modules_and_tests(package_fixture, tmp_path):
    destination = tmp_path / "runtime.zip"
    prep.build_runtime_package(package_fixture, destination)
    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
        for name in ("__init__.py", "config.py", "types.py", "localization.py", "matching.py", "export.py"):
            assert f"hair_colab/hair_annotation/{name}" in names
        for name in ("test_box_first_config.py", "test_localization.py", "test_matching.py", "test_platform_export.py"):
            assert f"hair_colab/tests/{name}" in names
        assert not any(part in {".git", ".DS_Store", "__pycache__", ".venv"} for name in names for part in Path(name).parts)


@pytest.mark.parametrize("field", ["workbook", "overrides", "translations", "product_images", "shelf_images", "repo_root"])
def test_builder_rejects_reviewed_labels_inputs_before_reading(package_fixture, tmp_path, field, monkeypatch):
    forbidden = tmp_path / "reviewed_labels" / "missing"
    invalid = replace(package_fixture, **{field: forbidden})
    monkeypatch.setattr(prep, "load_sheet2_skus", lambda *args: pytest.fail("must reject before reading workbook"))
    with pytest.raises(ValueError, match="reviewed_labels"):
        prep.build_runtime_package(invalid, tmp_path / "runtime.zip")


@pytest.mark.parametrize("source", ["workbook", "overrides", "translations"])
def test_builder_refuses_input_overwrite_even_with_overwrite(package_fixture, source):
    destination = getattr(package_fixture, source)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="alias|input"):
        prep.build_runtime_package(package_fixture, destination, overwrite=True)
    assert destination.read_bytes() == before


def test_builder_refuses_hardlink_alias_of_input(package_fixture, tmp_path):
    destination = tmp_path / "runtime.zip"
    os.link(package_fixture.workbook, destination)
    before = package_fixture.workbook.read_bytes()
    with pytest.raises(ValueError, match="alias|input"):
        prep.build_runtime_package(package_fixture, destination, overwrite=True)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("directory", ["product_images", "shelf_images"])
def test_builder_refuses_archive_inside_image_sources(package_fixture, directory):
    destination = getattr(package_fixture, directory) / "runtime.zip"
    with pytest.raises(ValueError, match="input|source"):
        prep.build_runtime_package(package_fixture, destination)
    assert not destination.exists()


def test_builder_rejects_reviewed_labels_output(package_fixture, tmp_path):
    destination = tmp_path / "Reviewed_Labels" / "runtime.zip"
    with pytest.raises(ValueError, match="reviewed_labels"):
        prep.build_runtime_package(package_fixture, destination)
    assert not destination.parent.exists()


def test_metadata_export_refuses_replacing_directory_containing_archive(package_fixture, tmp_path):
    destination = tmp_path / "runtime.zip"
    prep.build_runtime_package(package_fixture, destination)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="input|source|archive"):
        prep.export_registry_metadata(destination, tmp_path, overwrite=True)
    assert destination.read_bytes() == before


def test_metadata_export_rejects_reviewed_labels_before_archive_read(tmp_path, monkeypatch):
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args: pytest.fail("must not open reviewed_labels"))
    with pytest.raises(ValueError, match="reviewed_labels"):
        prep.export_registry_metadata(tmp_path / "reviewed_labels" / "missing.zip", tmp_path / "metadata")
    assert not (tmp_path / "metadata").exists()


@pytest.mark.parametrize("field", ["product_images", "shelf_images"])
def test_cli_rejects_metadata_in_image_sources_before_build(package_fixture, tmp_path, field):
    metadata = getattr(package_fixture, field) / "generated"
    archive = tmp_path / "runtime.zip"
    result = prep.main([
        "--workbook", str(package_fixture.workbook),
        "--product-images", str(package_fixture.product_images),
        "--shelf-images", str(package_fixture.shelf_images),
        "--repo-root", str(package_fixture.repo_root),
        "--overrides", str(package_fixture.overrides),
        "--translations", str(package_fixture.translations),
        "--expected-skus", "2", "--expected-shelves", "2",
        "--output", str(archive), "--metadata-output", str(metadata), "--overwrite",
    ])
    assert result == 2
    assert not metadata.exists()
    assert not archive.exists()


def test_cli_refuses_metadata_replacing_repository_source(package_fixture, tmp_path):
    archive = tmp_path / "runtime.zip"
    before = (package_fixture.repo_root / "yoloe_autolabel.py").read_bytes()
    result = prep.main([
        "--workbook", str(package_fixture.workbook),
        "--product-images", str(package_fixture.product_images),
        "--shelf-images", str(package_fixture.shelf_images),
        "--repo-root", str(package_fixture.repo_root),
        "--overrides", str(package_fixture.overrides),
        "--translations", str(package_fixture.translations),
        "--expected-skus", "2", "--expected-shelves", "2",
        "--output", str(archive), "--metadata-output", str(package_fixture.repo_root), "--overwrite",
    ])
    assert result == 2
    assert not archive.exists()
    assert (package_fixture.repo_root / "yoloe_autolabel.py").read_bytes() == before


def test_runtime_extract_runs_real_model_free_tests_and_validation(package_fixture, tmp_path):
    # Synthetic development data, not a claim about the unavailable real data.
    inputs = replace(package_fixture, repo_root=Path(__file__).resolve().parents[1])
    archive_path = tmp_path / "runtime.zip"
    prep.build_runtime_package(inputs, archive_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    project = extracted / "hair_colab"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
    print(result.stdout.strip())
    validation = subprocess.run([
        sys.executable, "yoloe_autolabel.py", "--config", "config.yaml",
        "--validate-only", "--require-enabled-count", "2",
    ], cwd=project, text=True, capture_output=True)
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "Validation successful: 2 SKU(s), 2 image(s)" in validation.stdout
    print(validation.stdout.strip())
    # Running the extracted suite cannot silently import the repository engine.
    assert (project / "yoloe_autolabel.py").read_bytes() == (inputs.repo_root / "yoloe_autolabel.py").read_bytes()

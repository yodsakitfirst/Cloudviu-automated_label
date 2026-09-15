import csv
import io
import zipfile
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
    assert cv2.imwrite(str(path), image)


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
    return prep.PackageInputs(
        workbook=workbook,
        product_images=products,
        shelf_images=shelves,
        repo_root=repo_root,
        overrides=overrides,
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
        assert archive.read("hair_colab/yoloe_autolabel.py") == b"engine-bytes"
        assert archive.read("hair_colab/colab_runtime.py") == b"def package_results(): pass\n"
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
                "sku_name": "สินค้า หนึ่ง",
                "enabled": "true",
            },
            {
                "class_id": "1",
                "barcode": "8851932487178",
                "brand": "Unspecified",
                "sku_name": "สินค้า สอง",
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

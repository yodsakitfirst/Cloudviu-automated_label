from pathlib import Path

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

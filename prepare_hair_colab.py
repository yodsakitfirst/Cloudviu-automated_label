"""Build a runtime-local Colab Enterprise package for the HAIR dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_PRODUCT_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_SHELF_EXTENSIONS = frozenset({".jpg", ".jpeg"})


@dataclass(frozen=True)
class HairSku:
    class_id: int
    barcode: str
    sku_name: str


@dataclass(frozen=True)
class PackageInputs:
    workbook: Path
    product_images: Path
    shelf_images: Path
    repo_root: Path
    overrides: Path
    expected_skus: int = 89
    expected_shelves: int = 632


@dataclass(frozen=True)
class PackageReport:
    archive_path: Path
    sku_count: int
    reference_count: int
    shelf_count: int


def normalize_product_name(value: str) -> str:
    """Normalize a product name only for exact, deterministic matching."""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _barcode_text(value: object, row_number: int) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Invalid barcode in Sheet2 row {row_number}")
    if isinstance(value, int):
        barcode = str(value)
    elif isinstance(value, float) and value.is_integer():
        barcode = str(int(value))
    else:
        barcode = str(value or "").strip()
    if not barcode.isascii() or not barcode.isdigit() or len(barcode) not in {13, 14}:
        raise ValueError(f"Invalid barcode in Sheet2 row {row_number}: {barcode!r}")
    return barcode


def load_sheet2_skus(path: Path, expected_count: int = 89) -> list[HairSku]:
    """Load the permanent class registry from Sheet2 rows 3 onward."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to read the SKU workbook") from exc

    workbook_path = Path(path).resolve()
    if not workbook_path.is_file():
        raise ValueError(f"Workbook does not exist: {workbook_path}")
    try:
        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot read workbook {workbook_path}: {exc}") from exc
    try:
        if "Sheet2" not in workbook.sheetnames:
            raise ValueError("Workbook is missing Sheet2")
        sheet = workbook["Sheet2"]
        headers = tuple(str(sheet.cell(2, column).value or "").strip() for column in (1, 2))
        if headers != ("Barcode", "Product Name"):
            raise ValueError(
                "Sheet2 row 2 must start with 'Barcode' and 'Product Name'"
            )

        source_rows: list[tuple[int, object, object]] = []
        for row_number in range(3, sheet.max_row + 1):
            barcode_value = sheet.cell(row_number, 1).value
            name_value = sheet.cell(row_number, 2).value
            if barcode_value is None and name_value is None:
                continue
            source_rows.append((row_number, barcode_value, name_value))
    finally:
        workbook.close()

    if len(source_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} SKUs, found {len(source_rows)}")

    skus: list[HairSku] = []
    barcodes: set[str] = set()
    names: set[str] = set()
    for class_id, (row_number, barcode_value, name_value) in enumerate(source_rows):
        barcode = _barcode_text(barcode_value, row_number)
        sku_name = unicodedata.normalize("NFC", str(name_value or "").strip())
        name_key = normalize_product_name(sku_name)
        if not sku_name:
            raise ValueError(f"Blank product name in Sheet2 row {row_number}")
        if barcode in barcodes:
            raise ValueError(f"Duplicate barcode in Sheet2 row {row_number}: {barcode}")
        if name_key in names:
            raise ValueError(f"Duplicate product name in Sheet2 row {row_number}: {sku_name}")
        barcodes.add(barcode)
        names.add(name_key)
        skus.append(HairSku(class_id, barcode, sku_name))
    return skus


def load_reference_overrides(path: Path) -> dict[str, str]:
    """Load explicit workbook-name to source-filename exceptions."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read reference overrides") from exc

    override_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read reference overrides {override_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("overrides"), dict):
        raise ValueError("Reference overrides must contain an 'overrides' mapping")
    overrides = payload["overrides"]
    if any(
        not isinstance(product_name, str)
        or not product_name.strip()
        or not isinstance(filename, str)
        or not filename.strip()
        for product_name, filename in overrides.items()
    ):
        raise ValueError("Reference overrides must map string product names to string filenames")
    return {product_name.strip(): filename.strip() for product_name, filename in overrides.items()}


def match_product_references(
    skus: Sequence[HairSku],
    product_dir: Path,
    overrides: Mapping[str, str],
) -> dict[int, Path]:
    """Match every SKU to exactly one reference without fuzzy matching."""
    root = Path(product_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Product image directory does not exist: {root}")
    images = sorted(
        (
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in _PRODUCT_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    by_name: dict[str, Path] = {}
    for image in images:
        key = normalize_product_name(image.stem)
        if key in by_name:
            raise ValueError(
                f"Ambiguous product reference names: {by_name[key].name} and {image.name}"
            )
        by_name[key] = image

    normalized_overrides: dict[str, tuple[str, str]] = {}
    for product_name, filename in overrides.items():
        key = normalize_product_name(product_name)
        if key in normalized_overrides:
            raise ValueError(f"Duplicate normalized override for product: {product_name}")
        normalized_overrides[key] = (product_name, filename)

    matched: dict[int, Path] = {}
    used_override_keys: set[str] = set()
    used_images: set[Path] = set()
    for sku in skus:
        name_key = normalize_product_name(sku.sku_name)
        if name_key in normalized_overrides:
            _, filename = normalized_overrides[name_key]
            candidate = (root / filename).resolve()
            if candidate.parent != root or not candidate.is_file():
                raise ValueError(
                    f"Reference override for {sku.sku_name!r} does not resolve to a file: {filename}"
                )
            if candidate.suffix.casefold() not in _PRODUCT_EXTENSIONS:
                raise ValueError(f"Unsupported product reference extension: {candidate.name}")
            used_override_keys.add(name_key)
        else:
            candidate = by_name.get(name_key)
            if candidate is None:
                raise ValueError(f"Missing reference for product: {sku.sku_name}")
        if candidate in used_images:
            raise ValueError(f"Product reference is assigned more than once: {candidate.name}")
        used_images.add(candidate)
        matched[sku.class_id] = candidate

    unused_overrides = sorted(set(normalized_overrides) - used_override_keys)
    if unused_overrides:
        original_names = [normalized_overrides[key][0] for key in unused_overrides]
        raise ValueError(f"Unused override product names: {original_names}")
    unused_images = [image.name for image in images if image not in used_images]
    if unused_images:
        raise ValueError(f"Unmatched product reference images: {unused_images}")
    return matched


def ascii_reference_name(sku: HairSku, source: Path) -> str:
    """Return the deterministic English-safe reference filename."""
    suffix = Path(source).suffix.casefold()
    if suffix not in _PRODUCT_EXTENSIONS:
        raise ValueError(f"Unsupported product reference extension: {source}")
    filename = f"class_{sku.class_id:03d}_{sku.barcode}{suffix}"
    if not filename.isascii():
        raise ValueError(f"Prepared reference name is not ASCII: {filename}")
    return filename


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to validate source images") from exc
    return cv2


def _validate_decodable_image(path: Path, label: str) -> None:
    cv2 = _load_cv2()
    image = cv2.imread(str(path))
    if image is None or image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{label} cannot be decoded: {path}")


def discover_shelf_images(path: Path, expected_count: int = 632) -> list[Path]:
    """Return the validated flat set of ASCII-named shelf JPGs."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Shelf image directory does not exist: {root}")
    images = sorted(
        (
            candidate.resolve()
            for candidate in root.iterdir()
            if candidate.is_file() and candidate.suffix.casefold() in _SHELF_EXTENSIONS
        ),
        key=lambda candidate: (candidate.name.casefold(), candidate.name),
    )
    if len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} shelf images, found {len(images)}")
    stems: dict[str, Path] = {}
    for image in images:
        if not image.name.isascii():
            raise ValueError(f"Shelf image filename is not ASCII: {image.name}")
        stem_key = image.stem.casefold()
        if stem_key in stems:
            raise ValueError(
                f"Duplicate shelf image stem: {stems[stem_key].name} and {image.name}"
            )
        stems[stem_key] = image
        _validate_decodable_image(image, "Shelf image")
    return images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise RuntimeError(f"Copied file checksum mismatch: {source} -> {destination}")


def _required_repo_files(repo_root: Path) -> dict[Path, Path]:
    return {
        repo_root / "yoloe_autolabel.py": Path("yoloe_autolabel.py"),
        repo_root / "requirements.txt": Path("requirements.txt"),
        repo_root / "tests" / "test_yoloe_autolabel.py": Path("tests/test_yoloe_autolabel.py"),
        repo_root / "colab" / "config.yaml": Path("config.yaml"),
        repo_root / "colab" / "hair_colab_enterprise.ipynb": Path("hair_colab_enterprise.ipynb"),
    }


def _write_registry_files(
    package_root: Path,
    skus: Sequence[HairSku],
    matched_references: Mapping[int, Path],
) -> None:
    manifest_path = package_root / "sku_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_id", "barcode", "brand", "sku_name", "enabled"),
        )
        writer.writeheader()
        for sku in skus:
            writer.writerow(
                {
                    "class_id": sku.class_id,
                    "barcode": sku.barcode,
                    "brand": "Unspecified",
                    "sku_name": sku.sku_name,
                    "enabled": "true",
                }
            )

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write reference prompts") from exc
    records = [
        {
            "class_id": sku.class_id,
            "image": f"references/{ascii_reference_name(sku, matched_references[sku.class_id])}",
        }
        for sku in skus
    ]
    (package_root / "reference_prompts.yaml").write_text(
        yaml.safe_dump(
            {"references": records},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _verify_ascii_tree(package_root: Path) -> None:
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if not relative.as_posix().isascii():
            raise ValueError(f"Prepared path is not ASCII: {relative}")


def _zip_tree(package_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
            archive_name = Path("hair_colab") / path.relative_to(package_root)
            if path.is_dir():
                if not any(path.iterdir()):
                    archive.writestr(f"{archive_name.as_posix()}/", b"")
                continue
            archive.write(path, archive_name.as_posix())


def build_runtime_package(
    inputs: PackageInputs,
    destination: Path,
    overwrite: bool = False,
) -> PackageReport:
    """Validate sources and atomically create the Colab runtime ZIP."""
    archive_path = Path(destination).resolve()
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {archive_path}")
    repo_root = Path(inputs.repo_root).resolve()
    required_files = _required_repo_files(repo_root)
    for source in required_files:
        if not source.is_file():
            raise ValueError(f"Required package input is missing: {source}")

    skus = load_sheet2_skus(inputs.workbook, inputs.expected_skus)
    overrides = load_reference_overrides(inputs.overrides)
    references = match_product_references(skus, inputs.product_images, overrides)
    for source in references.values():
        _validate_decodable_image(source, "Product reference")
    shelves = discover_shelf_images(inputs.shelf_images, inputs.expected_shelves)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hair-colab-package-", dir=archive_path.parent
    ) as temporary_directory:
        package_root = Path(temporary_directory) / "hair_colab"
        package_root.mkdir()
        for source, relative_destination in required_files.items():
            _copy_verified(source, package_root / relative_destination)
        _write_registry_files(package_root, skus, references)
        for sku in skus:
            source = references[sku.class_id]
            _copy_verified(
                source,
                package_root / "references" / ascii_reference_name(sku, source),
            )
        for source in shelves:
            _copy_verified(source, package_root / "shelf_images" / source.name)
        (package_root / "output").mkdir()
        _verify_ascii_tree(package_root)
        temporary_archive = Path(temporary_directory) / "hair_colab_runtime.zip"
        _zip_tree(package_root, temporary_archive)
        os.replace(temporary_archive, archive_path)

    return PackageReport(archive_path, len(skus), len(references), len(shelves))


def _positive_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--product-images", required=True)
    parser.add_argument("--shelf-images", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--overrides", default="colab/reference_name_overrides.yaml")
    parser.add_argument("--output", default="dist/hair_colab_runtime.zip")
    parser.add_argument("--expected-skus", type=_positive_argument, default=89)
    parser.add_argument("--expected-shelves", type=_positive_argument, default=632)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_runtime_package(
            PackageInputs(
                workbook=Path(args.workbook),
                product_images=Path(args.product_images),
                shelf_images=Path(args.shelf_images),
                repo_root=Path(args.repo_root),
                overrides=Path(args.overrides),
                expected_skus=args.expected_skus,
                expected_shelves=args.expected_shelves,
            ),
            Path(args.output),
            overwrite=args.overwrite,
        )
        print(
            f"Created {report.archive_path}: {report.sku_count} SKUs, "
            f"{report.reference_count} references, {report.shelf_count} shelf images"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

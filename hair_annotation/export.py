"""Strict, byte-preserving Platform datasets and human-review archives."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from hair_annotation.config import ExportConfig
from yoloe_autolabel import (
    _absolute_lexical,
    _assert_input_allowed,
    _reject_output_alias,
    _reject_reviewed_labels,
    load_sku_manifest,
    resolve_config_path,
)


def _validate_export(config: ExportConfig) -> None:
    fraction = config.train_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("export.train_fraction must be finite and strictly between 0 and 1")
    if not isinstance(config.split_seed, str) or not config.split_seed.strip():
        raise ValueError("export.split_seed must be a non-empty string")


def assign_splits(image_names: Sequence[str], train_fraction: float, seed: str) -> dict[str, str]:
    """Assign exact-count splits independently of enumeration order."""
    _validate_export(ExportConfig(train_fraction, seed))
    if any(not isinstance(name, str) or not name for name in image_names):
        raise ValueError("Image names must be non-empty strings")
    names = sorted(set(image_names), key=lambda name: (hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(), name))
    val_count = min(len(names) - 1, max(1, round(len(names) * (1.0 - train_fraction)))) if len(names) >= 2 else 0
    return {name: "val" if index < val_count else "train" for index, name in enumerate(names)}


def _input(path: Path) -> Path:
    lexical = _absolute_lexical(path)
    resolved = _assert_input_allowed(lexical)
    if resolved != lexical:
        raise ValueError(f"Input is redirected by a symlink or junction: {path}")
    return resolved


@dataclass(frozen=True)
class _ProjectSettings:
    images: Path
    manifest: Path
    raw: Path
    extensions: frozenset[str]
    export: ExportConfig
    protected: tuple[Path, ...]


def _project_settings(project: Path) -> _ProjectSettings:
    root = _input(Path(project))
    config_path = _input(root / "config.yaml")
    defaults = ExportConfig(0.90, "hair-osa-v1")
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    protected = [config_path]
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot read export config: {config_path}") from exc
        if not isinstance(data, dict) or any(not isinstance(data.get(section), dict) for section in ("dataset", "output", "export")):
            raise ValueError("Export config requires dataset, output, and export mappings")
        dataset = data["dataset"]
        def configured(*aliases: str) -> Path:
            value = next((dataset[name] for name in aliases if name in dataset), None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing configured dataset path: {aliases[0]}")
            return _input(resolve_config_path(config_path, value))
        images = configured("images", "shelf_images", "image_root")
        manifest = configured("manifest", "sku_manifest")
        output_value = data["output"].get("root")
        if not isinstance(output_value, str) or not output_value.strip():
            raise ValueError("Missing configured output.root")
        raw = _input(resolve_config_path(config_path, output_value) / "raw_predictions")
        extensions = dataset.get("extensions", dataset.get("image_extensions", extensions))
        export_data = data["export"]
        if "train_fraction" not in export_data or "split_seed" not in export_data:
            raise ValueError("Export config requires train_fraction and split_seed")
        defaults = ExportConfig(export_data["train_fraction"], export_data["split_seed"])
        for alias in ("references", "reference_definitions"):
            if alias in dataset:
                protected.append(configured(alias))
                break
    else:
        images = _input(root / "shelf_images")
        manifest = _input(root / "sku_manifest.csv")
        raw = _input(root / "output" / "raw_predictions")
    if not isinstance(extensions, (list, tuple)) or not extensions or any(not isinstance(extension, str) or not extension.strip().lstrip(".") for extension in extensions):
        raise ValueError("dataset.image_extensions must be a non-empty sequence of extensions")
    _validate_export(defaults)
    protected.extend((images, manifest, raw))
    return _ProjectSettings(images, manifest, raw, frozenset("." + extension.strip().lstrip(".").casefold() for extension in extensions), defaults, tuple(protected))


def _files(directory: Path) -> list[Path]:
    directory = _input(directory)
    if not directory.is_dir():
        raise ValueError(f"Required artifact directory is missing: {directory}")
    files = []
    for path in sorted(directory.rglob("*")):
        source = _input(path)
        if source.is_file():
            files.append(source)
    return files


def _manifest_names(path: Path) -> dict[int, str]:
    skus = load_sku_manifest(_input(path))
    if set(skus) != set(range(89)):
        raise ValueError("Permanent manifest class IDs must be exactly 0-88; class 89 is reserved for Needs Review")
    return {**{class_id: sku.sku_name for class_id, sku in skus.items()}, 89: "Needs Review"}


def _validate_label(path: Path) -> None:
    try:
        lines = _input(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read YOLO label: {path}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("expected five fields")
            class_id = int(fields[0])
            if not 0 <= class_id <= 89:
                raise ValueError("class ID must be 0-89")
            coordinates = [float(value) for value in fields[1:]]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError("coordinates must be finite and normalized")
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label {path}:{number}: {exc}") from exc


def _platform_members(settings: _ProjectSettings, config: ExportConfig) -> list[tuple[str, Path | bytes | None]]:
    _validate_export(config)
    names = _manifest_names(settings.manifest)
    images = [path for path in _files(settings.images) if path.suffix.casefold() in settings.extensions]
    labels = _files(settings.raw / "labels")
    by_stem: dict[str, Path] = {}
    for image in images:
        key = image.stem.casefold()
        if key in by_stem:
            raise ValueError(f"Duplicate case-insensitive image stem: {image.stem}")
        by_stem[key] = image
    label_stems: dict[str, Path] = {}
    for label in labels:
        if label.suffix.casefold() != ".txt":
            raise ValueError(f"Unexpected file in labels directory: {label}")
        key = label.stem.casefold()
        if key in label_stems:
            raise ValueError(f"Duplicate case-insensitive label stem: {label.stem}")
        label_stems[key] = label
        _validate_label(label)
    if set(by_stem) != set(label_stems):
        raise ValueError("Every image and label must have exactly one matching pair")
    if len(images) < 2:
        raise ValueError("Platform packaging requires at least two successful images")
    splits = assign_splits([image.name for image in images], config.train_fraction, config.split_seed)
    data = {"path": ".", "train": "images/train", "val": "images/val", "names": names}
    members: list[tuple[str, Path | bytes | None]] = [("data.yaml", yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8"))]
    for image in sorted(images, key=lambda path: path.name):
        split = splits[image.name]
        members.extend(((f"images/{split}/{image.name}", image), (f"labels/{split}/{image.stem}.txt", label_stems[image.stem.casefold()])))
    return members


def _review_members(settings: _ProjectSettings) -> list[tuple[str, Path | bytes | None]]:
    _manifest_names(settings.manifest)
    members: list[tuple[str, Path | bytes | None]] = [("sku_manifest.csv", settings.manifest)]
    for name in ("review_queue.csv", "summary.csv", "run.json", "provenance.json", "reference_diagnostics.json"):
        source = _input(settings.raw / name)
        if not source.is_file():
            raise ValueError(f"Required review artifact is missing: {source}")
        members.append((name, source))
    for name in ("metadata", "previews"):
        directory = settings.raw / name
        sources = _files(directory)
        members.append((f"{name}/", None))
        members.extend((f"{name}/{source.relative_to(directory).as_posix()}", source) for source in sources)
    members.append(("README.txt", b"Platform labels are machine suggestions requiring human review.\nAll Needs Review annotations must be reassigned before class 89 is deleted.\nThis archive contains review evidence, not a training dataset.\n"))
    return members


def _destination(path: Path, inputs: Sequence[Path], overwrite: bool) -> Path:
    lexical = _absolute_lexical(Path(path))
    _reject_reviewed_labels(lexical)
    destination = lexical.resolve()
    if lexical != destination:
        raise ValueError(f"Archive destination is redirected by a symlink or junction: {path}")
    _reject_output_alias(destination, inputs)
    for source in inputs:
        if source.is_dir() and destination.is_relative_to(source.resolve()):
            raise ValueError(f"Archive destination overlaps an input directory: {source}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Archive destination is not a file: {destination}")
        if not overwrite:
            raise FileExistsError(f"Results archive already exists: {destination}")
    return destination


def _member_inputs(settings: _ProjectSettings, members: Sequence[tuple[str, Path | bytes | None]]) -> list[Path]:
    # Protect original images and all raw evidence, even when excluded from a ZIP.
    return list(dict.fromkeys([
        *settings.protected,
        *_files(settings.images),
        *_files(settings.raw),
        *(source for _, source in members if isinstance(source, Path)),
    ]))


def _atomic_archive(destination: Path, members: Sequence[tuple[str, Path | bytes | None]], inputs: Sequence[Path], overwrite: bool) -> Path:
    destination = _destination(destination, inputs, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, source in members:
                if isinstance(source, Path):
                    archive.write(_input(source), name)
                else:
                    archive.writestr(name, source or b"")
        with zipfile.ZipFile(temporary) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP integrity validation failed: {bad_member}")
        _destination(destination, inputs, overwrite)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def package_platform_dataset(project: Path, destination: Path, config: ExportConfig, *, overwrite: bool = False) -> Path:
    """Preflight and atomically write only standard Ultralytics YOLO members."""
    settings = _project_settings(project)
    members = _platform_members(settings, config)
    inputs = _member_inputs(settings, members)
    return _atomic_archive(destination, members, inputs, overwrite)


def package_review_bundle(project: Path, destination: Path, *, overwrite: bool = False) -> Path:
    """Preflight and atomically write full-run review evidence separately."""
    settings = _project_settings(project)
    members = _review_members(settings)
    inputs = _member_inputs(settings, members)
    return _atomic_archive(destination, members, inputs, overwrite)

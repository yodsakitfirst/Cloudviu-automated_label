"""Runtime-local helpers used by the Colab Enterprise notebook."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def environment_report() -> dict[str, Any]:
    """Return a compact report of the active runtime and accelerator."""
    disk = shutil.disk_usage(Path.cwd())
    cuda_available = False
    cuda_device = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "disk_free_gib": round(disk.free / (1024**3), 1),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_pilot_config(
    project: Path,
    max_skus: int = 89,
    max_images: int = 10,
) -> Path:
    """Create a separate pilot config without mutating the full-run config."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to create the pilot config") from exc
    if max_skus <= 0 or max_images <= 0:
        raise ValueError("Pilot SKU and image limits must be positive")
    project_root = Path(project).resolve()
    config_path = project_root / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read full-run config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Full-run config must be a mapping: {config_path}")
    output = config.get("output")
    pilot = config.get("pilot")
    if not isinstance(output, dict) or not isinstance(pilot, dict):
        raise ValueError("Full-run config requires output and pilot mappings")
    output["root"] = "pilot_output"
    pilot.update(
        {
            "enabled": True,
            "class_ids": [],
            "max_skus": max_skus,
            "max_images": max_images,
        }
    )
    destination = project_root / "pilot_config.yaml"
    _atomic_write_text(
        destination,
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    )
    return destination


def package_results(
    project: Path,
    platform_destination: Path,
    review_destination: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Build separate Platform/review ZIPs after preflighting both archives."""
    from hair_annotation import export

    settings = export._project_settings(project)
    platform_members = export._platform_members(settings, settings.export)
    review_members = export._review_members(settings)
    inputs = export._member_inputs(settings, [*platform_members, *review_members])
    platform_path = export._destination(platform_destination, inputs, overwrite)
    review_path = export._destination(review_destination, inputs, overwrite)
    aliases = platform_path == review_path
    if platform_path.exists() and review_path.exists():
        aliases = aliases or os.path.samefile(platform_path, review_path)
    if aliases:
        raise ValueError("Platform and review destinations must be distinct, not aliases")
    platform_path = export._atomic_archive(platform_path, platform_members, inputs, overwrite)
    review_path = export._atomic_archive(review_path, review_members, inputs, overwrite)
    return platform_path, review_path

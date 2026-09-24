"""Runtime and optional dependency provenance capture."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Iterable


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class EnvironmentInfo:
    python: str
    platform: str
    packages: dict[str, str | None]
    cuda_available: bool | None
    cuda_version: str | None
    devices: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collect_environment(
    package_names: Iterable[str] = (
        "torch",
        "torchvision",
        "ultralytics",
        "albumentations",
        "pycocotools",
    ),
) -> EnvironmentInfo:
    """Capture versions and device facts without loading a model."""
    packages = {name: _package_version(name) for name in package_names}
    cuda_available: bool | None = None
    cuda_version: str | None = None
    devices: tuple[str, ...] = ()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            devices = tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
    except (ImportError, RuntimeError):
        pass
    return EnvironmentInfo(
        python=sys.version,
        platform=platform.platform(),
        packages=packages,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        devices=devices,
    )

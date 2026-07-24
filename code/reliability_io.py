"""Small reusable I/O and provenance helpers for the reliability pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def normalize_bids_labels(
    values: Optional[Iterable[str]],
    prefix: str = "",
) -> Optional[set[str]]:
    """Normalize optional selectors while accepting BIDS-style prefixes."""
    if values is None:
        return None
    return {str(value).removeprefix(prefix) for value in values}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_package_version(distribution_name: str) -> Optional[str]:
    """Return an installed distribution version, or ``None`` if unavailable."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def format_epoch_len(len_epoch: float) -> str:
    """Convert an epoch duration in seconds to a rounded millisecond tag."""
    return f"{int(np.round(len_epoch * 1000))}ms"


def atomic_write_json(
    output_path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = False,
) -> None:
    """Write indented JSON through a temporary sibling file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_feather(
    dataframe: Any,
    output_path: str | Path,
    overwrite: bool,
) -> None:
    """Write one Feather dataframe without exposing a partial output file."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        dataframe.to_feather(temporary_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

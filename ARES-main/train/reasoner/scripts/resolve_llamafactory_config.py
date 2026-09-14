#!/usr/bin/env python3
"""Resolve config-relative paths in a LLaMA-Factory yaml file."""

from __future__ import annotations

import pathlib
import sys

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to resolve LLaMA-Factory config paths.") from exc

PATH_KEYS = (
    "model_name_or_path",
    "dataset_dir",
    "output_dir",
    "logging_dir",
    "cache_dir",
)


def _resolve_path(value: str, base: pathlib.Path) -> str:
    if value.startswith("~"):
        return str(pathlib.Path(value).expanduser().resolve())
    if value.startswith("/"):
        return str(pathlib.Path(value).resolve())
    return str((base / value).resolve())


def resolve_config(src: pathlib.Path, dst: pathlib.Path) -> None:
    base = src.parent
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value:
            cfg[key] = _resolve_path(value, base)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <src.yaml> <dst.yaml>")
    resolve_config(pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]))


if __name__ == "__main__":
    main()

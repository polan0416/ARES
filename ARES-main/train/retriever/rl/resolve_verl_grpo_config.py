#!/usr/bin/env python3
"""Resolve config-relative paths in verl GRPO yaml files."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to resolve verl GRPO config paths.") from exc

PATH_KEYS = frozenset(
    {
        "path",
        "lora_adapter_path",
        "default_local_dir",
        "interaction_config_path",
        "builder_module_path",
        "train_files",
        "val_files",
    }
)


def _resolve_path(value: str, base: pathlib.Path) -> str:
    if value.startswith("~"):
        return str(pathlib.Path(value).expanduser().resolve())
    if value.startswith("/"):
        return str(pathlib.Path(value).resolve())
    return str((base / value).resolve())


def _resolve_node(node: Any, base: pathlib.Path) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in PATH_KEYS:
                if isinstance(value, str):
                    out[key] = _resolve_path(value, base)
                elif isinstance(value, list):
                    out[key] = [
                        _resolve_path(item, base) if isinstance(item, str) else _resolve_node(item, base)
                        for item in value
                    ]
                else:
                    out[key] = _resolve_node(value, base)
            else:
                out[key] = _resolve_node(value, base)
        return out
    if isinstance(node, list):
        return [_resolve_node(item, base) for item in node]
    return node


def resolve_config(src: pathlib.Path, dst: pathlib.Path) -> None:
    base = src.parent
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    interaction_path = None
    if isinstance(cfg, dict):
        interaction_path = (
            cfg.get("actor_rollout_ref", {})
            .get("rollout", {})
            .get("multi_turn", {})
            .get("interaction_config_path")
        )
    if isinstance(interaction_path, str) and interaction_path and not interaction_path.startswith("/"):
        resolved_interaction = _resolve_path(interaction_path, base)
        interaction_src = pathlib.Path(resolved_interaction)
        if interaction_src.is_file():
            interaction_dst = dst.parent / f"{interaction_src.stem}.resolved.yaml"
            resolve_config(interaction_src, interaction_dst)
            cfg["actor_rollout_ref"]["rollout"]["multi_turn"]["interaction_config_path"] = str(
                interaction_dst.resolve()
            )
    resolved = _resolve_node(cfg, base)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <src.yaml> <dst.yaml>")
    resolve_config(pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_environment_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, raw_value = stripped.split("=", 1)
    value = raw_value.strip().strip("'").strip('"')
    return key.strip(), value


def load_environment_file(path: Path | None = None) -> None:
    environment_path = path or Path(".env")
    if not environment_path.exists():
        logger.debug("[environment] no .env file at %s", environment_path)
        return

    loaded_keys: list[str] = []
    for line in environment_path.read_text().splitlines():
        entry = parse_environment_line(line)
        if entry is None:
            continue
        key, value = entry
        os.environ.setdefault(key, value)
        loaded_keys.append(key)

    logger.debug(
        "[environment] loaded environment keys: %s",
        ", ".join(loaded_keys) if loaded_keys else "(none)",
    )

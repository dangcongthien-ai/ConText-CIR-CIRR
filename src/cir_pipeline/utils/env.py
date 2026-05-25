"""Environment variable loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing project marker files."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        if (path / "configs" / "cirr_context_cir_cl_cc.yaml").exists() and (path / "src").exists():
            return path
    return current


def _manual_load_env(path: Path, override: bool = False) -> None:
    """Tiny .env parser fallback for KEY=VALUE lines."""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def load_project_env(start: str | Path | None = None, override: bool = False) -> Path | None:
    """Load `.env` from the project root if it exists."""

    env_path = find_project_root(start) / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=override)
    except Exception:
        _manual_load_env(env_path, override=override)
    return env_path

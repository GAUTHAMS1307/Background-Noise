from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


class FrameCSVLogger:
    """CSV logger for per-frame telemetry."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._path_opened = False
        self._writer = None
        self._file = None

    def _ensure_open(self, fieldnames: Iterable[str]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
            self._path_opened = True

    def log(self, row: dict) -> None:
        if self._writer is None:
            self._ensure_open(row.keys())
        self._writer.writerow(row)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self.close()

"""Log capture — tees stdout/stderr to a file while preserving terminal output."""

from __future__ import annotations

import io
import sys
from pathlib import Path


class _TeeWriter:
    """Writes to both the original stream and a file."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class LogCapture:
    """Captures stdout/stderr to a log file via tee."""

    def __init__(self):
        self._log_file: io.TextIOBase | None = None
        self._orig_stdout = None
        self._orig_stderr = None

    def start(self, log_dir: Path, run_id: int) -> Path:
        """Begin capturing. Creates log_dir if needed. Returns path to log file."""
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{run_id:03d}.log"
        self._log_file = open(log_path, "w")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._orig_stdout, self._log_file)
        sys.stderr = _TeeWriter(self._orig_stderr, self._log_file)
        return log_path

    def stop(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

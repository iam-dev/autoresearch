"""ExperimentHooks protocol — the contract all condition implementations satisfy."""

from __future__ import annotations

from typing import Protocol

from hooks.types import PreRunContext, RunConfig, RunResults


class ExperimentHooks(Protocol):
    def pre_run(self, config: RunConfig) -> PreRunContext:
        """Called BEFORE training. Returns context/suggestions."""
        ...

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        """Called AFTER training. Records results."""
        ...

    def start_log_capture(self, config: RunConfig) -> None:
        """Called after pre_run to start capturing stdout/stderr to a log file."""
        ...

    def stop_log_capture(self) -> None:
        """Called before post_run to stop capturing and flush the log file."""
        ...

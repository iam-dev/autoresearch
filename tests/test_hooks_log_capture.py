"""Tests for hooks/log_capture.py."""

import sys

from hooks.log_capture import LogCapture


class TestLogCapture:
    def test_captures_stdout(self, tmp_path):
        capture = LogCapture()
        log_dir = tmp_path / "logs"
        capture.start(log_dir, 1)
        print("test output")
        capture.stop()

        log_file = log_dir / "run_001.log"
        assert log_file.exists()
        assert "test output" in log_file.read_text()

    def test_restores_stdout(self, tmp_path):
        original_stdout = sys.stdout
        capture = LogCapture()
        capture.start(tmp_path / "logs", 1)
        assert sys.stdout is not original_stdout
        capture.stop()
        assert sys.stdout is original_stdout

    def test_returns_log_path(self, tmp_path):
        capture = LogCapture()
        log_dir = tmp_path / "logs"
        path = capture.start(log_dir, 1)
        capture.stop()
        assert path == log_dir / "run_001.log"

    def test_double_stop_is_safe(self, tmp_path):
        capture = LogCapture()
        capture.start(tmp_path / "logs", 1)
        capture.stop()
        capture.stop()  # no-op

    def test_stop_without_start_is_safe(self):
        capture = LogCapture()
        capture.stop()  # no-op

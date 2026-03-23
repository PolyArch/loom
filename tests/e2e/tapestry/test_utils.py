"""Common test utilities for Tapestry integration tests."""

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TIMEOUT_SEC = 120


def run_tapestry_tool(
    binary: Path,
    args: List[str],
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a Tapestry tool binary with the given arguments.

    Returns the CompletedProcess result. Does not raise on non-zero exit code.
    """
    cmd = [str(binary)] + args
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=cwd,
        env=merged_env,
    )


def load_json_report(output_dir: Path) -> Dict[str, Any]:
    """Load the JSON report from a tapestry output directory.

    Searches for report.json or tapestry_report.json in the output directory.
    """
    candidates = [
        output_dir / "report.json",
        output_dir / "tapestry_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    # Try any JSON file in the directory
    for f in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    raise FileNotFoundError(
        f"No JSON report found in {output_dir}. "
        f"Files present: {list(output_dir.iterdir()) if output_dir.exists() else []}"
    )


def assert_metric_in_range(
    metrics: Dict[str, Any],
    key: str,
    low: float,
    high: float,
    msg: str = "",
) -> None:
    """Assert that a numeric metric value falls within [low, high]."""
    assert key in metrics, f"Missing metric key '{key}' in {list(metrics.keys())}"
    val = float(metrics[key])
    context = f" ({msg})" if msg else ""
    assert low <= val <= high, (
        f"Metric '{key}' = {val} out of range [{low}, {high}]{context}"
    )


def assert_success_output(result: subprocess.CompletedProcess, tool_name: str) -> None:
    """Assert that a tapestry tool run succeeded."""
    assert result.returncode == 0, (
        f"{tool_name} failed (rc={result.returncode}).\n"
        f"STDOUT:\n{result.stdout[:2000]}\n"
        f"STDERR:\n{result.stderr[:2000]}"
    )
    assert "SUCCESS" in result.stdout, (
        f"{tool_name} did not report SUCCESS.\n"
        f"STDOUT:\n{result.stdout[:2000]}"
    )


def assert_files_exist(output_dir: Path, patterns: List[str]) -> None:
    """Assert that files matching the given glob patterns exist in output_dir."""
    for pattern in patterns:
        matches = list(output_dir.glob(pattern))
        assert len(matches) > 0, (
            f"Expected file matching '{pattern}' in {output_dir}, "
            f"but found none. Contents: {sorted(f.name for f in output_dir.iterdir())}"
        )


def count_files_matching(output_dir: Path, pattern: str) -> int:
    """Count files matching a glob pattern in the output directory."""
    return len(list(output_dir.glob(pattern)))


def check_no_error_strings(text: str, forbidden: Optional[List[str]] = None) -> None:
    """Check that output text does not contain error indicators."""
    if forbidden is None:
        forbidden = ["FATAL", "Segmentation fault", "Assertion failed"]
    for s in forbidden:
        assert s not in text, f"Found forbidden string '{s}' in output"

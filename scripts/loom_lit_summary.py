#!/usr/bin/env python3
"""Run LLVM lit and summarize unsupported tests by lit-owned reason."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


_MISSING_FEATURES_PREFIX = (
    "Test requires the following unavailable features: "
)
_INDEPENDENT_TEST_BATCH_SIZE = "1"


def _explicit_report_path(arguments: Sequence[str]) -> Path | None:
    result = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-o", "--output"):
            if index + 1 >= len(arguments):
                raise ValueError(f"{argument} requires a report path")
            result = Path(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--output=") or argument.startswith("-o="):
            result = Path(argument.split("=", 1)[1])
        elif argument.startswith("-o") and len(argument) > 2:
            result = Path(argument[2:])
        index += 1
    return result


def _unsupported_reason(output: object) -> str:
    text = " ".join(str(output or "").split())
    if text.startswith(_MISSING_FEATURES_PREFIX):
        features = text.removeprefix(_MISSING_FEATURES_PREFIX).split(",")
        return ", ".join(sorted(feature.strip() for feature in features))
    return text or "unspecified reason"


def unsupported_summary(report: Mapping[str, object]) -> tuple[str, ...]:
    tests = report.get("tests")
    if not isinstance(tests, list):
        raise ValueError("lit JSON report has no tests array")
    if not tests:
        return ()

    counts: dict[str, int] = {}
    for test in tests:
        if not isinstance(test, dict) or test.get("code") != "UNSUPPORTED":
            continue
        reason = _unsupported_reason(test.get("output"))
        counts[reason] = counts.get(reason, 0) + 1

    total = len(tests)
    return tuple(
        f"Unsupported ({reason}): {count} ({100.0 * count / total:.2f}%)"
        for reason, count in sorted(counts.items())
    )


def run_lit_with_unsupported_summary(
    llvm_lit: Path,
    test_root: Path,
    jobs: int,
    extra_arguments: Sequence[str],
    environment: Mapping[str, str],
    runner: Callable[..., object],
) -> None:
    report_path = _explicit_report_path(extra_arguments)
    owns_report = report_path is None
    if owns_report:
        descriptor, raw_path = tempfile.mkstemp(
            dir=test_root.parent,
            prefix=".loom-lit-",
            suffix=".json",
        )
        os.close(descriptor)
        report_path = Path(raw_path)

    assert report_path is not None
    report_arguments = [] if not owns_report else ["--output", str(report_path)]
    lit_environment = dict(environment)
    # Lit batches adjacent tests into one serial worker task by default. Smart
    # ordering puts the longest tests next to each other, so batching turns
    # independent long-running tests into the suite's critical path.
    lit_environment["LIT_BATCH_SIZE"] = _INDEPENDENT_TEST_BATCH_SIZE
    try:
        runner(
            [
                str(llvm_lit),
                "-sv",
                "--time-tests",
                f"-j{jobs}",
                *extra_arguments,
                *report_arguments,
                str(test_root),
            ],
            env=lit_environment,
        )
        report = json.loads(report_path.read_text())
        for line in unsupported_summary(report):
            print(line)
    finally:
        if owns_report:
            report_path.unlink(missing_ok=True)

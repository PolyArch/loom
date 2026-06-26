#!/usr/bin/env python3
"""Audit long-running test runners for explicit parallel execution controls."""

from __future__ import annotations

import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_script_has_jobs_option(path: Path) -> str:
    text = path.read_text()
    require("--jobs", f"{path} must expose an explicit --jobs option")
    require("LOOM_TEST_JOBS" in text, f"{path} must honor LOOM_TEST_JOBS")
    require("JOBS" in text, f"{path} must honor JOBS")
    return text


def assert_sweep_parallelized(path: Path) -> None:
    text = assert_script_has_jobs_option(path)
    require("wait -n" in text, f"{path} must throttle independent case jobs with wait -n")
    require("run_case_job" in text, f"{path} must isolate per-case sweep work in a job function")
    require("validate_unique_cases" in text, f"{path} must reject duplicate case jobs")
    require("duplicate --case" in text, f"{path} must diagnose duplicate case jobs")
    require(
        "for case_name in \"${CASES[@]}\"; do\n  case_out=\"${chain_root}/${case_name}\"\n  rm -rf" not in text,
        f"{path} still has the old serial case loop",
    )


def assert_app_runner_parallelized(path: Path) -> None:
    text = assert_script_has_jobs_option(path)
    require("wait -n" in text, f"{path} must throttle independent kernel jobs with wait -n")
    require("run_kernel_job" in text, f"{path} must isolate per-kernel work in a job function")
    require(
        'rm -rf "${STATUS_ROOT}" "${LOG_ROOT}"' in text,
        f"{path} must clear stale per-kernel status/log roots before launch",
    )


def assert_rollup_parallelized(path: Path) -> None:
    text = assert_script_has_jobs_option(path)
    require("wait -n" in text, f"{path} must join independent producer jobs with wait -n")
    require("run_rollup_producer_job" in text, f"{path} must isolate independent producer lanes")
    require(
        'rm -rf "${PRODUCER_LOG_DIR}" "${PRODUCER_STATUS_DIR}"' in text,
        f"{path} must clear stale producer status/log roots before launch",
    )


def assert_chain_breadth_parallelized(path: Path) -> None:
    text = path.read_text()
    require("ThreadPoolExecutor" in text, f"{path} must parallelize independent chain cases")
    require("LOOM_CHAIN_BREADTH_JOBS" in text, f"{path} must expose a chain-case worker budget")
    require("LOOM_TEST_JOBS" in text, f"{path} must honor the shared test worker budget")


def assert_artifact_gates_parallelized(path: Path) -> None:
    text = path.read_text()
    require("ThreadPoolExecutor" in text, f"{path} must parallelize independent artifact producers")
    require("LOOM_ARTIFACT_GATES_JOBS" in text, f"{path} must expose an artifact-gate worker budget")
    require("LOOM_TEST_JOBS" in text, f"{path} must honor the shared test worker budget")


def assert_app_build_dir_runner_parallelized(path: Path) -> None:
    text = path.read_text()
    require("ThreadPoolExecutor" in text, f"{path} must parallelize independent case runner checks")
    require("LOOM_APP_BUILD_DIR_JOBS" in text, f"{path} must expose an app BUILD_DIR worker budget")
    require("LOOM_TEST_JOBS" in text, f"{path} must honor the shared test worker budget")


def assert_source_compat_parallelized(path: Path) -> None:
    text = path.read_text()
    require("--jobs" in text, f"{path} must expose an explicit --jobs option")
    require("ThreadPoolExecutor" in text, f"{path} must parallelize independent compatibility checks")
    require("LOOM_SOURCE_COMPAT_JOBS" in text, f"{path} must expose a source-compat worker budget")
    require("LOOM_TEST_JOBS" in text, f"{path} must honor the shared test worker budget")


def assert_artifact_lit_group(repo: Path) -> None:
    cfg = (repo / "test/lit.cfg.py").read_text()
    local_cfg = repo / "test/artifacts/lit.local.cfg.py"
    require(local_cfg.exists(), "artifact lit tests must declare a parallelism group")
    local_text = local_cfg.read_text()
    require(
        "parallelism_groups[\"artifacts\"]" in cfg,
        "top-level lit config must cap artifact test concurrency",
    )
    require(
        "LOOM_ARTIFACT_TEST_JOBS" in cfg,
        "artifact lit concurrency must be externally configurable",
    )
    require(
        '"JOBS", "LOOM_TEST_JOBS"' in cfg,
        "lit must forward shared worker budgets into test processes",
    )
    require(
        '"LOOM_ARTIFACT_GATES_JOBS", "LOOM_CHAIN_BREADTH_JOBS"' in cfg,
        "lit must forward artifact sub-runner worker budgets into test processes",
    )
    require(
        'config.parallelism_group = "artifacts"' in local_text,
        "artifact lit tests must use the artifacts parallelism group",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <repo>")
    repo = Path(argv[1]).resolve()
    assert_artifact_lit_group(repo)
    assert_sweep_parallelized(repo / "test/e2e/run_cgra_sim_evidence_sweep.sh")
    assert_rollup_parallelized(repo / "test/e2e/run_cmsis_cgra_status_rollup.sh")
    assert_app_runner_parallelized(repo / "test/app/run_all.sh")
    assert_app_runner_parallelized(repo / "test/app/run_raise_all.sh")
    assert_app_runner_parallelized(repo / "test/app/run_dfg_all.sh")
    assert_chain_breadth_parallelized(repo / "test/artifacts/test_intermediate_artifact_chain_breadth.py")
    assert_artifact_gates_parallelized(repo / "test/artifacts/test_intermediate_artifacts.py")
    assert_app_build_dir_runner_parallelized(repo / "test/artifacts/test_app_runner_build_dir.py")
    assert_source_compat_parallelized(repo / "test/app/source_compat_summary.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

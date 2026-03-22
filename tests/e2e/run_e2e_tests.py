#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
from typing import List


DEFAULT_TIMEOUT_SEC = 900


@dataclass(frozen=True)
class E2ECase:
    name: str
    output_dir: Path
    app_source: Path
    app_include_dir: Path
    builder_exec: Path
    generated_fabric_file: Path
    loom_exec: Path
    sim_bundle_template: Path | None


@dataclass(frozen=True)
class E2EResult:
    case: E2ECase
    status: str
    return_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tests", nargs="*", default=[])
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("LOOM_E2E_JOBS", "0") or "0"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("LOOM_E2E_TIMEOUT", str(DEFAULT_TIMEOUT_SEC))),
    )
    return parser.parse_args()


def load_cases(manifest_path: Path) -> List[E2ECase]:
    cases: List[E2ECase] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            while len(row) < 8:
                row.append("")
            (name, output_dir, app_source, app_include_dir,
             builder_exec, generated_fabric_file, loom_exec,
             sim_bundle_template) = row[:8]
            cases.append(E2ECase(
                name=name,
                output_dir=Path(output_dir),
                app_source=Path(app_source),
                app_include_dir=Path(app_include_dir),
                builder_exec=Path(builder_exec),
                generated_fabric_file=Path(generated_fabric_file),
                loom_exec=Path(loom_exec),
                sim_bundle_template=(
                    Path(sim_bundle_template) if sim_bundle_template else None
                ),
            ))
    return cases


def validate_case_naming(case: E2ECase) -> None:
    if "." not in case.name:
        raise RuntimeError(
            f"e2e case name '{case.name}' must be '<app>.<adg>'"
        )

    app_name, hw_name = case.name.split(".", 1)
    if case.app_source.parent.name != app_name:
        raise RuntimeError(
            f"e2e case '{case.name}' does not match app path "
            f"'{case.app_source.parent.name}'"
        )
    if case.output_dir.name != case.name:
        raise RuntimeError(
            f"e2e output dir '{case.output_dir.name}' must equal case name "
            f"'{case.name}'"
        )
    expected_fabric_name = f"{hw_name}.fabric.mlir"
    if case.generated_fabric_file.name != expected_fabric_name:
        raise RuntimeError(
            f"e2e ADG artifact '{case.generated_fabric_file.name}' must be "
            f"'{expected_fabric_name}' for case '{case.name}'"
        )
    expected_output_dir = case.output_dir / expected_fabric_name
    if case.generated_fabric_file != expected_output_dir:
        raise RuntimeError(
            f"e2e ADG artifact path '{case.generated_fabric_file}' must live "
            f"under output dir '{case.output_dir}'"
        )


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def render_run_script(case: E2ECase, repo_root: Path) -> str:
    sim_bundle = case.output_dir / "sim.bundle.json"
    loom_cmd = [
        str(case.loom_exec),
        "-I",
        str(case.app_include_dir),
        str(case.app_source),
        "--adg",
        str(case.generated_fabric_file),
        "-o",
        str(case.output_dir),
    ]
    if case.sim_bundle_template is not None:
        loom_cmd.extend(["--simulate", "--sim-bundle", str(sim_bundle)])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(case.output_dir))}",
        shell_join([str(case.builder_exec), str(case.generated_fabric_file)]),
        shell_join(loom_cmd),
        "",
    ]
    return "\n".join(lines)


def render_gem5_run_script(case: E2ECase, repo_root: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        shell_join([str(case.output_dir / "run.cmd")]),
        shell_join([
            sys.executable,
            str(repo_root / "tools/gem5/run_loom_gem5_case.py"),
            "--case-dir",
            str(case.output_dir),
        ]),
        "",
    ]
    return "\n".join(lines)


def write_sim_bundle(case: E2ECase) -> None:
    if case.sim_bundle_template is None:
        return
    case.output_dir.mkdir(parents=True, exist_ok=True)
    target = case.output_dir / "sim.bundle.json"
    shutil.copyfile(case.sim_bundle_template, target)


def write_run_script(case: E2ECase, repo_root: Path) -> Path:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    write_sim_bundle(case)
    run_cmd = case.output_dir / "run.cmd"
    run_cmd.write_text(render_run_script(case, repo_root), encoding="utf-8")
    run_cmd.chmod(0o755)
    if case.sim_bundle_template is not None:
        run_gem5_cmd = case.output_dir / "run.gem5.cmd"
        run_gem5_cmd.write_text(
            render_gem5_run_script(case, repo_root), encoding="utf-8"
        )
        run_gem5_cmd.chmod(0o755)
    return run_cmd


def terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_case(case: E2ECase, repo_root: Path, timeout_sec: int) -> E2EResult:
    shutil.rmtree(case.output_dir, ignore_errors=True)
    run_cmd = write_run_script(case, repo_root)
    run_out = case.output_dir / "run.out"
    run_err = case.output_dir / "run.err"

    with run_out.open("w", encoding="utf-8") as out_handle, run_err.open(
        "w", encoding="utf-8"
    ) as err_handle:
        proc = subprocess.Popen(
            [str(run_cmd)],
            cwd=repo_root,
            stdout=out_handle,
            stderr=err_handle,
            start_new_session=True,
        )
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc)
            err_handle.write(f"\nTimed out after {timeout_sec} seconds.\n")
            err_handle.flush()
            return E2EResult(case=case, status="timeout", return_code=124)

    if return_code == 0:
        return E2EResult(case=case, status="pass", return_code=0)
    return E2EResult(case=case, status="fail", return_code=return_code)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    selected = set(args.tests)

    cases = load_cases(manifest_path)
    for case in cases:
        validate_case_naming(case)
    if selected:
        cases = [case for case in cases if case.name in selected]

    total = len(cases)
    skipped = 0
    passed = 0
    failed = 0
    timed_out = 0
    failure_lines: List[str] = []

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, max(total, 1)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(run_case, case, repo_root, args.timeout): case
            for case in cases
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            rel_out_dir = os.path.relpath(result.case.output_dir, repo_root)
            if result.status == "pass":
                passed += 1
            elif result.status == "timeout":
                timed_out += 1
                failure_lines.append(
                    f"- {rel_out_dir} timed out, please check "
                    f"{rel_out_dir}/run.{{err,out}}, or re-run with "
                    f"{rel_out_dir}/run.cmd"
                )
            else:
                failed += 1
                failure_lines.append(
                    f"- {rel_out_dir} failed, please check "
                    f"{rel_out_dir}/run.{{err,out}}, or re-run with "
                    f"{rel_out_dir}/run.cmd"
                )

    print(
        f"Total: {total}  Pass: {passed}  Fail: {failed}  "
        f"Timeout: {timed_out}  Skip: {skipped}"
    )
    for line in failure_lines:
        print(line)

    return 0 if failed == 0 and timed_out == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

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


DEFAULT_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class UnitCase:
    name: str
    output_dir: Path
    dfg_file: Path
    fabric_file: Path | None
    builder_exec: Path | None
    generated_fabric_file: Path | None
    fcc_exec: Path


@dataclass(frozen=True)
class UnitResult:
    case: UnitCase
    status: str
    return_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tests", nargs="*", default=[])
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("FCC_UNIT_JOBS", "0") or "0"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("FCC_UNIT_TIMEOUT", str(DEFAULT_TIMEOUT_SEC))),
    )
    return parser.parse_args()


def load_cases(manifest_path: Path) -> List[UnitCase]:
    cases: List[UnitCase] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
      reader = csv.reader(handle, delimiter="\t")
      for row in reader:
        if not row:
          continue
        while len(row) < 7:
          row.append("")
        name, output_dir, dfg_file, fabric_file, builder_exec, generated_fabric_file, fcc_exec = row[:7]
        cases.append(
            UnitCase(
                name=name,
                output_dir=Path(output_dir),
                dfg_file=Path(dfg_file),
                fabric_file=Path(fabric_file) if fabric_file else None,
                builder_exec=Path(builder_exec) if builder_exec else None,
                generated_fabric_file=Path(generated_fabric_file)
                if generated_fabric_file
                else None,
                fcc_exec=Path(fcc_exec),
            )
        )
    return cases


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def render_run_script(case: UnitCase, repo_root: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(case.output_dir))}",
    ]
    if case.builder_exec and case.generated_fabric_file:
        lines.append(
            shell_join([str(case.builder_exec), str(case.generated_fabric_file)])
        )
        adg_path = case.generated_fabric_file
    elif case.fabric_file:
        adg_path = case.fabric_file
    else:
        raise RuntimeError(f"case {case.name} has no ADG source")

    lines.append(
        shell_join(
            [
                str(case.fcc_exec),
                "--adg",
                str(adg_path),
                "--dfg",
                str(case.dfg_file),
                "-o",
                str(case.output_dir),
            ]
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_run_script(case: UnitCase, repo_root: Path) -> Path:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd = case.output_dir / "run.cmd"
    run_cmd.write_text(render_run_script(case, repo_root), encoding="utf-8")
    run_cmd.chmod(0o755)
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


def run_case(case: UnitCase, repo_root: Path, timeout_sec: int) -> UnitResult:
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
            return UnitResult(case=case, status="timeout", return_code=124)

    if return_code == 0:
        return UnitResult(case=case, status="pass", return_code=0)
    return UnitResult(case=case, status="fail", return_code=return_code)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    selected = set(args.tests)

    cases = load_cases(manifest_path)
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
    for line in sorted(failure_lines):
        print(line)

    return 0 if failed == 0 and timed_out == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

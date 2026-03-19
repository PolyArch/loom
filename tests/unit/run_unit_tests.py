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


def get_case_suffix(case: UnitCase) -> str:
    unit_name = case.dfg_file.parent.name
    if case.name == unit_name:
        return ""
    prefix = unit_name + "-"
    if case.name.startswith(prefix):
        return case.name[len(prefix):]
    return case.name


def get_expect_fail_file(case: UnitCase) -> Path | None:
    unit_dir = case.dfg_file.parent
    suffix = get_case_suffix(case)
    candidates = []
    if suffix:
        candidates.append(unit_dir / f"expect-fail-{suffix}.txt")
    candidates.append(unit_dir / "expect-fail.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def load_expect_fail_patterns(case: UnitCase) -> list[str] | None:
    path = get_expect_fail_file(case)
    if path is None:
        return None
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        patterns.append(text)
    return patterns


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


def get_adg_path(case: UnitCase) -> Path:
    if case.generated_fabric_file is not None:
        return case.generated_fabric_file
    if case.fabric_file is not None:
        return case.fabric_file
    raise RuntimeError(f"case {case.name} has no ADG source")


def normalize_adg_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".fabric"):
        return stem[:-7]
    return stem


def create_legacy_artifact_aliases(case: UnitCase) -> list[Path]:
    dfg_stem = case.dfg_file.stem
    adg_stem = normalize_adg_stem(get_adg_path(case))
    mixed_base = case.output_dir / f"{dfg_stem}.{adg_stem}"
    legacy_base = case.output_dir / dfg_stem
    if mixed_base == legacy_base:
        return []

    created: list[Path] = []
    for suffix in (
        ".config.bin",
        ".config.json",
        ".config.h",
        ".map.json",
        ".map.txt",
        ".viz.html",
        ".sim.trace",
        ".sim.stat",
        ".sim.setup.json",
    ):
        src = Path(f"{mixed_base}{suffix}")
        dst = Path(f"{legacy_base}{suffix}")
        if not src.exists() or dst.exists():
            continue
        try:
            dst.symlink_to(src.name)
        except OSError:
            shutil.copyfile(src, dst)
        created.append(dst)
    return created


def render_run_script(case: UnitCase, repo_root: Path) -> str:
    extra_args = load_fcc_args(case)
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

    cmd = [
        str(case.fcc_exec),
        "--adg",
        str(adg_path),
        "--dfg",
        str(case.dfg_file),
        "-o",
        str(case.output_dir),
    ]
    cmd.extend(extra_args)
    lines.append(shell_join(cmd))
    lines.append("")
    return "\n".join(lines)


def load_fcc_args(case: UnitCase) -> List[str]:
    args_file = case.dfg_file.parent / "fcc.args"
    if not args_file.exists():
        return []
    tokens: List[str] = []
    for line in args_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens.extend(shlex.split(stripped))
    return tokens


def run_optional_checker(case: UnitCase, repo_root: Path, timeout_sec: int,
                         run_out: Path, run_err: Path) -> tuple[str, int]:
    unit_dir = case.dfg_file.parent
    checker_py = unit_dir / "check.py"
    checker_sh = unit_dir / "check.sh"

    if checker_py.exists():
        cmd = [sys.executable, str(checker_py), case.name, str(case.output_dir)]
    elif checker_sh.exists():
        cmd = [str(checker_sh), case.name, str(case.output_dir)]
    else:
        return ("pass", 0)

    with run_out.open("a", encoding="utf-8") as out_handle, run_err.open(
        "a", encoding="utf-8"
    ) as err_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=out_handle,
            stderr=err_handle,
            start_new_session=True,
        )
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc)
            return ("timeout", 124)

    if return_code == 0:
        return ("pass", 0)
    return ("fail", return_code)


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
    expect_fail_patterns = load_expect_fail_patterns(case)

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

    if expect_fail_patterns is not None:
        if return_code == 0:
            with run_err.open("a", encoding="utf-8") as err_handle:
                err_handle.write(
                    "\nExpected this unit case to fail, but it succeeded.\n"
                )
            return UnitResult(case=case, status="fail", return_code=0)

        combined = run_out.read_text(encoding="utf-8") + "\n" + run_err.read_text(
            encoding="utf-8"
        )
        missing = [pattern for pattern in expect_fail_patterns if pattern not in combined]
        if missing:
            with run_err.open("a", encoding="utf-8") as err_handle:
                err_handle.write(
                    "\nExpected failure patterns were not found:\n"
                )
                for pattern in missing:
                    err_handle.write(f"  - {pattern}\n")
            return UnitResult(case=case, status="fail", return_code=return_code)
        return UnitResult(case=case, status="pass", return_code=0)

    if return_code == 0:
        alias_paths = create_legacy_artifact_aliases(case)
        try:
            checker_status, checker_code = run_optional_checker(
                case, repo_root, timeout_sec, run_out, run_err
            )
        finally:
            for alias in alias_paths:
                try:
                    alias.unlink()
                except FileNotFoundError:
                    pass
        if checker_status == "timeout":
            with run_err.open("a", encoding="utf-8") as err_handle:
                err_handle.write(
                    f"\nTimed out after {timeout_sec} seconds while running unit checker.\n"
                )
            return UnitResult(case=case, status="timeout", return_code=124)
        if checker_status == "fail":
            with run_err.open("a", encoding="utf-8") as err_handle:
                err_handle.write(
                    f"\nUnit checker failed with exit code {checker_code}.\n"
                )
            return UnitResult(case=case, status="fail", return_code=checker_code)
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

#!/usr/bin/env python3
"""Regenerate the exact host oracles of the Application portfolio manifest.

The manifest at `test/applications/manifest.json` pins, for every input row, an
oracle entry and its SHA-256. Whenever a row's selected extent or generated
input data changes, both have to be re-derived from one native run of that row.
This script performs exactly the host compile and invocation that
`loom-application-host-run` performs (`lib/Application/HostRunner.cpp`), writes
the observed stdout into the declared oracle entry, and updates the declared
digest in place. It never relaxes an oracle: a row whose program exits nonzero
is reported and left untouched.

  python3 scripts/regenerate_application_oracles.py --application gapbs-pagerank

With no selection it regenerates every exact-oracle row it can run. Rows with
cached inputs additionally need `--cache-root` (the directory the portfolio
cache paths are relative to).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "test/applications/manifest.json"
HOST_EXECUTION_DEFINE = "-DLOOM_APPLICATION_HOST_EXECUTION=1"
COMPILERS = {"c": "clang", "c++": "clang++"}


def compile_and_run(
    application: dict[str, Any],
    row: dict[str, Any],
    repository_root: Path,
    cache_root: Path | None,
    workspace: Path,
) -> str:
    build = application["build"]
    language = build["language"]
    compiler = COMPILERS[language]
    if shutil.which(compiler) is None:
        raise RuntimeError(f"cannot resolve host compiler '{compiler}'")
    source_root = repository_root / application["source"]["root"]
    executable = workspace / "application"
    command = [
        compiler,
        f"-working-directory={repository_root}",
        *build["compiler_options"],
        *row["compiler_options"],
        HOST_EXECUTION_DEFINE,
        "-x",
        language,
        *[str(source_root / source) for source in build["sources"]],
        *build["link_options"],
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, cwd=repository_root)

    arguments = [str(executable)]
    selected_cache = row["cached_inputs"]
    if selected_cache:
        if cache_root is None:
            raise RuntimeError(
                f"{application['identity']}/{row['name']} selects cached inputs; "
                "rerun with --cache-root"
            )
        declared = {entry["logical_name"]: entry["path"] for entry in
                    application["cached_inputs"]}
        arguments += [str(cache_root / declared[name]) for name in selected_cache]
        arguments += [
            str(row["profile"]["warmup_samples"]),
            str(row["profile"]["measured_samples"]),
        ]
    completed = subprocess.run(
        arguments, check=False, cwd=repository_root, stdout=subprocess.PIPE
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{application['identity']}/{row['name']} exited "
            f"{completed.returncode}; its own oracle rejected the run"
        )
    return completed.stdout.decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", action="append", default=[])
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    regenerated: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for application in manifest["applications"]:
            if arguments.application and (
                application["identity"] not in arguments.application
            ):
                continue
            for row in application["inputs"]:
                if arguments.input and row["name"] not in arguments.input:
                    continue
                oracle = row["oracle"]
                if oracle["kind"] != "exact":
                    continue
                output = compile_and_run(
                    application,
                    row,
                    arguments.repository_root.resolve(),
                    arguments.cache_root.resolve() if arguments.cache_root else None,
                    workspace,
                )
                entry = arguments.repository_root.resolve() / oracle["entry"]
                entry.write_text(output, encoding="utf-8")
                digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
                if digest != oracle["sha256"]:
                    manifest_text = manifest_text.replace(oracle["sha256"], digest)
                regenerated.append(
                    f"{application['identity']}/{row['name']}: {digest}"
                )
    MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
    for line in regenerated:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

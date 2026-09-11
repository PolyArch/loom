#!/usr/bin/env python3
"""Regenerate the exact host oracles of the Application portfolio manifest.

The manifest at `test/applications/manifest.json` pins, for every input row, an
oracle entry and its SHA-256. Whenever a row's selected extent or generated
input data changes, both have to be re-derived from one native run of that row.
This script performs exactly the host compile and invocation that
`loom-application-host-run` performs (`lib/Application/HostRunner.cpp`), writes
the observed stdout into the declared oracle entry, and updates that row's
declared digest in place. It never relaxes an oracle: a row whose program fails
to compile or exits nonzero aborts the run with its own diagnostic, and each
regenerated row is committed to the manifest before the next row starts, so an
abort never leaves an entry disagreeing with its declared digest.

  python3 scripts/regenerate_application_oracles.py --application gapbs-pagerank

With no selection it regenerates every exact-oracle row it can run. Rows with
cached inputs additionally need `--cache-root` (the directory the portfolio
cache paths are relative to).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    cache_root: Path | None,
    workspace: Path,
) -> str:
    build = application["build"]
    language = build["language"]
    compiler = COMPILERS[language]
    if shutil.which(compiler) is None:
        raise RuntimeError(f"cannot resolve host compiler '{compiler}'")
    source_root = REPOSITORY_ROOT / application["source"]["root"]
    executable = workspace / "application"
    command = [
        compiler,
        f"-working-directory={REPOSITORY_ROOT}",
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
    # The bounded host runner runs both stages under LC_ALL=C so formatted
    # output cannot depend on the invoking locale.
    environment = dict(os.environ, LC_ALL="C")
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT, env=environment)

    arguments = [str(executable)]
    selected_cache = row["cached_inputs"]
    if selected_cache:
        if cache_root is None:
            raise RuntimeError(
                f"{application['identity']}/{row['name']} selects cached inputs; "
                "rerun with --cache-root"
            )
        declared = {
            entry["logical_name"]: entry["path"]
            for entry in application["cached_inputs"]
        }
        arguments += [str(cache_root / declared[name]) for name in selected_cache]
        arguments += [
            str(row["profile"]["warmup_samples"]),
            str(row["profile"]["measured_samples"]),
        ]
    completed = subprocess.run(
        arguments,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{application['identity']}/{row['name']} exited "
            f"{completed.returncode}; its own oracle rejected the run"
        )
    return completed.stdout.decode("utf-8")


def rewrite_declared_digest(text: str, entry: str, digest: str) -> str:
    """Replace exactly the digest of the oracle object naming `entry`."""
    pattern = re.compile(
        r'("entry": "' + re.escape(entry) + r'",\s*\n\s*"sha256": ")[0-9a-f]{64}(")'
    )
    rewritten, count = pattern.subn(r"\g<1>" + digest + r"\g<2>", text)
    if count != 1:
        raise RuntimeError(f"oracle entry '{entry}' is not declared exactly once")
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", action="append", default=[])
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--cache-root", type=Path)
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cache_root = arguments.cache_root.resolve() if arguments.cache_root else None
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
                output = compile_and_run(application, row, cache_root, workspace)
                digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
                (REPOSITORY_ROOT / oracle["entry"]).write_text(
                    output, encoding="utf-8"
                )
                MANIFEST_PATH.write_text(
                    rewrite_declared_digest(
                        MANIFEST_PATH.read_text(encoding="utf-8"),
                        oracle["entry"],
                        digest,
                    ),
                    encoding="utf-8",
                )
                print(f"{application['identity']}/{row['name']}: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

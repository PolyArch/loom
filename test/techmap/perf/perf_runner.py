#!/usr/bin/env python3
"""Single-point isolated performance runner for the loom partitioner.

For a given (algo, n, seed, threads) point this script:
  1. Generates the synthetic input (cached on disk between runs).
  2. Builds an on-the-fly YAML config selecting the algorithm and threads.
  3. Pins to core 0 via `taskset -c 0` (and `nice -n -5` if root).
  4. Invokes `loom <input> -loom-partition-graph-into-subgraphs="config=..."`
     K times sequentially (no cross-run parallelism).
  5. Reports median, p95, min, max wall-clock per run in milliseconds, plus
     a `cost` proxy parsed from the loom output (count of `dataflow.subgraph`
     occurrences).

Stdout format (FileCheck friendly):
  PERF: ALGO=<a> N=<n> threads=<t> median_ms=<x> p95_ms=<y> min_ms=<z> max_ms=<w> cost=<c>
  PERF: PASS
  (or)
  PERF: FAIL median_ms=<x> exceeds threshold=<thr>

Per-run JSON lines are also emitted on stdout (one per run) prefixed
with `JSON: ` so the human-readable PERF lines remain unambiguous.

Honors LOOM_PERF_TIMEOUT_S env var (default 300s) per loom invocation.
"""

import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import fcntl  # POSIX-only; perf tests are gated to platforms with taskset.
except ImportError:
    fcntl = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def find_loom() -> str:
    """Locate the `loom` binary.

    Search order:
      1. $LOOM_BIN if set.
      2. Walk up from this script for a `build/tools/loom/loom`.
      3. PATH lookup via shutil.which.
    """
    env_bin = os.environ.get("LOOM_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    cur = HERE
    for _ in range(8):
        cand = cur / "build" / "tools" / "loom" / "loom"
        if cand.is_file():
            return str(cand)
        if cur.parent == cur:
            break
        cur = cur.parent

    on_path = shutil.which("loom")
    if on_path:
        return on_path

    raise FileNotFoundError(
        "could not locate loom binary; set LOOM_BIN or build the repo")


def find_python() -> str:
    return sys.executable or "python3"


def repo_temp_dir(name: str) -> Path:
    root = Path(os.environ.get("LOOM_TEMP_DIR", ROOT / "temp"))
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    base = os.environ.get("LOOM_PERF_CACHE")
    if base:
        d = Path(base)
    else:
        d = repo_temp_dir("techmap-perf-cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_synth(n: int, seed: int) -> Path:
    """Generate the synthetic .mlir if not already cached."""
    out = cache_dir() / f"synth_n{n}_s{seed}.mlir"
    if out.is_file() and out.stat().st_size > 0:
        return out
    gen = HERE / "gen_synth.py"
    cmd = [find_python(), str(gen),
           "--n", str(n), "--seed", str(seed), "--out", str(out)]
    subprocess.run(cmd, check=True)
    return out


def write_config(algo: str, threads: int, dst: Path) -> None:
    """Build a minimal YAML config selecting algorithm and threads."""
    lines = ["techmap:"]
    lines.append(f"  algorithm: {algo}")
    if threads > 0:
        lines.append(f"  threads: {threads}")
    # threads == 0 means default (do not emit the key).
    dst.write_text("\n".join(lines) + "\n")


def is_root() -> bool:
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False


def perf_lock_dir() -> Path:
    """Per-machine directory holding one lockfile per claimable core."""
    base = os.environ.get("LOOM_PERF_LOCK_DIR")
    d = Path(base) if base else repo_temp_dir("techmap-perf-locks")
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidate_perf_cores() -> list:
    """Return the cores eligible for perf pinning.

    Honors $LOOM_PERF_CORES (comma-separated list) when set. Otherwise
    uses every core in the process's CPU affinity mask, dropping core 0
    when the mask offers more than one option (core 0 commonly handles
    interrupts on Linux). Falls back to [0] if nothing else is
    available.
    """
    env = os.environ.get("LOOM_PERF_CORES")
    if env:
        cores = []
        for tok in env.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                cores.append(int(tok))
            except ValueError:
                pass
        if cores:
            return cores

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    if not affinity:
        return [0]
    if len(affinity) > 1 and 0 in affinity:
        affinity = [c for c in affinity if c != 0]
    return affinity


@contextlib.contextmanager
def claim_exclusive_core():
    """Acquire an exclusive lock on one of the candidate cores.

    Yields the core index that was claimed. The lock is held for the
    duration of the with-block via flock, so concurrent perf runners
    each end up on a distinct core. If no flock support is available
    (or no core could be claimed without blocking) falls back to the
    first candidate core without locking.
    """
    cores = candidate_perf_cores()
    if not cores or fcntl is None:
        yield cores[0] if cores else 0
        return

    lock_dir = perf_lock_dir()
    fhs = []
    try:
        for core in cores:
            path = lock_dir / f"core{core}.lock"
            fh = open(path, "w")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            fhs.append((core, fh))
            try:
                yield core
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
            return

        # All candidate cores busy: block on the first one rather than
        # racing without isolation.
        path = lock_dir / f"core{cores[0]}.lock"
        fh = open(path, "w")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fhs.append((cores[0], fh))
            try:
                yield cores[0]
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            if fh and not fh.closed:
                fh.close()
    finally:
        for _, fh in fhs:
            if not fh.closed:
                fh.close()


def build_pinned_cmd(loom: str, src: Path, cfg: Path, core: int) -> list:
    """Wrap the loom invocation in taskset/nice pinned to `core`.

    `nice -n -5` requires elevated privileges; skip it otherwise.
    """
    inner = [loom, str(src),
             f"-loom-partition-graph-into-subgraphs=config={cfg}"]
    pre = []
    if shutil.which("taskset"):
        pre = ["taskset", "-c", str(core)]
        if is_root() and shutil.which("nice"):
            pre += ["nice", "-n", "-5"]
    return pre + inner


def run_once(cmd: list, timeout_s: float) -> tuple:
    """Run loom once. Returns (elapsed_ms, stdout_text)."""
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    t1 = time.perf_counter()
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"loom failed with exit code {res.returncode}")
    return (t1 - t0) * 1000.0, res.stdout.decode("utf-8", errors="replace")


def percentile(sorted_vals, pct: float) -> float:
    """Linear-interpolated percentile on a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def fmt_ms(x: float) -> str:
    return f"{x:.3f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--algo", required=True,
                   choices=["greedy", "list", "beam", "sa", "ilp"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=0,
                   help="Worker threads. 0 = use default (omit the key).")
    p.add_argument("--runs", type=int, default=7,
                   help="Number of timed runs (default 7).")
    p.add_argument("--max-median-ms", type=float, default=None,
                   help="Optional median threshold; emit PERF: FAIL and "
                        "exit 1 if median exceeds it.")
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed warmup runs to discount cache effects.")
    args = p.parse_args(argv)

    if args.runs < 1:
        print("error: --runs must be >= 1", file=sys.stderr)
        return 2

    timeout_s = float(os.environ.get("LOOM_PERF_TIMEOUT_S", "300"))

    loom = find_loom()
    src = ensure_synth(args.n, args.seed)

    with tempfile.TemporaryDirectory(prefix="loom-perf-run-", dir=repo_temp_dir("test-runs")) as td, claim_exclusive_core() as core:
        cfg = Path(td) / "perf.yaml"
        write_config(args.algo, args.threads, cfg)
        cmd = build_pinned_cmd(loom, src, cfg, core)

        # Warmup: untimed; primes filesystem and CPU caches.
        for _ in range(max(0, args.warmup)):
            run_once(cmd, timeout_s)

        timings = []
        cost = 0
        for i in range(args.runs):
            ms, out = run_once(cmd, timeout_s)
            timings.append(ms)
            # Cost proxy = count of `dataflow.subgraph` op occurrences.
            run_cost = out.count("dataflow.subgraph")
            cost = run_cost  # last-run cost; deterministic so all equal.
            print("JSON: " + json.dumps({
                "algo": args.algo,
                "n": args.n,
                "seed": args.seed,
                "threads": args.threads,
                "run_index": i,
                "elapsed_ms": ms,
                "cost": run_cost,
            }))

    timings_sorted = sorted(timings)
    median_ms = statistics.median(timings_sorted)
    p95_ms = percentile(timings_sorted, 0.95)
    min_ms = timings_sorted[0]
    max_ms = timings_sorted[-1]

    print(
        f"PERF: ALGO={args.algo} N={args.n} threads={args.threads} "
        f"median_ms={fmt_ms(median_ms)} p95_ms={fmt_ms(p95_ms)} "
        f"min_ms={fmt_ms(min_ms)} max_ms={fmt_ms(max_ms)} cost={cost}"
    )

    if args.max_median_ms is not None and median_ms > args.max_median_ms:
        print(
            f"PERF: FAIL median_ms={fmt_ms(median_ms)} "
            f"exceeds threshold={fmt_ms(args.max_median_ms)}"
        )
        return 1

    print("PERF: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

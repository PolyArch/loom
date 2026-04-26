#!/usr/bin/env python3
"""Cross-algorithm bench reporting harness for the loom partitioner.

Runs one (algo, N, threads) data point at a time -- strictly serial across
points -- by invoking ``perf_runner.py`` as a subprocess. Aggregates the
``JSON: ...`` lines and the ``PERF: ALGO=...`` summary line into stable
artifacts under ``test/techmap/perf/results/<short-git-rev>/``:

  - ``bench.csv``         tabular results (one row per data point)
  - ``bench.md``          rendered Markdown grouped by algorithm
  - ``bench.svg``         log-log line plot (median_ms vs N, one line per algo)
  - ``bench.summary.txt`` mirror of the textual summary printed to stdout

Per-data-point safety cap is honored via ``LOOM_PERF_BUDGET_S`` (default
600 s); the inner ``perf_runner`` invocation is timed out at that bound.

Large-N points (N>=2000) only run when ``LOOM_PERF=long`` is in env.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent

# Try matplotlib once up-front; SVG generation is optional.
try:
    import matplotlib  # noqa: F401
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # noqa: F401
    HAVE_MPL = True
    MPL_ERR = None
except Exception as exc:  # pragma: no cover - best-effort fallback
    HAVE_MPL = False
    MPL_ERR = str(exc)

PERF_LINE_RE = re.compile(
    r"^PERF:\s+ALGO=(?P<algo>\S+)\s+N=(?P<n>\d+)\s+threads=(?P<threads>\d+)\s+"
    r"median_ms=(?P<median_ms>[\d.]+)\s+p95_ms=(?P<p95_ms>[\d.]+)\s+"
    r"min_ms=(?P<min_ms>[\d.]+)\s+max_ms=(?P<max_ms>[\d.]+)\s+"
    r"cost=(?P<cost>\d+)\s*$"
)

LARGE_N_THRESHOLD = 2000


def short_git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii", errors="replace").strip() or "unknown"
    except Exception:
        return "unknown"


def parse_csv_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list:
    return [int(x) for x in parse_csv_list(s)]


def host_summary() -> str:
    cpu_count = os.cpu_count() or 0
    uname = platform.uname()
    return f"cpu_count={cpu_count}, kernel={uname.system} {uname.release}"


def is_long_perf() -> bool:
    return os.environ.get("LOOM_PERF", "").strip().lower() == "long"


def run_point(algo, n, threads, seed, runs, warmup, budget_s):
    """Run one (algo, N, threads) data point via perf_runner.py subprocess.

    Returns a dict on success, or {"skipped": str/"error": str} on failure.
    """
    cmd = [
        sys.executable,
        str(HERE / "perf_runner.py"),
        "--algo", algo,
        "--n", str(n),
        "--seed", str(seed),
        "--threads", str(threads),
        "--runs", str(runs),
        "--warmup", str(warmup),
    ]
    env = os.environ.copy()
    # Forward to perf_runner's per-invocation timeout so it kills hung loom.
    env.setdefault("LOOM_PERF_TIMEOUT_S", str(int(budget_s)))
    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=budget_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {budget_s:.0f}s"}
    elapsed = time.perf_counter() - t0

    stdout = res.stdout.decode("utf-8", errors="replace")
    stderr = res.stderr.decode("utf-8", errors="replace")

    if res.returncode != 0:
        return {
            "error": (f"perf_runner exit={res.returncode} "
                      f"stderr={stderr.strip()[:200]}"),
        }

    summary = None
    json_lines = []
    for line in stdout.splitlines():
        if line.startswith("JSON: "):
            try:
                json_lines.append(json.loads(line[len("JSON: "):]))
            except json.JSONDecodeError:
                pass
        else:
            m = PERF_LINE_RE.match(line)
            if m:
                summary = m.groupdict()

    if summary is None:
        return {"error": "no PERF summary line in perf_runner output"}

    return {
        "algo": summary["algo"],
        "n": int(summary["n"]),
        "threads": int(summary["threads"]),
        "runs": runs,
        "median_ms": float(summary["median_ms"]),
        "p95_ms": float(summary["p95_ms"]),
        "min_ms": float(summary["min_ms"]),
        "max_ms": float(summary["max_ms"]),
        "cost": int(summary["cost"]),
        "json_lines": json_lines,
        "wall_s": elapsed,
    }


def write_csv(rows, dst: Path) -> None:
    fields = ["algo", "n", "threads", "runs",
              "median_ms", "p95_ms", "min_ms", "max_ms", "cost"]
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def write_markdown(rows, dst: Path, rev: str, ts_iso: str,
                   algos_in_order, host: str,
                   skipped_notes: list, error_notes: list) -> None:
    lines = []
    lines.append("# Loom partitioner bench results")
    lines.append("")
    lines.append(f"git rev: {rev}")
    lines.append(f"timestamp: {ts_iso}")
    lines.append(f"host: {host}")
    lines.append("")

    # Group rows by algo, preserving the user-specified algo order.
    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algo"], []).append(r)

    for algo in algos_in_order:
        if algo not in by_algo:
            continue
        lines.append(f"## {algo}")
        lines.append("")
        lines.append("| N    | threads | runs | median_ms | p95_ms | cost |")
        lines.append("|------|---------|------|-----------|--------|------|")
        # Sort by (N, threads) for stable rendering.
        for r in sorted(by_algo[algo], key=lambda x: (x["n"], x["threads"])):
            lines.append(
                f"| {r['n']} | {r['threads']} | {r['runs']} | "
                f"{r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['cost']} |"
            )
        lines.append("")

    if skipped_notes:
        lines.append("## Skipped")
        lines.append("")
        for note in skipped_notes:
            lines.append(f"- {note}")
        lines.append("")

    if error_notes:
        lines.append("## Errors")
        lines.append("")
        for note in error_notes:
            lines.append(f"- {note}")
        lines.append("")

    dst.write_text("\n".join(lines))


def write_svg(rows, dst: Path, rev: str, algos_in_order) -> bool:
    """Render the log-log plot. Returns True iff an SVG was written."""
    if not HAVE_MPL:
        return False
    import matplotlib.pyplot as plt

    # Group default-threads rows (threads == 0) per algo.
    by_algo = {}
    for r in rows:
        if r["threads"] != 0:
            continue
        by_algo.setdefault(r["algo"], []).append(r)

    if not by_algo:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for algo in algos_in_order:
        if algo not in by_algo:
            continue
        pts = sorted(by_algo[algo], key=lambda x: x["n"])
        xs = [p["n"] for p in pts]
        ys = [p["median_ms"] for p in pts]
        ax.plot(xs, ys, marker="o", label=algo)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (graph body ops)")
    ax.set_ylabel("median wall-clock (ms)")
    ax.set_title(f"Loom partitioner scaling (rev {rev})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(dst), format="svg")
    plt.close(fig)
    return True


def make_summary_text(rows, rev: str, ts_iso: str, host: str,
                      skipped_notes, error_notes) -> str:
    out = []
    out.append(f"Bench results for git rev {rev}")
    out.append(f"timestamp: {ts_iso}")
    out.append(f"host: {host}")
    out.append("")
    out.append(
        "algo     n     threads runs  median_ms   p95_ms   cost"
    )
    for r in sorted(rows, key=lambda x: (x["algo"], x["n"], x["threads"])):
        out.append(
            f"{r['algo']:<8} {r['n']:<5} {r['threads']:<7} "
            f"{r['runs']:<5} {r['median_ms']:>9.3f}  "
            f"{r['p95_ms']:>7.3f}  {r['cost']}"
        )
    if skipped_notes:
        out.append("")
        out.append("Skipped:")
        for n in skipped_notes:
            out.append(f"  - {n}")
    if error_notes:
        out.append("")
        out.append("Errors:")
        for n in error_notes:
            out.append(f"  - {n}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None,
                   help="Output dir. Defaults to "
                        "test/techmap/perf/results/<short-git-rev>/.")
    p.add_argument("--ns", default="100,200,500,1000,2000,5000",
                   help="Comma-separated list of N values.")
    p.add_argument("--algos", default="greedy,list,beam,sa,ilp",
                   help="Comma-separated list of algorithms.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--cold-warmup", type=int, default=1,
                   help="Untimed warmup runs per data point.")
    args = p.parse_args(argv)

    rev = short_git_rev()
    out_dir = (Path(args.out)
               if args.out
               else REPO_ROOT / "test" / "techmap" / "perf" /
                    "results" / rev)
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = parse_int_list(args.ns)
    algos = parse_csv_list(args.algos)

    budget_s = float(os.environ.get("LOOM_PERF_BUDGET_S", "600"))
    long_ok = is_long_perf()
    ts_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    host = host_summary()

    rows = []
    skipped_notes = []
    error_notes = []

    # Strictly serial across (algo, N, threads) data points.
    for algo in algos:
        for n in ns:
            if n >= LARGE_N_THRESHOLD and not long_ok:
                msg = (f"{algo} N={n}: skipped (gated by LOOM_PERF=long)")
                print(f"# {msg}")
                skipped_notes.append(msg)
                continue
            for threads in (1, 0):
                tag = f"{algo} N={n} threads={threads}"
                print(f"# running {tag} ...", flush=True)
                res = run_point(
                    algo=algo, n=n, threads=threads,
                    seed=args.seed, runs=args.runs,
                    warmup=args.cold_warmup, budget_s=budget_s,
                )
                if "error" in res:
                    msg = f"{tag}: {res['error']}"
                    print(f"# error: {msg}")
                    error_notes.append(msg)
                    continue
                print(
                    f"#   median_ms={res['median_ms']:.3f} "
                    f"p95_ms={res['p95_ms']:.3f} "
                    f"cost={res['cost']} (wall {res['wall_s']:.1f}s)",
                    flush=True,
                )
                rows.append(res)

    csv_path = out_dir / "bench.csv"
    md_path = out_dir / "bench.md"
    svg_path = out_dir / "bench.svg"
    summary_path = out_dir / "bench.summary.txt"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path, rev, ts_iso, algos, host,
                   skipped_notes, error_notes)

    svg_written = write_svg(rows, svg_path, rev, algos) if rows else False
    if not svg_written:
        # Remove any stale SVG from a previous run before writing the stub.
        if svg_path.exists():
            try:
                svg_path.unlink()
            except OSError:
                pass
        stub = out_dir / "bench.svg.SKIPPED"
        if HAVE_MPL:
            reason = ("no default-threads rows to plot; "
                      "SVG generation skipped.")
        else:
            reason = (f"matplotlib unavailable ({MPL_ERR}); "
                      "install matplotlib to enable SVG generation. "
                      "matplotlib is intentionally NOT in requirements.txt.")
        stub.write_text(reason + "\n")
    else:
        # Make sure no stale SKIPPED file lingers.
        stale = out_dir / "bench.svg.SKIPPED"
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass

    summary_text = make_summary_text(rows, rev, ts_iso, host,
                                     skipped_notes, error_notes)
    summary_path.write_text(summary_text)

    print()
    print(summary_text, end="")

    if error_notes:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

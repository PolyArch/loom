#!/usr/bin/env python3
"""E05: Failure Analysis.

Post-processes E04 mapping results to classify and analyze all mapping
failures. Reads E04 mapping_matrix.csv and run output logs to determine
failure categories.

Output: out/experiments/E05/failure_analysis.csv
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
E04_DIR = REPO_ROOT / "out" / "experiments" / "E04"
E04_RUNS = E04_DIR / "runs"
OUTPUT_DIR = REPO_ROOT / "out" / "experiments" / "E05"

# Failure classification patterns applied to combined stdout+stderr
FAILURE_PATTERNS = [
    # Order matters: more specific patterns first
    ("TECHMAP_FAIL", "missing_fu",
     re.compile(r"tech.?map.*(?:no candidate|no match|unsupported op|cannot map)", re.I)),
    ("TECHMAP_FAIL", "coverage_low",
     re.compile(r"techmap.*coverage.*(?:0\.\d{1,2}[^0-9]|insufficient)", re.I)),
    ("COMPILE_FAIL", "frontend_error",
     re.compile(r"(?:clang|llvm|frontend).*error", re.I)),
    ("COMPILE_FAIL", "dfg_conversion",
     re.compile(r"(?:scf.to.dfg|cf.to.scf|llvm.to.cf).*(?:fail|error)", re.I)),
    ("COMPILE_FAIL", "unsupported_construct",
     re.compile(r"unsupported.*(?:operation|construct|type)", re.I)),
    ("PLACEMENT_FAIL", "pe_insufficient",
     re.compile(r"placement.*(?:fail|insufficient|no.valid)", re.I)),
    ("ROUTING_FAIL", "congestion",
     re.compile(r"(?:routing|route).*(?:congesti|fail|unrouted)", re.I)),
    ("ROUTING_FAIL", "unrouted_edges",
     re.compile(r"(?:unrouted|failed).*(edge|path)", re.I)),
    ("II_FAIL", "recurrence_bound",
     re.compile(r"(?:recurrence|ii|initiation.interval).*(?:bound|exceed)", re.I)),
    ("TIMEOUT", "budget_exceeded",
     re.compile(r"(?:timeout|budget.*exceeded|timed.out)", re.I)),
]


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_e04_results() -> list[dict]:
    csv_path = E04_DIR / "mapping_matrix.csv"
    if not csv_path.exists():
        print(f"ERROR: E04 results not found at {csv_path}", file=sys.stderr)
        print("Run E04 first: python3 scripts/experiments/run_e04_mapping.py",
              file=sys.stderr)
        sys.exit(1)

    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def read_run_logs(core_type: str, kernel: str) -> str:
    """Read combined stdout+stderr from E04 run directory."""
    run_dir = E04_RUNS / core_type / kernel
    combined = ""
    for fname in ["run.stdout", "run.stderr"]:
        log_path = run_dir / fname
        if log_path.exists():
            combined += log_path.read_text(encoding="utf-8", errors="replace")
    return combined


def classify_from_logs(log_text: str) -> tuple[str, str, str]:
    """Classify failure from log text using pattern matching.

    Returns (failure_stage, failure_category, detail).
    """
    for stage, category, pattern in FAILURE_PATTERNS:
        match = pattern.search(log_text)
        if match:
            detail = match.group(0).strip()[:200]
            return stage, category, detail
    return "UNKNOWN", "unclassified", ""


def analyze_map_json(core_type: str, kernel: str) -> dict:
    """Extract additional failure info from map.json if it exists."""
    run_dir = E04_RUNS / core_type / kernel
    adg_stem = {
        "ctrl": "ctrl_core",
        "gp": "gp_core",
        "dsp": "dsp_core",
        "ai": "ai_core",
    }.get(core_type, core_type)

    for pattern in [
        run_dir / f"{kernel}.{adg_stem}.map.json",
        run_dir / f"{kernel}.map.json",
    ]:
        if pattern.exists():
            try:
                with open(pattern, encoding="utf-8") as f:
                    data = json.load(f)
                er = data.get("edge_routings", [])
                routed = sum(1 for e in er if e.get("kind") == "routed")
                unrouted = sum(1 for e in er if e.get("kind") == "unrouted")
                techmap = data.get("techmap", {})
                return {
                    "has_map_json": True,
                    "routed": routed,
                    "unrouted": unrouted,
                    "techmap_coverage": techmap.get("coverage_score", 0.0),
                    "selected_candidates": techmap.get("selected_candidate_count", 0),
                }
            except (json.JSONDecodeError, KeyError):
                pass
    return {"has_map_json": False}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    e04_rows = load_e04_results()
    failed_rows = [r for r in e04_rows if r["mapped"] != "True"]

    print(f"E05: Analyzing {len(failed_rows)} failures from E04")
    print(f"  Total E04 runs: {len(e04_rows)}")
    print(f"  Successful: {len(e04_rows) - len(failed_rows)}")
    print(f"  Failed: {len(failed_rows)}")
    print()

    failure_records = []
    for row in failed_rows:
        kernel = row["kernel"]
        domain = row["domain"]
        core_type = row["core_type"]

        # Use E04's failure classification first
        e04_stage = row.get("failure_stage", "")
        e04_detail = row.get("failure_detail", "")

        # Then try to refine from logs
        log_text = read_run_logs(core_type, kernel)
        log_stage, log_category, log_detail = classify_from_logs(log_text)

        # Get map.json info
        map_info = analyze_map_json(core_type, kernel)

        # Determine final classification
        if e04_stage and e04_stage != "UNKNOWN":
            final_stage = e04_stage
            final_detail = e04_detail
        elif log_stage != "UNKNOWN":
            final_stage = log_stage
            final_detail = log_detail
        else:
            final_stage = "UNKNOWN"
            final_detail = e04_detail or log_detail or "No diagnostic available"

        # Refine with map.json info
        if map_info.get("has_map_json") and map_info.get("unrouted", 0) > 0:
            if final_stage == "UNKNOWN" or final_stage == "COMPILE_FAIL":
                final_stage = "ROUTING_FAIL"
                final_detail = (
                    f"{map_info['unrouted']} unrouted edges, "
                    f"coverage={map_info.get('techmap_coverage', 0.0):.2f}"
                )

        failure_records.append({
            "kernel": kernel,
            "domain": domain,
            "core_type": core_type,
            "failure_stage": final_stage,
            "failure_category": log_category if log_stage != "UNKNOWN" else final_stage.lower(),
            "detail": final_detail,
            "has_map_json": map_info.get("has_map_json", False),
            "unrouted_edges": map_info.get("unrouted", 0),
            "techmap_coverage": map_info.get("techmap_coverage", 0.0),
        })

    # Write CSV
    csv_path = OUTPUT_DIR / "failure_analysis.csv"
    fieldnames = [
        "kernel", "domain", "core_type", "failure_stage",
        "failure_category", "detail",
        "has_map_json", "unrouted_edges", "techmap_coverage",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in failure_records:
            writer.writerow(rec)

    # Write provenance
    prov_path = OUTPUT_DIR / "provenance.json"
    stage_counts = Counter(r["failure_stage"] for r in failure_records)
    domain_counts = Counter(r["domain"] for r in failure_records)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "E05",
            "description": "Failure Analysis of E04 mapping results",
            "git_hash": git_hash,
            "timestamp": timestamp,
            "e04_total_runs": len(e04_rows),
            "e04_failures": len(failed_rows),
            "failure_stage_distribution": dict(stage_counts),
            "failure_domain_distribution": dict(domain_counts),
        }, f, indent=2)

    # Print summary
    print("Failure distribution by category:")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {count}")
    print()
    print("Failures by domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    print()
    print("Failures by core type:")
    core_counts = Counter(r["core_type"] for r in failure_records)
    for ct, count in sorted(core_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")
    print()

    print(f"E05 complete: {len(failure_records)} failures classified")
    print(f"  CSV: {csv_path}")
    print(f"  Provenance: {prov_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

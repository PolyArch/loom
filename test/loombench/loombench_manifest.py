#!/usr/bin/env python3
"""Emit a LoomBench manifest from legacy inventory and import status rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "app"))

import app_import_status  # noqa: E402
import old_app_corpus_inventory  # noqa: E402


CSV_HEADER = [
    "case",
    "source_row",
    "software_root",
    "source_fingerprint",
    "main_source",
    "implementation_sources",
    "headers",
    "feature_tags",
    "import_state",
    "manifest_case",
    "oracle",
    "input_profile",
    "tier_states",
    "owner",
    "reason",
]
VALID_IMPORT_STATES = {"accepted", "deferred", "excluded"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--import-status", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    return parser.parse_args(argv)


def read_csv(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing input CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != expected_header:
        raise SystemExit(f"unexpected CSV header for {path}: {reader.fieldnames}")
    return rows


def split_cell(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def relative_path_text(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def fingerprint_files(root: Path, filenames: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in sorted(filenames):
        path = root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def tier_states(import_state: str) -> dict[str, str]:
    if import_state == "accepted":
        return {
            "source": "pass",
            "raise": "blocked",
            "dataflow": "blocked",
            "cgra_status": "blocked",
        }
    if import_state == "excluded":
        return {
            "source": "unsupported",
            "raise": "unsupported",
            "dataflow": "unsupported",
            "cgra_status": "unsupported",
        }
    return {
        "source": "blocked",
        "raise": "blocked",
        "dataflow": "blocked",
        "cgra_status": "blocked",
    }


def build_cases(
    inventory_rows: list[dict[str, str]],
    import_rows: list[dict[str, str]],
    source_root: Path,
) -> list[dict[str, object]]:
    by_case = {row["case"]: row for row in import_rows}
    cases: list[dict[str, object]] = []
    for inventory in inventory_rows:
        case = inventory["case"]
        import_row = by_case.get(case)
        if import_row is None:
            raise SystemExit(f"missing import status row for legacy case: {case}")
        import_state = import_row["import_state"]
        if import_state not in VALID_IMPORT_STATES:
            raise SystemExit(f"{case}: invalid import_state {import_state!r}")
        source_files = [inventory.get("main_source", "")]
        source_files.extend(split_cell(inventory.get("implementation_sources", "")))
        source_files.extend(split_cell(inventory.get("headers", "")))
        source_files = [item for item in source_files if item]
        cases.append(
            {
                "case": case,
                "source_row": case,
                "software_root": relative_path_text(source_root / case),
                "source_fingerprint": fingerprint_files(source_root / case, source_files),
                "main_source": inventory.get("main_source", ""),
                "implementation_sources": split_cell(inventory.get("implementation_sources", "")),
                "headers": split_cell(inventory.get("headers", "")),
                "feature_tags": split_cell(inventory.get("feature_tags", "")),
                "import_state": import_state,
                "manifest_case": import_row.get("manifest_case", ""),
                "oracle": "legacy_reference",
                "input_profile": "legacy_default",
                "tier_states": tier_states(import_state),
                "owner": import_row.get("owner", "") or "test_migration",
                "reason": import_row.get("reason", ""),
            }
        )
    return cases


def csv_row(case: dict[str, object]) -> dict[str, str]:
    return {
        "case": str(case["case"]),
        "source_row": str(case["source_row"]),
        "software_root": str(case["software_root"]),
        "source_fingerprint": str(case["source_fingerprint"]),
        "main_source": str(case["main_source"]),
        "implementation_sources": ";".join(str(item) for item in case["implementation_sources"]),
        "headers": ";".join(str(item) for item in case["headers"]),
        "feature_tags": ";".join(str(item) for item in case["feature_tags"]),
        "import_state": str(case["import_state"]),
        "manifest_case": str(case["manifest_case"]),
        "oracle": str(case["oracle"]),
        "input_profile": str(case["input_profile"]),
        "tier_states": json.dumps(case["tier_states"], sort_keys=True, separators=(",", ":")),
        "owner": str(case["owner"]),
        "reason": str(case["reason"]),
    }


def write_outputs(output: Path, csv_output: Path | None, cases: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "kind": "loombench_manifest",
        "csv_projection": str(csv_output) if csv_output else "",
        "case_count": len(cases),
        "cases": cases,
    }
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(csv_row(case) for case in cases)


def build_manifest(inventory: Path, import_status_path: Path, source_root: Path) -> list[dict[str, object]]:
    inventory_rows = read_csv(inventory, old_app_corpus_inventory.HEADER)
    import_rows = read_csv(import_status_path, app_import_status.HEADER)
    seen: set[str] = set()
    for row in import_rows:
        case = row["case"]
        if case in seen:
            raise SystemExit(f"duplicate import status case: {case}")
        seen.add(case)
    return build_cases(inventory_rows, import_rows, source_root)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cases = build_manifest(Path(args.inventory), Path(args.import_status), Path(args.source_root))
    write_outputs(Path(args.output), Path(args.csv_output) if args.csv_output else None, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

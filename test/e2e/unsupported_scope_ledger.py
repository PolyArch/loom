#!/usr/bin/env python3
"""Emit unsupported-scope ledger rows from artifact contents."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


DISCOVERY_PATHS = (
    "temp/source-compat-summary.csv",
    "temp/compiler-pipeline-summary.csv",
    "temp/dataflow-primitive-coverage.csv",
    "temp/adg-hardware-summary.csv",
    "temp/pnr-mapping-summary.csv",
    "temp/sim-cycle-summary.csv",
    "temp/rtl-fpa-summary.csv",
    "temp/e2e-demonstrator-summary.csv",
    "temp/dse-candidate-summary.csv",
)

GAP_STATUSES = {"blocked", "unsupported", "skipped"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def discover_artifacts(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(value) for value in explicit]
    return [ROOT / value for value in DISCOVERY_PATHS if (ROOT / value).is_file()]


def case_identity(schema: intermediate_artifacts.CsvSchema, row: dict[str, str]) -> str:
    if schema.kind == "dataflow_primitive_coverage":
        return f"{row.get('workload', '')}:{row.get('primitive', '')}"
    values = [row.get(column, "") for column in schema.identity_columns]
    return ":".join(value for value in values if value) or "unknown"


def rows_for_artifact(path: Path) -> list[dict[str, str]]:
    schema = intermediate_artifacts.schema_for_path(path)
    if schema is None or not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            diagnostic = row.get("diagnostic", "")
            for column in schema.status_columns:
                status = row.get(column, "")
                if status not in GAP_STATUSES:
                    continue
                reason = f"{column}={status}"
                if diagnostic:
                    reason = f"{reason}; {diagnostic}"
                rows.append(
                    {
                        "stage": column,
                        "case": case_identity(schema, row),
                        "artifact": schema.kind,
                        "reason": reason,
                        "owner": "implementation",
                        "blocking_input": str(path),
                    }
                )
    return rows


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paths = discover_artifacts(args.artifact)
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(rows_for_artifact(path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("unsupported_scope", output, rows)
    else:
        intermediate_artifacts.write_csv("unsupported_scope", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

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


GAP_STATUSES = {"blocked", "unsupported", "skipped", "not_run"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def discover_artifacts(explicit: list[str]) -> list[Path]:
    return intermediate_artifacts.discover_artifact_paths(
        ROOT,
        explicit,
        include_unsupported_scope=False,
    )


def case_identity(schema: intermediate_artifacts.CsvSchema, row: dict[str, str]) -> str:
    if schema.kind == "dataflow_primitive_coverage":
        return f"{row.get('workload', '')}:{row.get('primitive', '')}"
    if schema.kind == "cgra_status":
        return f"{row.get('suite', '')}:{row.get('case', '')}:{row.get('source_row', '')}"
    values = [row.get(column, "") for column in schema.identity_columns]
    return ":".join(value for value in values if value) or "unknown"


GapKey = tuple[str, str, str]
StatusEvent = tuple[GapKey, str, dict[str, str] | None]


def status_events_for_artifact(path: Path) -> list[StatusEvent]:
    schema = intermediate_artifacts.schema_for_path(path)
    if schema is None or not path.is_file():
        return []
    events: list[StatusEvent] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            diagnostic = row.get("diagnostic", "")
            for column in schema.status_columns:
                status = row.get(column, "")
                key = (column, case_identity(schema, row), schema.kind)
                if status == "pass":
                    events.append((key, "pass", None))
                    continue
                if status not in GAP_STATUSES:
                    continue
                reason = f"{column}={status}"
                if diagnostic:
                    reason = f"{reason}; {diagnostic}"
                events.append(
                    (
                        key,
                        "gap",
                        {
                            "stage": column,
                            "case": key[1],
                            "artifact": schema.kind,
                            "reason": reason,
                            "owner": "implementation",
                            "blocking_input": str(path),
                        },
                    )
                )
    return events


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paths = discover_artifacts(args.artifact)
    gaps: list[tuple[GapKey, dict[str, str]]] = []
    for path in paths:
        for key, event, row in status_events_for_artifact(path):
            if event == "pass":
                gaps = [(gap_key, gap_row) for gap_key, gap_row in gaps if gap_key != key]
            elif row is not None:
                gaps.append((key, row))
    rows = [row for _, row in gaps]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("unsupported_scope", output, rows)
    else:
        intermediate_artifacts.write_csv("unsupported_scope", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

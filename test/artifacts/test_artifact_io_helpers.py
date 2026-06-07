#!/usr/bin/env python3
"""Regression test for shared artifact IO helpers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import artifact_test_common


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test" / "artifacts"))

import artifact_io_helpers  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO
    with TemporaryDirectory(prefix="loom-artifact-io-") as tmp:
        out_dir = Path(tmp)
        source_csv = out_dir / "source-compat-summary.csv"
        write_csv(
            source_csv,
            [
                {"case": "vecsum", "status": "pass"},
                {"case": "matmul", "status": "unsupported"},
            ],
        )
        if artifact_io_helpers.read_csv(source_csv) != [
            {"case": "vecsum", "status": "pass"},
            {"case": "matmul", "status": "unsupported"},
        ]:
            raise AssertionError("read_csv should preserve CSV rows")
        if artifact_io_helpers.read_csv(out_dir / "missing.csv") != []:
            raise AssertionError("read_csv should return an empty list for missing files")

        manifest = out_dir / "rtl-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "kind": "rtl_manifest",
                    "status": "pass",
                    "source_fabric_adg_identity": "shared_reduction_adg",
                }
            )
            + "\n"
        )
        invalid_json = out_dir / "invalid.json"
        invalid_json.write_text("{")
        list_json = out_dir / "list.json"
        list_json.write_text("[]\n")
        if artifact_io_helpers.read_json(manifest)["kind"] != "rtl_manifest":
            raise AssertionError("read_json should load object JSON")
        if artifact_io_helpers.read_json(out_dir / "missing.json") != {}:
            raise AssertionError("read_json should return an empty dict for missing files")
        try:
            artifact_io_helpers.read_json(invalid_json)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("read_json should preserve strict invalid JSON failures")
        if artifact_io_helpers.read_json_or_empty(invalid_json) != {}:
            raise AssertionError("read_json_or_empty should return an empty dict for invalid JSON")
        if artifact_io_helpers.read_json(list_json) != {}:
            raise AssertionError("read_json should return an empty dict for non-object JSON")

        grouped = artifact_io_helpers.group_paths([source_csv, manifest])
        if artifact_io_helpers.first_path(grouped, "source_compat") != source_csv:
            raise AssertionError(f"group_paths should use artifact kind detection: {grouped}")
        if artifact_io_helpers.first_path(grouped, "runtime_package") is not None:
            raise AssertionError("first_path should return None when a kind is absent")

        if not artifact_io_helpers.hardware_matches("fabric::shared_reduction_adg", "shared_reduction_adg"):
            raise AssertionError("hardware_matches should accept namespace-stripped hardware identities")
        if artifact_io_helpers.hardware_matches("fabric::shared_reduction_adg", "other_adg"):
            raise AssertionError("hardware_matches should reject unrelated hardware identities")
        if artifact_io_helpers.matching_rtl_manifest_path([manifest], "shared_reduction_adg") != manifest:
            raise AssertionError("matching_rtl_manifest_path should match pass RTL manifests by source ADG")

        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "-m",
                "py_compile",
                "test/artifacts/artifact_io_helpers.py",
            ],
            "artifact IO helper py_compile",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

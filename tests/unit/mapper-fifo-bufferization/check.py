#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    out_dir = Path(sys.argv[2])
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    fifo_configs = {entry["name"]: entry for entry in mapping.get("fifo_configs", [])}
    fifo_cfg = fifo_configs.get("fifo_0")
    if fifo_cfg is None:
        raise SystemExit("missing fifo_0 config")
    if not fifo_cfg.get("bypassable"):
        raise SystemExit("fifo_0 should be bypassable")
    if fifo_cfg.get("bypassed"):
        raise SystemExit("fifo_0 should be forced buffered by mapper")
    if fifo_cfg.get("depth") != 4:
        raise SystemExit(f"unexpected fifo depth payload: {fifo_cfg}")

    timing = mapping.get("timing", {})
    if not timing.get("available"):
        raise SystemExit("timing summary missing")
    if timing.get("mapper_selected_buffered_fifo_count") != 1:
        raise SystemExit(f"expected one mapper-selected buffered fifo, got {timing}")
    if timing.get("mapper_selected_buffered_fifo_depths") != [4]:
        raise SystemExit(f"unexpected mapper-selected fifo depths: {timing}")
    if not timing.get("critical_path_edges"):
        raise SystemExit("critical_path_edges should be non-empty")

    search = mapping.get("search", {})
    for key in (
        "placement_seed_lane_count",
        "routed_lane_count",
        "local_repair_attempts",
        "local_repair_successes",
        "route_aware_checkpoint_rescore_passes",
        "route_aware_neighborhood_attempts",
        "fifo_bufferization_accepted_toggles",
        "outer_joint_accepted_rounds",
    ):
        if key not in search:
            raise SystemExit(f"missing search key {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

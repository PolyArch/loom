#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    out_dir = Path(sys.argv[2])
    mapping = json.loads((out_dir / "dfg.fabric.map.json").read_text(encoding="utf-8"))

    fifo_configs = {entry["name"]: entry for entry in mapping.get("fifo_configs", [])}
    fifo_cfg = fifo_configs.get("fifo_0")
    if fifo_cfg is None:
        raise SystemExit("missing fifo_0 config")
    if fifo_cfg.get("depth") != 4:
        raise SystemExit(f"unexpected fifo depth payload: {fifo_cfg}")
    if not fifo_cfg.get("bypassed", False):
        raise SystemExit("recurrence-guard case should leave fifo_0 bypassed")

    timing = mapping.get("timing", {})
    if not timing.get("available"):
        raise SystemExit("timing summary missing")
    if timing.get("mapper_selected_buffered_fifo_count") != 0:
        raise SystemExit(f"recurrence-guard case should reject mapper FIFO buffering: {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

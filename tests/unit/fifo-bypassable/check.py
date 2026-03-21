#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    fifo_slice = slices.get(("fifo_0", "fifo"))
    if fifo_slice is None:
        raise SystemExit("missing fifo slice")
    if not fifo_slice["complete"]:
        raise SystemExit("fifo slice marked incomplete")
    words = config["words"][
        fifo_slice["word_offset"]:fifo_slice["word_offset"] + fifo_slice["word_count"]
    ]
    if words != [1]:
        raise SystemExit(f"unexpected fifo config words: {words}")

    fifo_configs = {entry["name"]: entry for entry in mapping.get("fifo_configs", [])}
    fifo_cfg = fifo_configs.get("fifo_0")
    if fifo_cfg is None:
        raise SystemExit("missing fifo_configs entry")
    if fifo_cfg.get("depth") != 2:
        raise SystemExit(f"unexpected fifo depth payload: {fifo_cfg}")
    if not fifo_cfg["bypassable"] or not fifo_cfg["bypassed"]:
        raise SystemExit(f"unexpected fifo config payload: {fifo_cfg}")

    timing = mapping.get("timing", {})
    if not timing.get("available"):
        raise SystemExit("missing timing summary")
    if timing.get("mapper_selected_buffered_fifo_count") != 0:
        raise SystemExit(f"unexpected mapper-selected buffered fifo count: {timing}")
    if timing.get("mapper_selected_buffered_fifo_depths") not in ([], None):
        raise SystemExit(f"unexpected mapper-selected fifo depths: {timing}")

    routed = [entry for entry in mapping["edge_routings"] if entry["kind"] == "routed"]
    if len(routed) != 1:
        raise SystemExit(f"expected one routed edge, got {routed}")
    if '"kind": "fifo"' not in html:
        raise SystemExit("viz html missing fifo component")
    return 0


if __name__ == "__main__":
    sys.exit(main())

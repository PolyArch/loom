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

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    mem_slice = slices.get(("extmem_0", "extmemory"))
    if mem_slice is None:
        raise SystemExit("missing extmemory slice")
    if not mem_slice["complete"]:
        raise SystemExit("extmemory slice marked incomplete")

    words = config["words"][
        mem_slice["word_offset"]:mem_slice["word_offset"] + mem_slice["word_count"]
    ]
    expected = [1, 0, 1, 0, 2, 1, 1, 2, 0, 2]
    if words != expected:
        raise SystemExit(f"unexpected extmemory addr_offset_table words: {words}")

    memory_regions = mapping.get("memory_regions", [])
    if len(memory_regions) != 1:
        raise SystemExit(f"expected one memory_regions entry, got {memory_regions}")
    region = memory_regions[0]
    if region.get("memory_kind") != "extmemory":
        raise SystemExit(f"expected extmemory region payload, got {region}")
    if region.get("addr_offset_table") != expected:
        raise SystemExit(f"unexpected exported addr_offset_table: {region}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

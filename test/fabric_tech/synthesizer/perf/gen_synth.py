#!/usr/bin/env python3
"""Synthetic tier-A workload generator for synthesizer perf tests.

Emits an MLIR module with N `func.func`s, each carrying a single
`dataflow.subgraph` of identical topology (yield <- bin-op of two block
args), tagged with the same `loom.synth_group`. Body op identity
varies across the N functions but always stays inside the same
hardware-share group, so the synthesizer must converge on a single
`fabric.op [op_list...]` covering all inputs (tier A).

Output is byte-identical for identical (n, seed). Output is written
to the path given by --out, or to stdout if --out is omitted or "-"
(so the script can be piped directly into `loom-synth-fu-dump`).
"""

import argparse
import random
import sys
from pathlib import Path


# Members of the arith.addi/subi share group plus a couple of bigger
# share groups so the generator can vary op identity within a
# share-group across N inputs without changing topology. Each entry is
# (op-mlir-name, share-group-key). The `key` is the canonical group
# label we tag onto each emitted func.func via `loom.synth_group`, so
# only ops with matching keys can ever land in the same synthesized
# FU. We restrict to the addi/subi group (smallest, fastest to
# synthesize) plus an integer-bitwise group for variety.
SHARE_GROUPS = {
    "alu_int_32": ["arith.addi", "arith.subi"],
}


def render_func(idx: int, op_name: str, group: str) -> str:
    """Emit one tier-A `func.func` wrapping a single-binop dataflow.subgraph.

    The function name is `pat<idx>` (deterministic from idx). The
    block-arg / SSA names follow the same convention as the existing
    handwritten unit tests (single_share_group.mlir, etc.) so the
    generated MLIR parses with the same dialect surface.
    """
    return (
        f"func.func @pat{idx}(%a: i32, %b: i32) -> i32\n"
        f"    attributes {{loom.synth_group = \"{group}\"}} {{\n"
        f"  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {{\n"
        f"    %s = {op_name} %x, %y : i32\n"
        f"    dataflow.yield %s : i32\n"
        f"  }}\n"
        f"  return %r : i32\n"
        f"}}\n"
    )


def render_module(n: int, seed: int, group: str) -> str:
    rng = random.Random()
    rng.seed((seed * 1000003) ^ n)
    members = SHARE_GROUPS[group]
    header = (
        f"// Auto-generated tier-A synthesizer perf workload.\n"
        f"// n={n} seed={seed} group={group}.\n"
        f"// Do not edit by hand. Produced by gen_synth.py.\n\n"
    )
    bodies = []
    for i in range(n):
        op_name = members[rng.randrange(len(members))]
        bodies.append(render_func(i, op_name, group))
    return header + "\n".join(bodies)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True,
                   help="Number of tier-A subgraph func.funcs to emit.")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic RNG seed (default 42).")
    p.add_argument("--group", type=str, default="alu_int_32",
                   choices=sorted(SHARE_GROUPS.keys()),
                   help="loom.synth_group label and op pool.")
    p.add_argument("--out", type=str, default="-",
                   help="Output .mlir path. Default '-' writes to stdout.")
    args = p.parse_args(argv)

    if args.n < 1:
        print("error: --n must be >= 1", file=sys.stderr)
        return 2

    text = render_module(args.n, args.seed, args.group)
    if args.out == "-" or args.out == "":
        sys.stdout.write(text)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

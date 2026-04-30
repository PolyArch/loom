#!/usr/bin/env python3
"""Deterministic synthetic dataflow.graph generator for perf regression.

Emits an MLIR file containing:
  * One fabric.fu per supported op kind in OP_LIBRARY (six single-op FUs).
  * One func.func wrapping a dataflow.graph with N body ops.

Each body op picks one of the OP_LIBRARY kinds uniformly at random and
draws its two operands from the prior body ops or graph block args (so
the body is forward-only acyclic). The terminator yields the last body
op's result.

Output is byte-identical for identical (n, seed).
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Op kinds usable in the synthetic graph. Each entry is (mlir-op-name,
# fabric-symbol, fu-symbol) where fu-symbol must be a unique func name.
OP_LIBRARY = [
    ("arith.addi", "@arith.addi", "fu_addi"),
    ("arith.muli", "@arith.muli", "fu_muli"),
    ("arith.subi", "@arith.subi", "fu_subi"),
    ("arith.andi", "@arith.andi", "fu_andi"),
    ("arith.ori",  "@arith.ori",  "fu_ori"),
    ("arith.xori", "@arith.xori", "fu_xori"),
]


def render_fu_library() -> str:
    """Emit one single-op fabric.fu wrapped in a fabric.module per entry."""
    blocks = []
    for op_name, fab_sym, fu_sym in OP_LIBRARY:
        blocks.append(
            f"fabric.module @{fu_sym}(%a : !fabric.bits<32>, "
            f"%b : !fabric.bits<32>) {{\n"
            f"  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {{\n"
            f"    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)\n"
            f"                  -> !fabric.bits<32> {{\n"
            f"      %k = fabric.op [{fab_sym}] (%x, %y)\n"
            f"           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>\n"
            f"      fabric.yield %k : !fabric.bits<32>\n"
            f"    }}\n"
            f"  }}\n"
            f"  fabric.yield\n"
            f"}}\n"
        )
    return "\n".join(blocks)


def render_graph(n: int, rng: random.Random) -> str:
    """Emit a single function-wrapped dataflow.graph with N body ops.

    Body op naming convention: %v0, %v1, ... %v{n-1}.
    Block args: %x, %y (two i32 args).
    """
    lines = []
    lines.append("func.func @graph_synth(%a: i32, %b: i32) -> i32 {")
    lines.append("  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {")

    # Pool of SSA names available as operands. Start with the two block args.
    pool = ["%x", "%y"]
    for i in range(n):
        op_name, _, _ = OP_LIBRARY[rng.randrange(len(OP_LIBRARY))]
        # Forward-only: only previously-defined SSA values are eligible.
        # rng.choice keeps determinism.
        lhs = rng.choice(pool)
        rhs = rng.choice(pool)
        ssa = f"%v{i}"
        lines.append(f"    {ssa} = {op_name} {lhs}, {rhs} : i32")
        pool.append(ssa)

    last = f"%v{n - 1}" if n > 0 else "%x"
    lines.append(f"    dataflow.yield {last} : i32")
    lines.append("  }")
    lines.append("  return %r : i32")
    lines.append("}")
    return "\n".join(lines)


def render_module(n: int, seed: int) -> str:
    rng = random.Random()
    # Mix n and seed deterministically so different sizes do not share the
    # exact same op stream prefix.
    rng.seed((seed * 1000003) ^ n)
    header = (
        f"// Auto-generated synthetic dataflow.graph. n={n} seed={seed}.\n"
        f"// Do not edit by hand. Produced by gen_synth.py.\n"
    )
    return (
        header
        + "\n"
        + render_fu_library()
        + "\n"
        + render_graph(n, rng)
        + "\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True,
                   help="Number of body ops in the dataflow.graph.")
    p.add_argument("--seed", type=int, required=True,
                   help="Deterministic RNG seed.")
    p.add_argument("--out", type=str, required=True,
                   help="Output .mlir file path.")
    args = p.parse_args(argv)

    if args.n < 0:
        print("error: --n must be >= 0", file=sys.stderr)
        return 2

    text = render_module(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate randomized FU + pattern modules and verify enumerate/match
round-trip consistency.

For each random FU we build patterns matching every op_list member
(positive cases) and a few patterns drawn from outside the op_list
(negative cases), then run -loom-map-subgraph-to-fus and check the
annotations.
"""

import os
import random
import re
import subprocess
import sys

LOOM = os.environ.get("LOOM_BIN", "loom")
SEED = int(os.environ.get("LOOM_FUZZ_SEED", "12345"))
NUM_FUS = int(os.environ.get("LOOM_FUZZ_FUS", "12"))


INT_BIN_GROUPS = [
    ["arith.addi", "arith.subi"],
    ["arith.divsi", "arith.remsi"],
    ["arith.divui", "arith.remui"],
    ["arith.shli", "arith.shrsi", "arith.shrui"],
    ["arith.andi", "arith.ori", "arith.xori"],
    ["arith.minsi", "arith.maxsi"],
    ["arith.minui", "arith.maxui"],
]
INT_BIN_SINGLETONS = ["arith.muli"]

FLOAT_BIN_GROUPS = [
    ["arith.addf", "arith.subf"],
    ["arith.divf", "arith.remf"],
    ["arith.minimumf", "arith.maximumf"],
]
FLOAT_BIN_SINGLETONS = ["arith.mulf"]

FLOAT_UN_SINGLETONS = ["math.sin", "math.cos", "math.sqrt"]


def run(args, input=None):
    proc = subprocess.run(args, input=input, capture_output=True, text=True,
                          check=False)
    return proc.returncode, proc.stdout, proc.stderr


def kind_of(op):
    if op in INT_BIN_SINGLETONS or any(op in g for g in INT_BIN_GROUPS):
        return "int_bin"
    if op in FLOAT_BIN_SINGLETONS or any(op in g for g in FLOAT_BIN_GROUPS):
        return "float_bin"
    if op in FLOAT_UN_SINGLETONS:
        return "float_un"
    raise RuntimeError(f"unknown op {op}")


def gen_fu(rng, idx):
    flavor = rng.choice(["int_bin", "float_bin", "float_un"])
    if flavor == "int_bin":
        choice = rng.choice(INT_BIN_GROUPS + [[op] for op in INT_BIN_SINGLETONS])
        width = rng.choice([8, 16, 32])
    elif flavor == "float_bin":
        choice = rng.choice(
            FLOAT_BIN_GROUPS + [[op] for op in FLOAT_BIN_SINGLETONS])
        width = rng.choice([16, 32, 64])
    else:
        choice = [rng.choice(FLOAT_UN_SINGLETONS)]
        width = rng.choice([16, 32, 64])
    arity = 1 if flavor == "float_un" else 2
    bits = f"!fabric.bits<{width}>"
    sym_list = ", ".join(f"@{s}" for s in choice)
    if arity == 2:
        text = (
            f"\nfabric.module @hw_{idx}(%a : {bits}, %b : {bits}) {{\n"
            f"  fabric.pe [spatial] (%pa = %a : {bits}, %pb = %b : {bits}) -> {bits} {{\n"
            f"    fabric.fu(%x = %pa : {bits}, %y = %pb : {bits}) -> {bits} {{\n"
            f"      %k = fabric.op [{sym_list}] (%x, %y)\n"
            f"           : ({bits}, {bits}) -> {bits}\n"
            f"      fabric.yield %k : {bits}\n"
            f"    }}\n"
            f"  }}\n"
            f"  fabric.yield\n"
            f"}}\n"
        )
    else:
        text = (
            f"\nfabric.module @hw_{idx}(%a : {bits}) {{\n"
            f"  fabric.pe [spatial] (%pa = %a : {bits}) -> {bits} {{\n"
            f"    fabric.fu(%x = %pa : {bits}) -> {bits} {{\n"
            f"      %k = fabric.op [{sym_list}] (%x) : ({bits}) -> {bits}\n"
            f"      fabric.yield %k : {bits}\n"
            f"    }}\n"
            f"  }}\n"
            f"  fabric.yield\n"
            f"}}\n"
        )
    ty_prefix = "i" if flavor == "int_bin" else "f"
    return text, choice, flavor, ty_prefix, width


def gen_pattern(name, op, ty_prefix, width):
    sw_ty = f"{ty_prefix}{width}"
    if kind_of(op) == "float_un":
        body = f"%k = {op} %a : {sw_ty}"
        sig = f"%x: {sw_ty}"
        bb = f"%a = %x : {sw_ty}"
    else:
        body = f"%k = {op} %a, %b : {sw_ty}"
        sig = f"%x: {sw_ty}, %y: {sw_ty}"
        bb = f"%a = %x : {sw_ty}, %b = %y : {sw_ty}"
    return (
        f"\nfunc.func @{name}({sig}) -> {sw_ty} {{\n"
        f"  %r = dataflow.subgraph({bb}) -> {sw_ty} attributes "
        f"{{loom.is_pattern}} {{\n"
        f"    {body}\n"
        f"    dataflow.yield %k : {sw_ty}\n"
        f"  }}\n"
        f"  return %r : {sw_ty}\n"
        f"}}\n"
    )


def all_ops_of_kind(kind):
    if kind == "int_bin":
        return INT_BIN_SINGLETONS + sum(INT_BIN_GROUPS, [])
    if kind == "float_bin":
        return FLOAT_BIN_SINGLETONS + sum(FLOAT_BIN_GROUPS, [])
    return FLOAT_UN_SINGLETONS


def parse_attr(text, fname, attr):
    fn_match = re.search(
        r"func\.func @" + re.escape(fname) + r"\b[\s\S]*?return", text)
    if not fn_match:
        return None
    body = fn_match.group(0)
    if attr == "loom.unmatched":
        return "loom.unmatched" in body
    m = re.search(re.escape(attr) + r" = \"([^\"]*)\"", body)
    return m.group(1) if m else None


def main():
    rng = random.Random(SEED)
    failures = []

    for i in range(NUM_FUS):
        fu_text, op_list, flavor, ty_prefix, width = gen_fu(rng, i)

        positive = []
        for j, op in enumerate(op_list):
            n = f"pat_{i}_{j}"
            positive.append((n, op, gen_pattern(n, op, ty_prefix, width)))

        decoys = []
        pool = [o for o in all_ops_of_kind(flavor) if o not in op_list]
        rng.shuffle(pool)
        for j, d in enumerate(pool[:2]):
            n = f"decoy_{i}_{j}"
            decoys.append((n, d, gen_pattern(n, d, ty_prefix, width)))

        module = "module {\n" + fu_text
        for _, _, t in positive + decoys:
            module += t
        module += "\n}\n"

        rc, out, err = run([LOOM, "-loom-map-subgraph-to-fus"], input=module)
        if rc != 0:
            failures.append(
                f"FU#{i} flavor={flavor} loom failed rc={rc}\n"
                f"--- stderr ---\n{err}\n--- module ---\n{module}")
            continue

        for n, op, _ in positive:
            cfg = parse_attr(out, n, "loom.match_config")
            mfu = parse_attr(out, n, "loom.matched_fu")
            if cfg is None or mfu is None:
                failures.append(
                    f"FU#{i} pat={n} op={op} expected match got "
                    f"cfg={cfg!r} mfu={mfu!r}\n--- output ---\n{out}")
                continue
            if len(op_list) > 1 and op not in cfg:
                failures.append(
                    f"FU#{i} pat={n} op={op} matched but cfg missing op: cfg={cfg!r}")

        for n, op, _ in decoys:
            unmatched = parse_attr(out, n, "loom.unmatched")
            mfu = parse_attr(out, n, "loom.matched_fu")
            if not unmatched and mfu is not None:
                failures.append(
                    f"FU#{i} decoy={n} op={op} expected unmatched got mfu={mfu!r}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        sys.exit(1)
    print(f"OK: {NUM_FUS} FUs, all positive/negative checks passed")


if __name__ == "__main__":
    main()

# Fabric Hardware Share Groups

This document specifies the set of software op kinds that may legally co-occur
in the `op_list` of a single `fabric.op`. The list is enforced by
`FuOp::verify` and the canonical source is `hwShareGroups()` in
`lib/Fabric/IR/FabricOps.cpp`.

## Why share groups exist

A `fabric.op` represents one block of hardware datapath. Its `op_list`
attribute names the software op kinds that this block can execute. If
`op_list` has more than one entry, the entries must all map to the **same
physical datapath** with only different output-tap or control-bit
configurations. Two software op kinds belong in the same `op_list` only when
the synthesizable RTL truly shares its datapath between them.

Counter-examples that the verifier rejects:

* `fabric.op [@arith.addi, @arith.muli]` -- a multiplier and an adder are
  separate datapaths in any standard synthesis flow; they cannot share a
  physical block. Model this as two `fabric.op`s plus a `fabric.mux` to
  select between their outputs.
* `fabric.op [@arith.addi, @arith.subf]` -- integer addition and floating-
  point subtraction share no RTL beyond an XOR.

Multi-member groups encode genuine RTL sharing: a single ALU that performs
`a + b` or `a - b` by inverting one operand; a single Booth multiplier whose
control flag selects unsigned vs. signed product; a CORDIC iterator whose two
output taps yield `sin(x)` and `cos(x)` simultaneously.

## Verifier rule

`FuOp::verify` enforces, for every `fabric.op` whose `op_list` has more than
one entry:

1. Every entry must be in **some** multi-member share group.
2. All entries must be in the **same** share group.

Singleton `op_list`s (one entry) are always legal regardless of the table.
A symbol that is not in any multi-member group is implicitly its own
singleton and must occupy a `fabric.op` alone.

When `op_list` has more than one entry, `sw_configs` must contain an
`op_sel` key whose `StringAttr` value matches one of the symbols in
`op_list`. `op_sel` is the runtime knob that selects which member of the
share group is active in the current configuration.

## Canonical share-group table

| # | Members | RTL sharing rationale |
|---|---|---|
| 1 | `arith.addi`, `arith.subi` | Subtraction is addition with one operand inverted plus a carry-in. One adder tree, one control bit. |
| 2 | `arith.divsi`, `arith.remsi` | Quotient and remainder fall out of the same signed long-division iteration. |
| 3 | `arith.divui`, `arith.remui` | Unsigned counterpart of group 2. |
| 4 | `arith.shli`, `arith.shrsi`, `arith.shrui` | One barrel shifter with direction and arithmetic-vs-logical select bits. |
| 5 | `arith.andi`, `arith.ori`, `arith.xori` | One bit-wise ALU; the function is selected by 2 control bits per bit-slice. |
| 6 | `arith.minsi`, `arith.maxsi` | One signed comparator; min vs max is a single output mux on the result. |
| 7 | `arith.minui`, `arith.maxui` | Unsigned counterpart of group 6. |
| 8 | `arith.sitofp`, `arith.uitofp` | Same mantissa / exponent generator; signedness affects only the absolute-value preprocessor. |
| 9 | `arith.fptosi`, `arith.fptoui` | Same mantissa / exponent extractor; signedness affects only the post-rounding sign attach. |
| 10 | `arith.addf`, `arith.subf` | Floating-point adder with sign invert on operand B. |
| 11 | `arith.divf`, `arith.remf` | Same Newton / SRT iteration core; remainder is the divisor-multiply residue. |
| 12 | `arith.minimumf`, `arith.maximumf` | Floating-point comparator with one output-mux control. |
| 13 | `math.sin`, `math.cos` | Single CORDIC rotator; sine and cosine are the X / Y outputs of the same iteration. |
| 14 | `math.sinh`, `math.cosh` | Hyperbolic-mode CORDIC; same symmetry as group 13. |
| 15 | `math.exp`, `math.exp2`, `math.expm1` | Same exponential series core; arguments and post-corrections differ. |
| 16 | `math.log`, `math.log2`, `math.log10`, `math.log1p` | Same logarithm CORDIC / approximation core; output multiplier and pre-bias differ. |
| 17 | `math.floor`, `math.ceil`, `math.round`, `math.trunc`, `math.roundeven` | One rounding network with mode-select control. |
| 18 | `math.sqrt`, `math.rsqrt` | Same Newton iteration; reciprocal is one extra division step shared with the iteration result. |
| 19 | `math.tanh`, `math.erf` | Same Pade or LUT-based approximation core within shared input ranges. |

The canonical source of truth is `hwShareGroups()` in
`lib/Fabric/IR/FabricOps.cpp`. If you ever need to update this document,
mirror that table.

## How to extend

Adding a new share group is intentionally a code change, not a configuration
knob, because each group must correspond to a real RTL implementation that
the fabric backend can synthesize.

1. Confirm that your hardware really does share its datapath between the
   member ops. If you are not building or buying a custom block that does
   this, do not add the group.
2. Add the new entry to `hwShareGroups()` in `lib/Fabric/IR/FabricOps.cpp`.
3. Update this document's table with the rationale for the sharing.
4. Add a unit test under `test/fabric/unit/fu_match/` (or the closest
   appropriate location) that exercises the new group via a multi-member
   `op_list` and verifies that the enumerator emits a template per `op_sel`
   value.

## What to do when sharing does not exist

If you want a `fabric.fu` that can perform two software ops that are NOT in
any share group (for example, "this FU can do an `arith.muli` or an
`arith.addi` selected at runtime"), do not try to express it via a
multi-member `op_list`. Instead model it as:

```mlir
fabric.fu(...) {
  %m = fabric.op [@arith.muli] (%a, %b) : ...
  %s = fabric.op [@arith.addi] (%a, %b) : ...
  %out = fabric.mux %m, %s : ...
  fabric.yield %out : ...
}
```

Each computation has its own datapath block; the `fabric.mux`'s `sw_config`
selects which output is observed. The enumerator will emit a separate
materialized template per mux selection, and dedup will collapse any
configurations that produce isomorphic software graphs.

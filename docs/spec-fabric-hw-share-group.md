# Fabric Hardware Share Groups

This document specifies the semantics of the global authority for which typed
software modes may share one physical datapath. The sole normative member
registry is `include/Common/HwShareGroups.def`; `hwShareGroups()` consumes that
file directly. This document does not maintain a second member table.

## Why share groups exist

A `fabric.op` represents one physical datapath. `op_list` selects an
operation-family envelope admitted by this registry. `hw_params` declares the
complete typed and attributed modes implemented within that envelope. The
distinct operation identities in those modes must equal `op_list`; neither
representation may extend sharing beyond this registry.

Two software operation kinds belong to one group only when synthesizable RTL
truly shares the datapath. The registry owns that global fact; individual
Fabric descriptions may select a subset but may not invent another group.

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

## Verifier Rules

Fabric verification enforces the following:

1. Every multi-member `op_list` entry belongs to one registered group.
2. All entries in that `op_list` belong to the same registered group.
3. The distinct `op_list` entries exactly equal the operation identities in
   `hw_params`.
4. Every `hw_params` mode forms a valid instance of its selected registered
   software operation.
5. `hw_params` modes are complete typed and attributed tuples, not independent
   per-field allowed sets.

Singleton `op_list`s remain legal. A symbol absent from every multi-member
group is an implicit singleton and must occupy a `fabric.op` alone.

Canonical FU encodings select `hw_params` modes by index. They do not persist
an `op_sel` or mode `sw_config` into the Fabric topology. Mapping or a backend
may derive `sw_configs = {mode = N}` transiently from the selected encoding.

## Resolved Registry Entries

`arith.cmpi` and `llvm.icmp` intentionally share the `integer_compare`
family. Integer comparisons use the same predicate-controlled bit comparator.
Pointer `llvm.icmp` modes are limited by exact mode type and predicate
verification; family membership does not erase those typed restrictions.

The `arm_packed_signed_lane_alu` family intentionally includes
`llvm.arm.sadd16` with `llvm.arm.qadd16`, `llvm.arm.qsub16`, and
`llvm.arm.qsub8`. The wrapping add uses the same lane adder with saturation
disabled; subtraction, saturation, and lane width are implementation controls.
These names are currently intrinsic semantic keys used by legacy Fabric
descriptions. They are not registered MLIR operation names, so normalized
`hw_params` modes reject them until an explicit adapter can materialize the
corresponding registered `llvm.call_intrinsic` representation.

## How to extend

Adding a new share group is intentionally a code change, not a configuration
knob, because each group must correspond to a real RTL implementation that
the fabric backend can synthesize.

1. Confirm that your hardware really does share its datapath between the
   member ops. If you are not building or buying a custom block that does
   this, do not add the group.
2. Add the new entry to `include/Common/HwShareGroups.def`.
3. Document any non-obvious implementation rationale without duplicating the
   complete member table.
4. Add a unit test showing that complete typed `hw_params` modes are accepted
   only within the registered group and are selected by explicit valid
   encodings.

## What to do when sharing does not exist

If an FU can perform two software operations that are not in one share group,
do not combine them in one `op_list`. Use separate physical `fabric.op`
resources and explicit routing.

For every FU input consumed by both mutually exclusive datapaths, insert an
input demux or equivalent explicit route selector. Join shared FU outputs with
an output mux. Each valid encoding must correlate all input route selections,
the active operation mode, and the matching output selection.

An output mux with implicit input broadcast is not a valid substitute. It
leaves inactive datapaths consuming values and therefore does not materialize
the selected software function. Distinct boundary correspondence remains part
of configured-function identity and must not be collapsed by isomorphism
deduplication.

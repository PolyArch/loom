// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/verifier_required.yaml dump-stats=true' 2>&1 | FileCheck %s

// Per spec: `verifier_failed` triggers when the synthesized FU does
// not pass MLIR's verifier (`FuOp::verify` or a nested `OpOp::verify`).
// The spec adds: "Indicates a compiler bug; the FU is dropped, no IR
// is appended."
//
// Reachability note: each of the four strategies (anchor, mcs,
// incremental, incremental_random) ends with an explicit
// `mlir::verify(wrapper)` call before transferring ownership of the
// wrapper. The four strategies, when given well-typed tier-A / tier-B
// / tier-C input, always emit IR that matches FuOp's structural
// invariants (children are exclusively `fabric.op` / `fabric.mux` /
// `fabric.demux` / `fabric.yield`, types lift from software to
// `fabric.bits<N>` 1:1, etc.). The `verifier_failed` enum value is
// the catch for a strategy bug -- a future tier or a future
// regression that emits ill-formed IR. Without instrumenting one of
// the strategies to emit a deliberately-broken wrapper we cannot
// fabricate a synthesizer input that organically reaches this branch.
// The pass scaffolding's worker-to-main-thread re-parse path also
// demotes a parse failure to `verifier_failed`, but a parse failure
// of well-formed printed IR is itself unreachable from valid input.
// This test pins the spec wording so a future implementer can flip
// it from XFAIL to PASS when the gap closes.
//
// XFAIL: *
// CHECK: warning:
// CHECK-SAME: synthesis failed: verifier_failed
// CHECK: loom.synth_failed = "verifier_failed"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

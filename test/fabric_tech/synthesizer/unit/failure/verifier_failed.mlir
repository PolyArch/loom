// RUN: loom-synth-verifier-test 2>&1 | FileCheck %s

// Per spec: `verifier_failed` triggers when the synthesized FU does
// not pass MLIR's verifier (`FuOp::verify` or a nested `OpOp::verify`).
// The spec adds: "Indicates a compiler bug; the FU is dropped, no IR
// is appended."
//
// Reachability note: each of the four production strategies (anchor,
// mcs, incremental, incremental_random) ends with an explicit
// `mlir::verify(wrapper)` call before transferring ownership of the
// wrapper. Today's strategies, when given well-typed input, emit IR
// that satisfies `FuOp::verify`'s structural invariants (children
// exclusively `fabric.op` / `fabric.mux` / `fabric.demux` /
// `fabric.yield`, types lift from software to `fabric.bits<N>` 1:1).
// The `verifier_failed` enum value is the catch for a strategy bug --
// a future tier or a future regression that emits ill-formed IR.
//
// To exercise the diagnostic + attribute emission path organically
// without instrumenting the production strategies, this test drives
// the shared `annotateAndDiagnoseGroupFailure` helper (factored out
// of `GeneralizeSubgraphsToFuPass`) via `loom-synth-verifier-test`.
// The helper deliberately constructs a wrapper whose inner
// `fabric.fu` body contains an `arith.addi` -- not in the allow-set
// enforced by `FuOp::verify` -- confirms `mlir::verify(wrapper)`
// rejects it, and then emits the canonical synthesis-failure
// diagnostic + `loom.synth_failed` attribute exactly the way the
// production pass would on a real strategy bug. If a real regression
// in `FuOp::verify` or in any strategy ever flips the path on
// naturally, the same helper is the single source of truth, so the
// production diagnostic stays byte-identical.

// CHECK: warning:
// CHECK-SAME: synthesis failed: verifier_failed
// CHECK: loom.synth_failed = "verifier_failed"

// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/no_legal_required.yaml dump-stats=true' 2>&1 | FileCheck %s

// Per spec: `no_legal_materialization` triggers when "the strategy
// produced an FU whose enumerated materializations do not satisfy
// `OpOp::verify` / `FuOp::verify`, or whose port assignments
// contradict the dialect's static type rules. Distinct from
// `verifier_failed` in that the FU itself passes `mlir::verify`
// but the enumerator emits nothing legal against it."
//
// Reachability note: the four strategies emit FUs whose enumerator
// materializations are by construction in-spec (`hw_params` axes
// derived from observed inputs, op_list constrained to one share
// group with a uniform width, etc.). For `SubgraphEnumerator` to
// produce an empty / fully-illegal candidate set the FU would need
// to violate `OpOp::verify` invariants -- which is the same bug
// pathway as `verifier_failed` but caught one stage later. The
// closed enum value is dead-code-defensive against a future strategy
// whose `hw_params` synthesis disagrees with the enumerator's
// realizability rules. Until such a strategy exists we cannot
// fabricate a synthesizer input that organically reaches this branch.
// This test pins the spec wording so a future implementer can flip
// it from XFAIL to PASS when the gap closes.
//
// XFAIL: *
// CHECK: warning:
// CHECK-SAME: synthesis failed: no_legal_materialization
// CHECK: loom.synth_failed = "no_legal_materialization"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

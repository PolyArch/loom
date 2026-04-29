// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/coverage_required.yaml dump-stats=true' 2>&1 | FileCheck %s

// Per spec: `coverage_verify_failed` triggers when the coverage
// verifier rejects an FU even though the strategy thought it was a
// valid materialization. The Anchor strategy's spec contract is
// "every input the BFS accepted is structurally enumerable by the
// resulting FU", so for tier-A inputs the post-build CoverageVerifier
// always reports `allCovered=true` against the same input set the
// strategy just consumed. Likewise, Incremental / IncrementalRandom /
// MCS each cross-check coverage internally before declaring success;
// none of them organically emits a wrapper that fails the verifier.
//
// Reachability note: the closed enum value `coverage_verify_failed`
// guards a runtime invariant -- "if the strategy says it succeeded,
// the verifier agrees" -- that the current four strategies all keep.
// The check at lib/Fabric/Tech/Synthesizer/Anchor.cpp's
// `if (!result.coverage.allCovered())` (and the equivalent guards in
// MCS / Incremental / IncrementalRandom) is dead-code-defensive against
// future strategies whose enumerator approximation might disagree
// with `subgraphsIsomorphic`. Until such a strategy exists we cannot
// fabricate a synthesizer input that organically produces this enum
// value without modifying production code; this test pins the spec
// wording so a future implementer can flip it from XFAIL to PASS
// when the gap closes.
//
// XFAIL: *
// CHECK: warning:
// CHECK-SAME: synthesis failed: coverage_verify_failed
// CHECK: loom.synth_failed = "coverage_verify_failed"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/anchor.yaml' 2>&1 | FileCheck %s --check-prefix=ANY
// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/incremental.yaml' 2>&1 | FileCheck %s --check-prefix=ANY
// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/mcs.yaml' 2>&1 | FileCheck %s --check-prefix=ANY
// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/incremental_random.yaml' 2>&1 | FileCheck %s --check-prefix=ANY
//
// Cross-strategy equivalence: a single tier-A workload (two i32
// subgraphs sharing the arith.addi/subi hardware-share group) is
// synthesized by each of the four strategies. Different strategies
// legitimately produce different FU IR (e.g. node ordering, op-list
// permutations within mux/demux variants), so the only invariant we
// assert is full coverage and a `success` reason on the canonical
// `synth-stat` line. The pass-scaffolding emits one synth-stat remark
// per group; using `--check-prefix=ANY` lets the same FileCheck spec
// validate every strategy.
//
// Acceptance criterion 4 of the spec ("strategy-agnostic coverage"):
// every supported strategy must report `covered=N/N` on a tier-A
// input. This test pins that contract.
//
// ANY: synth-stat group=alu_int_32
// ANY-SAME: reason=success
// ANY-SAME: covered=2/2
// ANY: func.func @fu_alu_int_32
// ANY-SAME: loom.synthesized_for = "alu_int_32"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

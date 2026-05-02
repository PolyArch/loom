// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/../incremental/incremental.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INC
// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MCS --implicit-check-not=unrealized_conversion_cast

// Real MCES can share the `arith.addi` skeleton even when both inputs
// have same-length tails from incompatible hardware share groups. The
// Incremental strategy cannot express this as a one-op head or tail
// extension, so it must fail while MCS succeeds.

// INC: warning:
// INC-SAME: group "mces_tail_split": synthesis failed: topology_mismatch
// INC: remark: {{.*}}synth-stat group=mces_tail_split strategy=incremental reason=topology_mismatch

// MCS: remark: {{.*}}synth-stat group=mces_tail_split strategy=mcs reason=success
// MCS-SAME: cost=2.020000e+02
// MCS-SAME: covered=2/2
// MCS-SAME: nodes=3/1/1
// MCS: fabric.module @fu_mces_tail_split
// MCS: fabric.pe [spatial]
// MCS: fabric.fu
// MCS: fabric.op [@arith.addi]
// MCS-DAG: fabric.demux
// MCS-DAG: fabric.op [@arith.muli]
// MCS-DAG: fabric.op [@arith.divsi]
// MCS-DAG: fabric.mux
// MCS: fabric.yield

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "mces_tail_split"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_add_then_div(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "mces_tail_split"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.divsi %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

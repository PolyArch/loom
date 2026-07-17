// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor_with_mux.yaml dump-stats=true' 2>&1 | FileCheck %s

// Software i32 and i64 modes share one bits<64> physical datapath.

// CHECK: remark: {{.*}}synth-stat group=alu_int_mixed_bw strategy=anchor reason=success
// CHECK-SAME: covered=3/3
// CHECK-SAME: encodings=3
// CHECK-SAME: covered_encodings=3
// CHECK-SAME: extra_capability=0
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK-SAME: (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>

func.func @pat_addi_i32_a(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %s = arith.addi %a, %b : i32
  return %s : i32
}

func.func @pat_addi_i32_b(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %s = arith.subi %a, %b : i32
  return %s : i32
}

func.func @pat_addi_i64(%a: i64, %b: i64) -> i64
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %s = arith.addi %a, %b : i64
  return %s : i64
}

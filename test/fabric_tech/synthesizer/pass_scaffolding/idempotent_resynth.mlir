// RUN: loom %s -loom-synthesize-configured-functions 2>&1 | FileCheck %s

// The module already contains a top-level `func.func @fu_y` tagged with
// `loom.synthesized_for = "y"` that is a real synthesized wrapper:
//   * body shape: exactly one inner `fabric.fu` plus a `func.return`
//     terminator
//   * inner fabric.fu passes its own verifier
//   * signature matches the lift of the input configured function's block-arg
//     types (i32, i32) and yield types (i32) to fabric.bits<32>
// Re-running the pass is a no-op for that group: the precheck detects
// the marker, validates the body shape and signature, and emits a
// `remark: skipping idempotent re-synth`. The input func.func is
// neither annotated with `loom.synth_failed` nor stripped.

// CHECK: remark: {{.*}}group "y": skipping idempotent re-synth
// CHECK-NOT: loom.synth_failed
// CHECK-DAG: fabric.module @fu_y
// CHECK-DAG: loom.synthesized_for = "y"

fabric.module @fu_y(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    attributes {loom.synthesized_for = "y"} {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%aa = %pa : !fabric.bits<32>,
              %bb = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %x = fabric.op [@arith.addi] (%aa, %bb)
           {hw_params = [{op = @arith.addi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32],
             attributes = {overflowFlags = #arith.overflow<none>}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %x : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "y"} {
  %s = arith.addi %a, %b : i32
  return %s : i32
}

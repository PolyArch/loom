// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph conditional_div \
// RUN:   --arg 0=false --arg 1=42 --arg 2=0 --arg 3=7 \
// RUN:   --output %t.inactive.json
// RUN: FileCheck %s --check-prefix=INACTIVE < %t.inactive.json

// A canonical actor with conditional undefined behavior remains lazy after
// region lowering. Its captured operands receive only the selected lane's
// tokens, so an inactive division by zero never fires.

// CHECK-LABEL: dataflow.graph private @conditional_div
// CHECK: %[[NUMERATOR:.*]]:2 = dataflow.demux %arg1, %arg2 : (i1, i32) -> (i32, i32)
// CHECK: %[[DIVISOR:.*]]:2 = dataflow.demux %arg1, %arg3 : (i1, i32) -> (i32, i32)
// CHECK: %[[FALLBACK:.*]]:2 = dataflow.demux %arg1, %arg4 : (i1, i32) -> (i32, i32)
// CHECK: %[[QUOTIENT:.*]] = arith.divui %[[NUMERATOR]]#1, %[[DIVISOR]]#1 : i32
// CHECK: %[[SELECTED:.*]] = dataflow.mux %arg1, %[[FALLBACK]]#0, %[[QUOTIENT]] : (i1, i32, i32) -> i32
// CHECK: dataflow.sync {{.*}}, %[[SELECTED]] : (none, i32) -> (none, i32)
// CHECK-NOT: scf.if

// INACTIVE: "final_outputs": [
// INACTIVE-NEXT: "none",
// INACTIVE-NEXT: "i32:7"
// INACTIVE-NOT: "arith.divui"
// INACTIVE: "status": "pass"
dataflow.graph private @conditional_div(
    %start: none, %cond: i1, %numerator: i32, %divisor: i32, %fallback: i32)
    -> (i32)
    attributes {input_segments = array<i32: 4, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  %selected = scf.if %cond -> (i32) {
    %quotient = arith.divui %numerator, %divisor : i32
    scf.yield %quotient : i32
  } else {
    scf.yield %fallback : i32
  }
  dataflow.graph.return %start, %selected : none, i32
}

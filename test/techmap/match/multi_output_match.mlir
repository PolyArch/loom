// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: a multi-output match. The FU produces two outputs from a single
// dataflow.stream (value, ready-cond). The user pattern that yields both
// must bind to it as one subgraph.

func.func @hw_stream(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>,
                     %step: !fabric.bits<32>) {
  %i, %r = fabric.fu(%l = %lb : !fabric.bits<32>,
                     %u = %ub : !fabric.bits<32>,
                     %s = %step : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<1>) {
    %x, %y = fabric.op [@dataflow.stream] (%l, %u, %s)
             {hw_params = [{step_op = ["+="], cont_cond = ["<"]}]}
             : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
               -> (!fabric.bits<32>, !fabric.bits<1>)
    fabric.yield %x, %y : !fabric.bits<32>, !fabric.bits<1>
  }
  return
}

// CHECK-LABEL: @pat_stream
func.func @pat_stream(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_stream#0"
  %i, %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32)
           -> (i32, i1) attributes {loom.is_pattern} {
    %x, %y = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i32
    dataflow.yield %x, %y : i32, i1
  }
  return %i, %r : i32, i1
}

// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: a multi-output match. The FU produces two outputs from a single
// dataflow.stream (value, ready-cond). The user pattern that yields both
// must bind to it as one subgraph.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (stream's TypeParam(0) data ports accept any width); the
// pattern is correspondingly typed as i1.

fabric.module @hw_stream(%lb : !fabric.bits<1>, %ub : !fabric.bits<1>, %step : !fabric.bits<1>) {
  fabric.spatial_pe(%plb = %lb : !fabric.bits<1>,
                    %pub = %ub : !fabric.bits<1>,
                    %pstep = %step : !fabric.bits<1>)
                   -> (!fabric.bits<1>, !fabric.bits<1>) {
    fabric.fu(%l = %plb : !fabric.bits<1>,
              %u = %pub : !fabric.bits<1>,
              %s = %pstep : !fabric.bits<1>)
             -> (!fabric.bits<1>, !fabric.bits<1>) {
      %x, %y = fabric.op [@dataflow.stream] (%l, %u, %s)
               {hw_params = [{step_op = ["+="], cont_cond = ["<"]}]}
               : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
                 -> (!fabric.bits<1>, !fabric.bits<1>)
      fabric.yield %x, %y : !fabric.bits<1>, !fabric.bits<1>
    }
  }
  fabric.yield
}


// CHECK-LABEL: @pat_stream
func.func @pat_stream(%lb: i1, %ub: i1, %step: i1) -> (i1, i1) {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_stream#0"
  %i, %r = dataflow.subgraph(%l = %lb : i1, %u = %ub : i1, %s = %step : i1)
           -> (i1, i1) attributes {loom.is_pattern} {
    %x, %y = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i1
    dataflow.yield %x, %y : i1, i1
  }
  return %i, %r : i1, i1
}

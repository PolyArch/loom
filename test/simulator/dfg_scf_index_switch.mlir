// RUN: not loom-dfg-sim %s --graph residual_index_switch --arg 0=2 \
// RUN:   --output %t.json 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: finalized graph contains residual structured operation 'scf.index_switch'

module {
  dataflow.graph private @residual_index_switch(
      %ctrl: none, %selector: index) -> (i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = scf.index_switch %selector -> i32
    case 1 {
      %one = arith.constant 1 : i32
      scf.yield %one : i32
    }
    default {
      %zero = arith.constant 0 : i32
      scf.yield %zero : i32
    }
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%ctrl : none)
  }
}

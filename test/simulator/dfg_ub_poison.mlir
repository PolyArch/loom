// RUN: not loom-dfg-sim %s --graph poison_top --output %t.top.json 2>&1 | FileCheck %s --check-prefix=POISON-REJECT

// POISON-REJECT: finalized graph contains unregistered actor 'ub.poison'

module {
  dataflow.graph private @poison_top(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %poison = ub.poison : i32
    %published:2 = dataflow.sync %ctrl, %poison
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

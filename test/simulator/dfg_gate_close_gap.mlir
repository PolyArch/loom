// RUN: not loom-dfg-sim %s --graph gate_close_gap \
// RUN:   --arg 0=true --arg 0=false --arg 1=7 --arg 2=none \
// RUN:   --output %t.json 2>&1 | FileCheck %s

// CHECK: retirement frontier does not cover close/reset of 'dataflow.gate'

module {
  dataflow.graph private @gate_close_gap(
      %start: none, %phase: i1, %value: i32, %unit: none) -> ()
      attributes {input_segments = array<i32: 0, 3, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %child_phase, %child_value = dataflow.gate %phase, %value : i32
    %tokens = dataflow.invariant %phase, %unit : none
    %complete:2 = dataflow.demux %phase, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%complete#0 : none)
  }
}

// RUN: not loom-dfg-sim %s --graph gate_close_gap \
// RUN:   --output %t.json 2>&1 | FileCheck %s

// CHECK: retirement frontier does not cover close/reset of 'dataflow.gate'

module {
  dataflow.graph private @gate_close_gap(
      %start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %c0 = dataflow.constant %start {const_value = 0 : i32} : i32
    %c1 = dataflow.constant %start {const_value = 1 : i32} : i32
    %c2 = dataflow.constant %start {const_value = 2 : i32} : i32
    %iv, %phase = dataflow.stream %c0, %c2, %c1
        step add while slt : i32
    %values = dataflow.invariant %phase, %c1 : i32
    %child_phase, %child_value = dataflow.gate %phase, %values : i32
    %tokens = dataflow.invariant %phase, %start : none
    %complete:2 = dataflow.demux %phase, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%complete#0 : none)
  }
}

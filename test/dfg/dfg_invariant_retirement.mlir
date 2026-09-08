// RUN: loom-dfg-sim %s --graph invariant_retires_gate --arg 0=0 --arg 1=3 --arg 2=1 --output %t.active.json
// RUN: FileCheck %s < %t.active.json
// RUN: loom-dfg-sim %s --graph invariant_retires_gate --arg 0=0 --arg 1=0 --arg 2=1 --output %t.empty.json
// RUN: FileCheck %s < %t.empty.json

// A later loop's invariant cannot emit until its init event arrives. Its exit
// therefore retains the earlier loop and capture gate's close witnesses,
// including the branch that bypasses the gate in an empty activation.
// CHECK: "status": "pass"

module {
  dataflow.graph private @invariant_retires_gate(
      %start: none, %lb: i16, %ub: i16, %step: i16) -> ()
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %first_iv, %first_phase = dataflow.stream %lb, %ub, %step step add while slt : i16
    %first_value = dataflow.invariant %first_phase, %start : none
    %gate_phase, %gate_value = dataflow.gate %first_phase, %first_value : none
    %gate_lanes:2 = dataflow.demux %gate_phase, %gate_value : (i1, none) -> (none, none)
    %iv, %phase = dataflow.stream %lb, %ub, %step step add while slt : i16
    %first_lanes:2 = dataflow.demux %first_phase, %first_value : (i1, none) -> (none, none)
    %nonempty = arith.cmpi slt, %lb, %ub : i16
    %first_done = dataflow.mux %nonempty, %first_lanes#0, %gate_lanes#0 : (i1, none, none) -> none
    %execution = dataflow.invariant %phase, %first_done : none
    %lane:2 = dataflow.demux %phase, %execution : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories() complete(%lane#0 : none)
  }
}

// RUN: loom-dfg-sim %s --graph stateful_close --arg 0=0 --arg 1=1 --output %t.stateful.json
// RUN: FileCheck %s --check-prefix=STATEFUL < %t.stateful.json
// RUN: loom-dfg-sim %s --graph stream_output --arg 0=0 --arg 1=3 --arg 2=1 --output %t.stream.json
// RUN: FileCheck %s --check-prefix=STREAM < %t.stream.json
// RUN: loom-dfg-sim %s --graph stream_i8_signed_wrap --arg 0=127 --arg 1=124 --arg 2=1 --output %t.i8-wrap.json
// RUN: FileCheck %s --check-prefix=I8-WRAP < %t.i8-wrap.json

// STATEFUL-DAG: "graph": "stateful_close"
// STATEFUL-DAG: "status": "pass"
// STATEFUL-DAG: "dataflow.stream": 5
// STATEFUL-DAG: "dataflow.carry": 6
// STATEFUL-DAG: "dataflow.gate": 10
// STATEFUL-DAG: "dataflow.invariant": 22
// STATEFUL-DAG: "i32:4"

// STREAM-DAG: "graph": "stream_output"
// STREAM-DAG: "status": "pass"
// STREAM-DAG: "final_stream_outputs":
// STREAM-DAG: "i32:0"
// STREAM-DAG: "i32:2"
// STREAM-DAG: "i1:false"

// I8-WRAP-DAG: "graph": "stream_i8_signed_wrap"
// I8-WRAP-DAG: "status": "pass"
// I8-WRAP-DAG: "dynamic_work_items": 1
// I8-WRAP-DAG: "i8:127"
// I8-WRAP-DAG: "i1:true"
// I8-WRAP-DAG: "i1:false"
// I8-WRAP-NOT: "i8:-128"

module {
  dataflow.graph private @stateful_close(
      %start: none, %initial: i32, %increment: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lower = dataflow.constant %start {const_value = 0 : i32} : i32
    %upper = dataflow.constant %start {const_value = 4 : i32} : i32
    %step = dataflow.constant %start {const_value = 1 : i32} : i32
    %iv, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i32
    %carry = dataflow.carry %phase, %initial, %next : i32
    %carry_phase, %body_carry = dataflow.gate %phase, %carry : i32
    %increments = dataflow.invariant %phase, %increment : i32
    %increment_phase, %body_increment =
        dataflow.gate %phase, %increments : i32
    %next = arith.addi %body_carry, %body_increment : i32
    %exit:2 = dataflow.demux %phase, %carry
        : (i1, i32) -> (i32, i32)

    %stream_units = dataflow.invariant %phase, %start : none
    %stream_close:2 = dataflow.demux %phase, %stream_units
        : (i1, none) -> (none, none)
    %carry_units = dataflow.invariant %carry_phase, %start : none
    %carry_close:2 = dataflow.demux %carry_phase, %carry_units
        : (i1, none) -> (none, none)
    %increment_units = dataflow.invariant %increment_phase, %start : none
    %increment_close:2 = dataflow.demux %increment_phase, %increment_units
        : (i1, none) -> (none, none)

    %retired:4 = dataflow.sync %stream_close#0, %carry_close#0,
        %increment_close#0, %exit#0
        : (none, none, none, i32) -> (none, none, none, i32)
    dataflow.graph.return values(%retired#3 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  dataflow.graph private @stream_output(
      %start: none, %lower: i32, %upper: i32, %step: i32) -> (i32, i1)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %iv, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i32, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @stream_i8_signed_wrap(
      %start: none, %lower: i8, %upper: i8, %step: i8) -> (i8, i1)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %iv, %phase = dataflow.stream %lower, %upper, %step
        step add while sgt : i8
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i8, i1) memories()
        complete(%close#0 : none)
  }
}

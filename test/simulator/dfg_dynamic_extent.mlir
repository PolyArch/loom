// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: loom-dfg-sim %s --graph sum --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=0.000000e+00 --arg 4=1.000000e+00 --output %t.dir/n4.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=0 --arg 1=8 --arg 2=1 --arg 3=0.000000e+00 --arg 4=1.000000e+00 --output %t.dir/n8.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=0 --arg 1=16 --arg 2=1 --arg 3=0.000000e+00 --arg 4=1.000000e+00 --output %t.dir/n16.json
// RUN: FileCheck %s --check-prefix=N4 < %t.dir/n4.json
// RUN: FileCheck %s --check-prefix=N8 < %t.dir/n8.json
// RUN: FileCheck %s --check-prefix=N16 < %t.dir/n16.json

// N4-DAG: "workload": "sum"
// N4-DAG: "graph": "sum"
// N4-DAG: "status": "pass"
// N4-DAG: "dynamic_work_items": 4
// N4-DAG: "arith.addf": 4
// N4-DAG: "dataflow.stream": 5
// N4-DAG: "f32:4"

// N8-DAG: "workload": "sum"
// N8-DAG: "graph": "sum"
// N8-DAG: "status": "pass"
// N8-DAG: "dynamic_work_items": 8
// N8-DAG: "arith.addf": 8
// N8-DAG: "dataflow.stream": 9
// N8-DAG: "f32:8"

// N16-DAG: "workload": "sum"
// N16-DAG: "graph": "sum"
// N16-DAG: "status": "pass"
// N16-DAG: "dynamic_work_items": 16
// N16-DAG: "arith.addf": 16
// N16-DAG: "dataflow.stream": 17
// N16-DAG: "f32:16"

module {
  dataflow.graph private @sum(%ctrl: none, %lb: i64, %ub: i64,
                                   %step: i64, %init: f32, %increment: f32)
      -> (f32)
      attributes {input_segments = array<i32: 5, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %iv, %phase = dataflow.stream %lb, %ub, %step
        step add while slt : i64
    %carry = dataflow.carry %phase, %init, %next : f32
    %body_phase, %body_carry = dataflow.gate %phase, %carry : f32
    %exit:2 = dataflow.demux %phase, %carry : (i1, f32) -> (f32, f32)
    %increments = dataflow.invariant %phase, %increment : f32
    %increment_phase, %body_increment = dataflow.gate %phase, %increments : f32
    %next = arith.addf %body_carry, %body_increment : f32
    %units = dataflow.invariant %phase, %ctrl : none
    %closed:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    %body_units = dataflow.invariant %body_phase, %ctrl : none
    %body_closed:2 = dataflow.demux %body_phase, %body_units
        : (i1, none) -> (none, none)
    %increment_units = dataflow.invariant %increment_phase, %ctrl : none
    %increment_closed:2 = dataflow.demux %increment_phase, %increment_units
        : (i1, none) -> (none, none)
    %retired:3 = dataflow.sync %closed#0, %body_closed#0,
        %increment_closed#0 : (none, none, none) -> (none, none, none)
    %published:2 = dataflow.sync %retired#0, %exit#0
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }
}

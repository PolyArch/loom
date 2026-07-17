// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=0.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --output %t.dir/n4.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=8 --arg 3=1 --arg 4=0.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --output %t.dir/n8.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=16 --arg 3=1 --arg 4=0.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --arg 5=1.000000e+00 --output %t.dir/n16.json
// RUN: FileCheck %s --check-prefix=N4 < %t.dir/n4.json
// RUN: FileCheck %s --check-prefix=N8 < %t.dir/n8.json
// RUN: FileCheck %s --check-prefix=N16 < %t.dir/n16.json

// N4-DAG: "workload": "sum"
// N4-DAG: "graph": "sum"
// N4-DAG: "status": "pass"
// N4-DAG: "operation_cost_score": 39
// N4-DAG: "wavefront_steps": 14
// N4-DAG: "event_count": 25
// N4-DAG: "dynamic_work_items": 4
// N4-DAG: "arith.addf": 4
// N4-DAG: "dataflow.stream": 5
// N4-DAG: "f32:4"

// N8-DAG: "workload": "sum"
// N8-DAG: "graph": "sum"
// N8-DAG: "status": "pass"
// N8-DAG: "operation_cost_score": 67
// N8-DAG: "wavefront_steps": 26
// N8-DAG: "event_count": 45
// N8-DAG: "dynamic_work_items": 8
// N8-DAG: "arith.addf": 8
// N8-DAG: "dataflow.stream": 9
// N8-DAG: "f32:8"

// N16-DAG: "workload": "sum"
// N16-DAG: "graph": "sum"
// N16-DAG: "status": "pass"
// N16-DAG: "operation_cost_score": 123
// N16-DAG: "wavefront_steps": 50
// N16-DAG: "event_count": 85
// N16-DAG: "dynamic_work_items": 16
// N16-DAG: "arith.addf": 16
// N16-DAG: "dataflow.stream": 17
// N16-DAG: "f32:16"

module {
  dataflow.graph.func private @sum(%ctrl: none, %lb: i64, %ub: i64,
                                   %step: i64, %init: f32, %increment: f32)
      -> (none, f32) {
    %iv, %phase = dataflow.stream %lb, %ub, %step
        {step_op = "+=", cont_cond = "<"} : i64
    %carry = dataflow.carry %phase, %init, %next : f32
    %body_phase, %body_carry = dataflow.gate %phase, %carry : f32
    %exit:2 = dataflow.demux %phase, %carry : (i1, f32) -> (f32, f32)
    %next = arith.addf %body_carry, %increment : f32
    dataflow.graph.return %ctrl, %exit#0 : none, f32
  }
}

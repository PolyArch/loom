// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=0.000000e+00 --output %t.dir/n4.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=8 --arg 3=1 --arg 4=0.000000e+00 --output %t.dir/n8.json
// RUN: loom-dfg-sim %s --graph sum --arg 0=none --arg 1=0 --arg 2=16 --arg 3=1 --arg 4=0.000000e+00 --output %t.dir/n16.json
// RUN: FileCheck %s --check-prefix=N4 < %t.dir/n4.json
// RUN: FileCheck %s --check-prefix=N8 < %t.dir/n8.json
// RUN: FileCheck %s --check-prefix=N16 < %t.dir/n16.json

// N4-DAG: "workload": "sum"
// N4-DAG: "graph": "sum"
// N4-DAG: "status": "pass"
// N4-DAG: "operation_cost_score": 37
// N4-DAG: "wavefront_steps": 12
// N4-DAG: "event_count": 27
// N4-DAG: "dynamic_work_items": 4
// N4-DAG: "arith.addf": 5
// N4-DAG: "dataflow.stream": 5
// N4-DAG: "f32:4"

// N8-DAG: "workload": "sum"
// N8-DAG: "graph": "sum"
// N8-DAG: "status": "pass"
// N8-DAG: "operation_cost_score": 61
// N8-DAG: "wavefront_steps": 20
// N8-DAG: "event_count": 47
// N8-DAG: "dynamic_work_items": 8
// N8-DAG: "arith.addf": 9
// N8-DAG: "dataflow.stream": 9
// N8-DAG: "f32:8"

// N16-DAG: "workload": "sum"
// N16-DAG: "graph": "sum"
// N16-DAG: "status": "pass"
// N16-DAG: "operation_cost_score": 109
// N16-DAG: "wavefront_steps": 36
// N16-DAG: "event_count": 87
// N16-DAG: "dynamic_work_items": 16
// N16-DAG: "arith.addf": 17
// N16-DAG: "dataflow.stream": 17
// N16-DAG: "f32:16"

module {
  dataflow.graph.func private @sum(%ctrl: none, %lb: i64, %ub: i64,
                                   %step: i64, %init: f32)
      -> (none, f32) {
    %one = dataflow.constant %ctrl {const_value = 1.000000e+00 : f32} : f32
    %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i64
    %inc = dataflow.invariant %rwc, %one : f32
    %carry = dataflow.carry %rwc, %init, %next : f32
    %next = arith.addf %carry, %inc : f32
    dataflow.graph.return %ctrl, %carry : none, f32
  }
}

// RUN: loom-dfg-sim %s --graph sum_load --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --memref 4=1.000000e+00,2.000000e+00,3.000000e+00 --arg 5=0.000000e+00 --output %t.incomplete.json
// RUN: FileCheck %s --check-prefix=INCOMPLETE < %t.incomplete.json
// RUN: loom-dfg-sim %s --graph sum_load --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=5 --arg 3=1 --memref 4=1.000000e+00,2.000000e+00,3.000000e+00 --arg 5=0.000000e+00 --output %t.oob.json
// RUN: FileCheck %s --check-prefix=OOB < %t.oob.json

// INCOMPLETE-DAG: "status": "blocked"
// INCOMPLETE-DAG: "DFG-sim stopped before all returned values produced complete outputs"

// OOB-DAG: "status": "blocked"
// OOB-DAG: "dataflow.load address is out of range"
// OOB-DAG: "DFG-sim stopped with runtime diagnostics"

module {
  dataflow.graph.func private @sum_load(%ctrl: none, %lb: i64, %ub: i64,
                                        %step: i64, %mem: memref<?xf32>,
                                        %init: f32) -> (none, f32) {
    %idx64, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i64
    %idx = arith.index_cast %idx64 : i64 to index
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xf32>
    %carry = dataflow.carry %rwc, %init, %next : f32
    %next = arith.addf %carry, %data : f32
    %done_sync = dataflow.sync %done : (none) -> none
    dataflow.graph.return %done_sync, %carry : none, f32
  }
}

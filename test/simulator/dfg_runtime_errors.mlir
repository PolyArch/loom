// RUN: loom-dfg-sim %s --graph sum_load --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0.000000e+00 --memref 4=1.000000e+00,2.000000e+00,3.000000e+00 --output %t.incomplete.json
// RUN: FileCheck %s --check-prefix=COMPLETE < %t.incomplete.json
// RUN: loom-dfg-sim %s --graph sum_load --arg 0=0 --arg 1=5 --arg 2=1 --arg 3=0.000000e+00 --memref 4=1.000000e+00,2.000000e+00,3.000000e+00 --output %t.oob.json
// RUN: FileCheck %s --check-prefix=OOB < %t.oob.json

// COMPLETE-DAG: "status": "pass"
// COMPLETE-DAG: "dataflow.load": 3
// COMPLETE-DAG: "final_outputs":
// COMPLETE-DAG: "none",
// COMPLETE-DAG: "f32:6"

// OOB-DAG: "status": "blocked"
// OOB-DAG: "graph did not fire its retirement frontier"
// OOB-DAG: "dataflow.load address is out of range"

module {
  dataflow.graph private @sum_load(%ctrl: none, %lb: i64, %ub: i64,
                                        %step: i64, %init: f32,
                                        %mem: memref<?xf32>) -> (f32)
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %idx64, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i64
    %read_frontier = dataflow.carry %rwc, %ctrl, %done : none
    %read_lane:2 = dataflow.demux %rwc, %read_frontier
        : (i1, none) -> (none, none)
    %idx = arith.index_cast %idx64 : i64 to index
    %data, %done = dataflow.load %mem[%idx] %read_lane#1 : memref<?xf32>
    %carry = dataflow.carry %rwc, %init, %next : f32
    %next = arith.addf %carry, %data : f32
    %exit:2 = dataflow.demux %rwc, %carry : (i1, f32) -> (f32, f32)
    %retired:2 = dataflow.sync %read_lane#0, %exit#0
        : (none, f32) -> (none, f32)
    dataflow.graph.return values(%retired#1 : f32) streams() memories()
        complete(%retired#0 : none)
  }
}

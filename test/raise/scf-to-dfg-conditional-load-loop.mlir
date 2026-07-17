// RUN: loom-raise-opt --loom-lower-reduction-to-stream --loom-lower-graph-invariant --loom-lower-graph-control %s > %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=STRUCT < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph g_cond_load_red --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --memref 4=10,20,30,40 --arg 5=0 --arg 6=2 --arg 7=4 --output %t.sim.json
// RUN: FileCheck %s --check-prefix=SIM < %t.sim.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph g_cond_load_red --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --memref 4=10,20,30,40 --arg 5=0 --arg 6=0 --arg 7=2 --output %t.true-then-false.sim.json
// RUN: FileCheck %s --check-prefix=SIM-TF < %t.true-then-false.sim.json

// STRUCT-LABEL: dataflow.graph.func private @g_cond_load_red
// STRUCT: dataflow.stream
// STRUCT: dataflow.carry
// STRUCT: dataflow.gate
// STRUCT: dataflow.demux
// STRUCT-NOT: scf.for
// STRUCT-NOT: scf.if
// STRUCT: dataflow.constant %arg0 {const_value = 0 : index} : index
// STRUCT: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// STRUCT: dataflow.load %arg4[%{{.*}}] %arg0 : memref<?xf32>
// STRUCT: dataflow.demux
// STRUCT: dataflow.mux
// STRUCT: dataflow.graph.return %arg0

// SIM-DAG: "status": "pass"
// SIM-DAG: "f32:70"
// SIM-NOT: dataflow.graph.return value produced

// SIM-TF-DAG: "status": "pass"
// SIM-TF-DAG: "f32:30"
// SIM-TF-NOT: dataflow.graph.return value produced
dataflow.graph.func private @g_cond_load_red(%ctrl: none, %lb: i64, %ub: i64,
                                             %step: i64,
                                             %input: memref<?xf32>,
                                             %init: f32, %lo: i64,
                                             %hi: i64)
    -> (none, f32) {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i64 {
    %gt = arith.cmpi sge, %i, %lo : i64
    %lt = arith.cmpi slt, %i, %hi : i64
    %ok = arith.andi %gt, %lt : i1
    %next = scf.if %ok -> (f32) {
      %idx = arith.index_cast %i : i64 to index
      %data, %done = dataflow.load %input[%idx] %ctrl : memref<?xf32>
      %sum = arith.addf %acc, %data : f32
      scf.yield %sum : f32
    } else {
      scf.yield %acc : f32
    }
    scf.yield %next : f32
  }
  dataflow.graph.return %ctrl, %r : none, f32
}

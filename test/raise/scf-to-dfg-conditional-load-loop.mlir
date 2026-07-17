// RUN: loom-raise-opt --loom-lower-graph-memory %s > %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=STRUCT < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph g_cond_load_red --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=0 --arg 4=2 --arg 5=4 --memref 6=10,20,30,40 --output %t.sim.json
// RUN: FileCheck %s --check-prefix=SIM < %t.sim.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph g_cond_load_red --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=0 --arg 4=0 --arg 5=2 --memref 6=10,20,30,40 --output %t.true-then-false.sim.json
// RUN: FileCheck %s --check-prefix=SIM-TF < %t.true-then-false.sim.json

// STRUCT-LABEL: dataflow.graph.func private @g_cond_load_red
// STRUCT: dataflow.stream
// STRUCT: dataflow.carry
// STRUCT: dataflow.demux
// STRUCT-NOT: scf.for
// STRUCT-NOT: scf.if
// STRUCT-NOT: arith.select
// STRUCT: dataflow.load %arg7[
// STRUCT: dataflow.mux

// SIM-DAG: "status": "pass"
// SIM-DAG: "f32:70"
// SIM-TF-DAG: "status": "pass"
// SIM-TF-DAG: "f32:30"

dataflow.graph.func private @g_cond_load_red(
    %ctrl: none, %lb: i64, %ub: i64, %step: i64,
    %init: f32, %lo: i64, %hi: i64, %input: memref<?xf32>)
    -> (none, f32)
    attributes {input_segments = array<i32: 6, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %result = scf.for %i = %lb to %ub step %step
      iter_args(%acc = %init) -> (f32) : i64 {
    %ge = arith.cmpi sge, %i, %lo : i64
    %lt = arith.cmpi slt, %i, %hi : i64
    %selected = arith.andi %ge, %lt : i1
    %next = scf.if %selected -> (f32) {
      %index = arith.index_cast %i : i64 to index
      %data = memref.load %input[%index] : memref<?xf32>
      %sum = arith.addf %acc, %data : f32
      scf.yield %sum : f32
    } else {
      scf.yield %acc : f32
    }
    scf.yield %next : f32
  }
  dataflow.graph.return %ctrl, %result : none, f32
}

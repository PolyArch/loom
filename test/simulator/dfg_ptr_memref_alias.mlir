// RUN: loom-dfg-sim %s --graph sum_ptr_load --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0.000000e+00 --memref 4=2.000000e+00,4.000000e+00,8.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "sum_ptr_load"
// CHECK-DAG: "graph": "sum_ptr_load"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.load": 3
// CHECK-DAG: "f32:14"

module {
  llvm.func @external_decl(!llvm.ptr) -> i32

  func.func private @host_stub() {
    return
  }

  dataflow.graph private @sum_ptr_load(%ctrl: none, %lb: i64, %ub: i64,
                                      %step: i64, %init: f32,
                                      %memory: memref<?xf32>) -> (f32)
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %iv, %phase = dataflow.stream %lb, %ub, %step
        step add while slt : i64
    %read_frontier = dataflow.carry %phase, %ctrl, %done : none
    %read_lane:2 = dataflow.demux %phase, %read_frontier
        : (i1, none) -> (none, none)
    %idx = arith.index_cast %iv : i64 to index
    %data, %done = dataflow.load %memory[%idx] %read_lane#1 : memref<?xf32>
    %carry = dataflow.carry %phase, %init, %next : f32
    %body_phase, %body_carry = dataflow.gate %phase, %carry : f32
    %exit:2 = dataflow.demux %phase, %carry : (i1, f32) -> (f32, f32)
    %next = arith.addf %body_carry, %data : f32
    %body_close:2 = dataflow.demux %body_phase, %body_carry
        : (i1, f32) -> (f32, f32)
    %nonempty = arith.cmpi slt, %lb, %ub : i64
    %completion:2 = dataflow.demux %nonempty, %read_lane#0
        : (i1, none) -> (none, none)
    %active_retired:2 = dataflow.sync %completion#1, %body_close#0
        : (none, f32) -> (none, f32)
    %control = dataflow.mux %nonempty, %completion#0, %active_retired#0
        : (i1, none, none) -> none
    %retired:2 = dataflow.sync %control, %exit#0
        : (none, f32) -> (none, f32)
    dataflow.graph.return values(%retired#1 : f32) streams() memories()
        complete(%retired#0 : none)
  }
}

// RUN: loom-dfg-sim %s --graph store_then_load --arg 0=none --arg 1=1 --memref 2=0.000000e+00,0.000000e+00,0.000000e+00 --arg 3=7.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "store_then_load"
// CHECK-DAG: "graph": "store_then_load"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "optimistic_event_count"
// CHECK-DAG: "optimistic_cycles": 2
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 2
// CHECK-DAG: "f32:7"

module {
  dataflow.graph.func private @store_then_load(%ctrl: none, %idx: index,
                                               %ptr: !llvm.ptr, %value: f32)
      -> (none, f32) {
    %store_mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xf32>
    %load_mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xf32>
    %store_done = dataflow.store %store_mem[%idx] %value %ctrl : memref<?xf32>
    %data, %load_done = dataflow.load %load_mem[%idx] %store_done : memref<?xf32>
    dataflow.graph.return %load_done, %data : none, f32
  }
}

// RUN: loom-dfg-sim %s --graph missing_graph --workload missing_workload --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "missing_workload"
// CHECK-DAG: "graph": "missing_graph"
// CHECK-DAG: "status": "unsupported"
// CHECK-DAG: "dynamic_work_items": 0
// CHECK-DAG: "final_outputs": []
// CHECK-DAG: "final_memory_state": {}
// CHECK-DAG: "dataflow.graph 'missing_graph' was not found"

module {
  dataflow.graph private @existing_graph(%ctrl: none) -> () {
    dataflow.graph.return %ctrl : none
  }
}

// RUN: loom-pnr-map --dfg-mlir %s --graph math_exp_graph --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload math_exp --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=ARTIFACT < %t.json

// ARTIFACT-DAG: "status": "fail"
// ARTIFACT-DAG: "missing hardware resource for software op math.exp
// ARTIFACT-DAG: "resource_pressure"
// ARTIFACT-DAG: "operation": "math.exp"

module {
  dataflow.graph.func private @math_exp_graph(%ctrl: none, %x: f32)
      -> (none, f32) {
    %y = math.exp %x : f32
    dataflow.graph.return %ctrl, %y : none, f32
  }
}

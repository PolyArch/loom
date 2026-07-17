// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph math_exp_graph --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload math_exp --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=ARTIFACT < %t.json

// ARTIFACT-DAG: "status": "fail"
// ARTIFACT-DAG: "missing hardware resource for software op math.exp
// ARTIFACT-DAG: "resource_pressure"
// ARTIFACT-DAG: "operation": "math.exp"

module {
  dataflow.graph private @math_exp_graph(%ctrl: none, %x: f32)
      -> (f32) {
    %y = math.exp %x : f32
    dataflow.graph.return %ctrl, %y : none, f32
  }
}

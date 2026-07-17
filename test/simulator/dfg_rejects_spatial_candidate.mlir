// RUN: not loom-dfg-sim %s --graph valid_graph --output %t.json 2>&1 | FileCheck %s

// CHECK: finalized program contains temporary loom.spatial_region

module {
  "loom.spatial_region"() ({
  }) : () -> ()

  dataflow.graph private @valid_graph(%start: none) -> () {
    dataflow.graph.return %start : none
  }
}

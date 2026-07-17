// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph direct_route --hardware-mlir %s --hardware direct_route_adg --workload direct_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: direct_route,direct_route_adg,direct_route__direct_route__direct_route_adg,3,2,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "source_endpoint": "direct_route_adg::fabric.op#0.result0"
// JSON-DAG: "sink_endpoint": "direct_route_adg::fabric.op#1.operand0"
// JSON-DAG: "hardware_ref": "direct_route_adg::ssa_edge#0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @direct_route(%ctrl: none, %lhs: i32, %rhs: i32, %limit: i32)
      -> (none, i1) {
    %sum = arith.addi %lhs, %rhs : i32
    %cmp = arith.cmpi slt, %sum, %limit : i32
    dataflow.graph.return %ctrl, %cmp : none, i1
  }

  fabric.module @direct_route_adg(%ctrl : !fabric.bits<0>,
                                  %i32a : !fabric.bits<32>,
                                  %i32b : !fabric.bits<32>,
                                  %i32c : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>,
                         %pc = %i32c : !fabric.bits<32>,
                         %pd = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %limit = %pc : !fabric.bits<32>,
                %token = %pd : !fabric.bits<32> to !fabric.bits<0>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %cmp = fabric.op [@arith.cmpi] (%sum, %limit)
               {hw_params = [{predicate = ["slt"]}]}
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
        %done, %published = fabric.op [@dataflow.sync] (%token, %cmp)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<1>)
              -> (!fabric.bits<0>, !fabric.bits<1>)
        fabric.yield
      }
    }
    fabric.yield
  }
}

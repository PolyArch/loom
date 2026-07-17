// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph pe_boundary_route --hardware-mlir %s --hardware pe_boundary_route_adg --workload pe_boundary_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: pe_boundary_route,pe_boundary_route_adg,pe_boundary_route__pe_boundary_route__pe_boundary_route_adg,3,2,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "segment_kind": "module_path"
// JSON-DAG: "source_endpoint": "pe_boundary_route_adg::fabric.op#0.result0"
// JSON-DAG: "sink_endpoint": "pe_boundary_route_adg::fabric.fu#0.result0"
// JSON-DAG: "source_endpoint": "pe_boundary_route_adg::fabric.fu#0.result0"
// JSON-DAG: "sink_endpoint": "pe_boundary_route_adg::fabric.pe#0.result0"
// JSON-DAG: "source_endpoint": "pe_boundary_route_adg::fabric.pe#0.result0"
// JSON-DAG: "sink_endpoint": "pe_boundary_route_adg::fabric.op#1.operand0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph private @pe_boundary_route(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %product = arith.muli %sum, %rhs : i32
    dataflow.graph.return %ctrl, %product : none, i32
  }

  fabric.module @pe_boundary_route_adg(%ctrl : !fabric.bits<0>,
                                       %lhs : !fabric.bits<32>,
                                       %rhs : !fabric.bits<32>) {
    %rhs_to_sum, %rhs_to_product = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sum = fabric.pe [spatial] (%pa = %lhs : !fabric.bits<32>,
                                %pb = %rhs_to_sum : !fabric.bits<32>)
        -> !fabric.bits<32> {
      %fu_sum = fabric.fu(%fa = %pa : !fabric.bits<32>,
                          %fb = %pb : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
                 : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%px = %sum : !fabric.bits<32>,
                         %py = %rhs_to_product : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      %fu_product = fabric.fu(%fx = %px : !fabric.bits<32>,
                              %fy = %py : !fabric.bits<32>,
                              %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.muli] (%fx, %fy)
                 : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}

// RUN: loom-pnr-map --dfg-mlir %s --graph disconnected_route --hardware-mlir %s --hardware disconnected_adg --workload disconnected_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: disconnected_route,disconnected_adg,disconnected_route__disconnected_route__disconnected_adg,2,0,2,0,fail

// JSON-DAG: "status": "fail"
// JSON-DAG: "routed_edges": 0
// JSON-DAG: "unrouted_edges": 2
// JSON-DAG: "diagnostics": [
// JSON-DAG: "unrouted software edges lack Fabric ADG connectivity"
// JSON-NOT: "source_endpoint"
// JSON-NOT: "sink_endpoint"

module {
  dataflow.graph.func private @disconnected_route(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %doubled = arith.addi %sum, %sum : i32
    dataflow.graph.return %ctrl, %doubled : none, i32
  }

  fabric.module @disconnected_adg(%i32a : !fabric.bits<32>,
                                  %i32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.yield
  }
}

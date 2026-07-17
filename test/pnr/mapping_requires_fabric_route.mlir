// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph disconnected_route --hardware-mlir %s --hardware disconnected_adg --workload disconnected_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: disconnected_route,disconnected_adg,disconnected_route__disconnected_route__disconnected_adg,3,1,2,0,fail

// JSON-DAG: "status": "fail"
// JSON-DAG: "routed_edges": 1
// JSON-DAG: "unrouted_edges": 2
// JSON-DAG: "diagnostics": [
// JSON-DAG: "unrouted software edges lack Fabric ADG connectivity"
// JSON-DAG: "unrouted_edge_details": [
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand0"
// JSON-DAG: "source_endpoint": "disconnected_adg::fabric.op#0.result0"
// JSON-DAG: "sink_endpoint": "disconnected_adg::fabric.op#1.operand0"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand1"
// JSON-DAG: "source_endpoint": "disconnected_adg::fabric.op#0.result0"
// JSON-DAG: "sink_endpoint": "disconnected_adg::fabric.op#1.operand1"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @disconnected_route(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %doubled = arith.addi %sum, %sum : i32
    dataflow.graph.return %ctrl, %doubled : none, i32
  }

  fabric.module @disconnected_adg(%ctrl : !fabric.bits<0>,
                                  %i32a : !fabric.bits<32>,
                                  %i32b : !fabric.bits<32>) {
    %i32a_to_first, %i32a_to_second = fabric.switch [spatial] %i32a
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %i32b_to_first, %i32b_to_second = fabric.switch [spatial] %i32b
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.pe [spatial] (%pa = %i32a_to_first : !fabric.bits<32>,
                         %pb = %i32b_to_first : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.pe [spatial] (%pa = %i32a_to_second : !fabric.bits<32>,
                         %pb = %i32b_to_second : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %sum)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield
      }
    }
    fabric.yield
  }
}

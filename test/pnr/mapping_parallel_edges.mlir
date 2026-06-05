// RUN: loom-pnr-map --dfg-mlir %s --graph parallel_edges --hardware-mlir %s --hardware parallel_adg --workload parallel_edges --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: parallel_edges,parallel_adg,parallel_edges__parallel_adg,2,2,0,0,pass

// JSON-DAG: "routed_edges": 2
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand0"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand1"
// JSON-DAG: "target": "parallel_edges__parallel_adg::route#0"
// JSON-DAG: "target": "parallel_edges__parallel_adg::route#1"

module {
  dataflow.graph.func private @parallel_edges(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %doubled = arith.addi %sum, %sum : i32
    dataflow.graph.return %ctrl, %doubled : none, i32
  }

  fabric.module @parallel_adg(%i32a : !fabric.bits<32>,
                              %i32b : !fabric.bits<32>,
                              %i32c : !fabric.bits<32>) {
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

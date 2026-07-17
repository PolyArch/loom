// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph add_graph --hardware-mlir %s --hardware add_adg --workload add_graph --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: add_graph,add_adg,add_graph__add_graph__add_adg,2,1,0,0,pass

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "add_graph"
// JSON-DAG: "status": "pass"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.sync#0.operand1"

module {
  func.func private @host_stub() {
    %unused = ub.poison : i32
    return
  }

  dataflow.graph.func private @add_graph(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  fabric.module @add_adg(%ctrl : !fabric.bits<0>,
                         %i32a : !fabric.bits<32>,
                         %i32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>,
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

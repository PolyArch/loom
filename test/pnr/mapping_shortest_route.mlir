// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph shortest_route --hardware-mlir %s --hardware shortest_route_adg --workload shortest_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: shortest_route,shortest_route_adg,shortest_route__shortest_route__shortest_route_adg,3,2,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "routed_edges": 2
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "register": "segment_count"
// JSON-DAG: "target": "shortest_route__shortest_route__shortest_route_adg::route#0"
// JSON-DAG: "value": "11"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph private @shortest_route(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %doubled = arith.addi %sum, %rhs : i32
    dataflow.graph.return %ctrl, %doubled : none, i32
  }

  fabric.module @shortest_route_adg(%ctrl : !fabric.bits<0>,
                                    %i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>) {
    %i32b_to_source, %i32b_to_sink = fabric.switch [spatial] %i32b
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %src = fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                                %pb = %i32b_to_source : !fabric.bits<32>)
        -> !fabric.bits<32> {
      %fu_sum = fabric.fu(%lhs = %pa : !fabric.bits<32>,
                          %rhs = %pb : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %src_to_long, %src_to_join = fabric.switch [spatial] %src
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %long0 = fabric.switch [spatial] %src_to_long
        [{connectivity_table = ["1"]}]
        : (!fabric.bits<32>) -> !fabric.bits<32>
    %long1 = fabric.switch [spatial] %long0
        [{connectivity_table = ["1"]}]
        : (!fabric.bits<32>) -> !fabric.bits<32>
    %short_narrow = fabric.switch [spatial] %src_to_join
        [{connectivity_table = ["1"]}]
        : (!fabric.bits<32> to !fabric.bits<8>) -> !fabric.bits<8>
    %joined = fabric.switch [spatial] %long1, %short_narrow
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<8> to !fabric.bits<32>)
       -> !fabric.bits<32>
    fabric.pe [spatial] (%value = %joined : !fabric.bits<32>,
                         %right = %i32b_to_sink : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      %fu_sum = fabric.fu(%lhs = %value : !fabric.bits<32>,
                          %rhs = %right : !fabric.bits<32>,
                          %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %sum)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}

// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph long_add_chain --hardware-mlir %s --hardware long_chain_adg --workload long_add_chain --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: long_add_chain,long_chain_adg,long_add_chain__long_add_chain__long_chain_adg,22,21,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 22
// JSON-DAG: "routed_edges": 21
// JSON-DAG: "unrouted_edges": 0

module {
  dataflow.graph.func private @long_add_chain(%ctrl: none, %seed: i32,
                                               %rhs: i32) -> (none, i32) {
    %v0 = arith.addi %seed, %rhs : i32
    %v1 = arith.addi %v0, %rhs : i32
    %v2 = arith.addi %v1, %rhs : i32
    %v3 = arith.addi %v2, %rhs : i32
    %v4 = arith.addi %v3, %rhs : i32
    %v5 = arith.addi %v4, %rhs : i32
    %v6 = arith.addi %v5, %rhs : i32
    %v7 = arith.addi %v6, %rhs : i32
    %v8 = arith.addi %v7, %rhs : i32
    %v9 = arith.addi %v8, %rhs : i32
    %v10 = arith.addi %v9, %rhs : i32
    %v11 = arith.addi %v10, %rhs : i32
    %v12 = arith.addi %v11, %rhs : i32
    %v13 = arith.addi %v12, %rhs : i32
    %v14 = arith.addi %v13, %rhs : i32
    %v15 = arith.addi %v14, %rhs : i32
    %v16 = arith.addi %v15, %rhs : i32
    %v17 = arith.addi %v16, %rhs : i32
    %v18 = arith.addi %v17, %rhs : i32
    %v19 = arith.addi %v18, %rhs : i32
    %v20 = arith.addi %v19, %rhs : i32
    dataflow.graph.return %ctrl, %v20 : none, i32
  }

  fabric.module @long_chain_adg(%ctrl : !fabric.bits<0>,
                                %seed : !fabric.bits<32>,
                                %rhs : !fabric.bits<32>) {
    %rhs_to_p0, %rhs_to_p1, %rhs_to_p2, %rhs_to_p3, %rhs_to_p4,
        %rhs_to_p5, %rhs_to_p6, %rhs_to_p7, %rhs_to_p8, %rhs_to_p9,
        %rhs_to_p10, %rhs_to_p11, %rhs_to_p12, %rhs_to_p13, %rhs_to_p14,
        %rhs_to_p15, %rhs_to_p16, %rhs_to_p17, %rhs_to_p18, %rhs_to_p19,
        %rhs_to_p20 = fabric.switch [spatial] %rhs
          [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1",
                                  "1", "1", "1", "1", "1", "1", "1", "1",
                                  "1", "1", "1", "1", "1"]}]
          : (!fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    %p0 = fabric.pe [spatial] (%lhs = %seed : !fabric.bits<32>,
                               %r = %rhs_to_p0 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p1 = fabric.pe [spatial] (%lhs = %p0 : !fabric.bits<32>,
                               %r = %rhs_to_p1 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p2 = fabric.pe [spatial] (%lhs = %p1 : !fabric.bits<32>,
                               %r = %rhs_to_p2 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p3 = fabric.pe [spatial] (%lhs = %p2 : !fabric.bits<32>,
                               %r = %rhs_to_p3 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p4 = fabric.pe [spatial] (%lhs = %p3 : !fabric.bits<32>,
                               %r = %rhs_to_p4 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p5 = fabric.pe [spatial] (%lhs = %p4 : !fabric.bits<32>,
                               %r = %rhs_to_p5 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p6 = fabric.pe [spatial] (%lhs = %p5 : !fabric.bits<32>,
                               %r = %rhs_to_p6 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p7 = fabric.pe [spatial] (%lhs = %p6 : !fabric.bits<32>,
                               %r = %rhs_to_p7 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p8 = fabric.pe [spatial] (%lhs = %p7 : !fabric.bits<32>,
                               %r = %rhs_to_p8 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p9 = fabric.pe [spatial] (%lhs = %p8 : !fabric.bits<32>,
                               %r = %rhs_to_p9 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p10 = fabric.pe [spatial] (%lhs = %p9 : !fabric.bits<32>,
                                %r = %rhs_to_p10 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p11 = fabric.pe [spatial] (%lhs = %p10 : !fabric.bits<32>,
                                %r = %rhs_to_p11 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p12 = fabric.pe [spatial] (%lhs = %p11 : !fabric.bits<32>,
                                %r = %rhs_to_p12 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p13 = fabric.pe [spatial] (%lhs = %p12 : !fabric.bits<32>,
                                %r = %rhs_to_p13 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p14 = fabric.pe [spatial] (%lhs = %p13 : !fabric.bits<32>,
                                %r = %rhs_to_p14 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p15 = fabric.pe [spatial] (%lhs = %p14 : !fabric.bits<32>,
                                %r = %rhs_to_p15 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p16 = fabric.pe [spatial] (%lhs = %p15 : !fabric.bits<32>,
                                %r = %rhs_to_p16 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p17 = fabric.pe [spatial] (%lhs = %p16 : !fabric.bits<32>,
                                %r = %rhs_to_p17 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p18 = fabric.pe [spatial] (%lhs = %p17 : !fabric.bits<32>,
                                %r = %rhs_to_p18 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    %p19 = fabric.pe [spatial] (%lhs = %p18 : !fabric.bits<32>,
                                %r = %rhs_to_p19 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%lhs = %p19 : !fabric.bits<32>,
                         %r = %rhs_to_p20 : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>, %b = %r : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%a, %b)
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

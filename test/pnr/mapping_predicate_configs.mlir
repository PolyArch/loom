// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph cmpf_graph --hardware-mlir %s --hardware predicate_adg --workload cmpf_graph --output %t.pass.csv --artifact %t.pass.json
// RUN: FileCheck %s --check-prefix=CSV-PASS < %t.pass.csv
// RUN: FileCheck %s --check-prefix=JSON-PASS < %t.pass.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph cmpf_graph --hardware-mlir %s --hardware predicate_mismatch_adg --workload cmpf_graph --output %t.fail.csv --artifact %t.fail.json
// RUN: FileCheck %s --check-prefix=CSV-FAIL < %t.fail.csv

// CSV-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-PASS-NEXT: cmpf_graph,predicate_adg,cmpf_graph__cmpf_graph__predicate_adg,1,0,0,0,pass

// JSON-PASS-DAG: "register": "sw_configs.predicate"
// JSON-PASS-DAG: "value": "ugt"
// JSON-PASS-DAG: "source": "placement:arith.cmpf#0"

// CSV-FAIL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-FAIL-NEXT: cmpf_graph,predicate_mismatch_adg,cmpf_graph__cmpf_graph__predicate_mismatch_adg,0,0,0,1,fail,missing hardware resource for software op arith.cmpf

module {
  dataflow.graph.func private @cmpf_graph(%ctrl: none, %lhs: f32,
                                          %rhs: f32) -> (none, i1) {
    %pred = arith.cmpf ugt, %lhs, %rhs : f32
    dataflow.graph.return %ctrl, %pred : none, i1
  }

  fabric.module @predicate_adg(%f32a : !fabric.bits<32>,
                               %f32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %f32a : !fabric.bits<32>,
                      %pb = %f32b : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %pred = fabric.op [@arith.cmpf] (%lhs, %rhs)
                {hw_params = [{predicate = ["ugt"]}]}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
        fabric.yield
      }
    }
    fabric.yield
  }

  fabric.module @predicate_mismatch_adg(%f32a : !fabric.bits<32>,
                                        %f32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %f32a : !fabric.bits<32>,
                      %pb = %f32b : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %pred = fabric.op [@arith.cmpf] (%lhs, %rhs)
                {hw_params = [{predicate = ["ule"]}]}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
        fabric.yield
      }
    }
    fabric.yield
  }
}

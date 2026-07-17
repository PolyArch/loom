// RUN: loom-pnr-map --dfg-mlir %s --graph typed_sync --hardware-mlir %s --hardware typed_sync_adg --workload typed_sync --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: typed_sync,typed_sync_adg,typed_sync__typed_sync__typed_sync_adg,2,1,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.sync#0.operand1"

module {
  dataflow.graph private @typed_sync(
      %ctrl: none, %lhs: i32, %rhs: i32) -> (i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %published:2 = dataflow.sync %ctrl, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  fabric.module @typed_sync_adg(
      %ctrl: !fabric.bits<0>, %lhs: !fabric.bits<32>,
      %rhs: !fabric.bits<32>) {
    %sum = fabric.pe [spatial] (
        %pa = %lhs : !fabric.bits<32>,
        %pb = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(
          %fa = %pa : !fabric.bits<32>,
          %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    %done, %value = fabric.pe [spatial] (
        %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>,
        %pv = %sum : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>) {
      fabric.fu(
          %fc = %pc : !fabric.bits<32> to !fabric.bits<0>,
          %fv = %pv : !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>) {
        %synced_ctrl, %synced_value = fabric.op [@dataflow.sync] (%fc, %fv)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
            -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %synced_ctrl : !fabric.bits<0> to !fabric.bits<32>,
                     %synced_value : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}

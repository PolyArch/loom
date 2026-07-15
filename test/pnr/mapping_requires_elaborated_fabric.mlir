// RUN: not loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %s --hardware named_mem_adg --workload named_mem --output %t.mapping.csv --artifact %t.mapping.json 2>&1 | FileCheck %s
// RUN: loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %s --hardware named_templates_ignored_adg --workload named_templates_ignored --output %t.ignored.csv --artifact %t.ignored.json
// RUN: FileCheck %s --check-prefix=IGNORED-CSV < %t.ignored.csv
// RUN: FileCheck %s --check-prefix=IGNORED-JSON < %t.ignored.json

// CHECK: PnR requires fully elaborated Fabric hardware; unresolved fabric.instantiate @MEM remains in @named_mem_adg
// IGNORED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// IGNORED-CSV-NEXT: named_templates_ignored,named_templates_ignored_adg,named_templates_ignored__named_mem_graph__named_templates_ignored_adg,1,0,0,0,pass,mapped software graph to fabric resources
// IGNORED-JSON-DAG: "status": "pass"
// IGNORED-JSON-DAG: "placed_records": 1

module {
  dataflow.graph.func private @named_mem_graph(
      %ctrl: none, %mem: memref<?xi32>, %index: index) -> (none, i32) {
    %value, %done = dataflow.load %mem[%index] %ctrl : memref<?xi32>
    dataflow.graph.return %done, %value : none, i32
  }

  fabric.module @named_mem_adg(%mgr : memref<?x!fabric.bits<32>>,
                               %addr : !fabric.bits<32>,
                               %ctrl : !fabric.bits<0>) {
    fabric.mem @MEM [spatial]
        (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
         -> (!fabric.bits<32>, !fabric.bits<0>)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
    %value, %done = fabric.instantiate @MEM(
        %mgr : memref<?x!fabric.bits<32>>,
        %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
    fabric.yield
  }

  fabric.module @named_templates_ignored_adg(
      %mgr : memref<?x!fabric.bits<32>>, %addr : !fabric.bits<32>,
      %ctrl : !fabric.bits<0>) {
    fabric.pe @UNUSED_PE [spatial] (!fabric.bits<32>)
        -> (!fabric.bits<32>) {
    ^bb0(%arg: !fabric.bits<32>):
      fabric.fu @UNUSED_FU (!fabric.bits<32>) -> (!fabric.bits<32>) {
      ^bb0(%fu_arg: !fabric.bits<32>):
        %value = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
      fabric.yield %arg : !fabric.bits<32>
    }
    fabric.switch @UNUSED_SWITCH [spatial]
        (!fabric.bits<32>) -> (!fabric.bits<32>)
        [{connectivity_table = ["1"]}]
    fabric.mem @UNUSED_MEM [spatial]
        (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
         -> (!fabric.bits<32>, !fabric.bits<0>)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
    %value, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
       -> (!fabric.bits<32>, !fabric.bits<0>)
    fabric.yield
  }
}

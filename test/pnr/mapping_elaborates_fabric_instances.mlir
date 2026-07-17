// RUN: loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %s --hardware named_mem_adg --workload named_mem --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=ELABORATED-CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=ELABORATED-JSON < %t.mapping.json
// RUN: loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %s --hardware named_mem_system --hardware-root-kind system --acc-core acc0 --workload named_mem_system --output %t.system.csv --artifact %t.system.json
// RUN: FileCheck %s --check-prefix=SYSTEM-JSON < %t.system.json
// RUN: loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %s --hardware named_templates_ignored_adg --workload named_templates_ignored --output %t.ignored.csv --artifact %t.ignored.json
// RUN: FileCheck %s --check-prefix=IGNORED-CSV < %t.ignored.csv
// RUN: FileCheck %s --check-prefix=IGNORED-JSON < %t.ignored.json
// RUN: rm -f %t.failed.csv %t.failed.json
// RUN: not loom-pnr-map --dfg-mlir %s --graph named_mem_graph --hardware-mlir %S/../fabric/elaboration/failure_atomicity.mlir --hardware later --workload failed_elaboration --output %t.failed.csv --artifact %t.failed.json 2>&1 | FileCheck %s --check-prefix=ELABORATION-ERROR
// RUN: test ! -e %t.failed.csv
// RUN: test ! -e %t.failed.json

// ELABORATED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ELABORATED-CSV-NEXT: named_mem,named_mem_adg,named_mem__named_mem_graph__named_mem_adg,1,0,0,0,pass,mapped software graph to fabric resources
// ELABORATED-JSON-DAG: "hardware": "named_mem_adg"
// ELABORATED-JSON-DAG: "status": "pass"
// ELABORATED-JSON-DAG: "placed_records": 1
// SYSTEM-JSON-DAG: "hardware": "named_mem_system::acc0"
// SYSTEM-JSON-DAG: "hardware_root_kind": "fabric.system"
// SYSTEM-JSON-DAG: "hardware_system": "named_mem_system"
// SYSTEM-JSON-DAG: "selected_acc_core": "acc0"
// SYSTEM-JSON-DAG: "spatialcore_template": "named_mem_adg"
// SYSTEM-JSON-DAG: "status": "pass"
// IGNORED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// IGNORED-CSV-NEXT: named_templates_ignored,named_templates_ignored_adg,named_templates_ignored__named_mem_graph__named_templates_ignored_adg,1,0,0,0,pass,mapped software graph to fabric resources
// IGNORED-JSON-DAG: "status": "pass"
// IGNORED-JSON-DAG: "placed_records": 1
// ELABORATION-ERROR: cannot inline fabric.module @callee because module-scoped semantic configuration differs
// ELABORATION-ERROR: PnR could not elaborate selected fabric.module @later

module {
  dataflow.graph.func private @named_mem_graph(
      %ctrl: none, %index: index, %mem: memref<?xi32>) -> (none, i32)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
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

  fabric.system @named_mem_system memory_model = "sequential" {
    fabric.node @acc0 kind = "acc_core"
        ports = ["mem.aw:output"]
        attributes {spatial = @named_mem_adg, scalar = "rv32im"}
  }
}

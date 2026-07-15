// RUN: loom-pnr-map --dfg-mlir %s --graph switch_route --hardware-mlir %s --hardware switch_forbidden_adg --workload switch_forbidden --output %t.forbidden.csv --artifact %t.forbidden.json
// RUN: FileCheck %s --check-prefix=FORBID-CSV < %t.forbidden.csv
// RUN: FileCheck %s --check-prefix=FORBID-JSON < %t.forbidden.json

// RUN: loom-pnr-map --dfg-mlir %s --graph switch_route --hardware-mlir %s --hardware switch_allowed_adg --workload switch_allowed --output %t.allowed.csv --artifact %t.allowed.json
// RUN: FileCheck %s --check-prefix=ALLOW-CSV < %t.allowed.csv
// RUN: FileCheck %s --check-prefix=ALLOW-JSON < %t.allowed.json

// FORBID-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FORBID-CSV-NEXT: switch_forbidden,switch_forbidden_adg,switch_forbidden__switch_route__switch_forbidden_adg,2,0,1,0,fail,unrouted software edges lack Fabric ADG connectivity

// FORBID-JSON-DAG: "status": "fail"
// FORBID-JSON-DAG: "routed_edges": 0
// FORBID-JSON-DAG: "unrouted_edges": 1
// FORBID-JSON-DAG: "routes": []

// ALLOW-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ALLOW-CSV-NEXT: switch_allowed,switch_allowed_adg,switch_allowed__switch_route__switch_allowed_adg,2,1,0,0,pass,mapped software graph to fabric resources

// ALLOW-JSON-DAG: "status": "pass"
// ALLOW-JSON-DAG: "segment_kind": "resource_edge"
// ALLOW-JSON-DAG: "segment_kind": "module_path"
// ALLOW-JSON-DAG: "source_endpoint": "switch_allowed_adg::mem.load#0.result0"
// ALLOW-JSON-DAG: "sink_endpoint": "switch_allowed_adg::fabric.switch#{{[0-9]+}}.operand1"
// ALLOW-JSON-DAG: "source_endpoint": "switch_allowed_adg::fabric.switch#{{[0-9]+}}.operand1"
// ALLOW-JSON-DAG: "sink_endpoint": "switch_allowed_adg::fabric.switch#{{[0-9]+}}.result0"
// ALLOW-JSON-DAG: "source_endpoint": "switch_allowed_adg::fabric.switch#{{[0-9]+}}.result0"
// ALLOW-JSON-DAG: "sink_endpoint": "switch_allowed_adg::fabric.op#0.operand0"
// ALLOW-JSON-NOT: ".out"
// ALLOW-JSON-NOT: ".in"

module {
  dataflow.graph.func private @switch_route(%ctrl: none, %mem: memref<?xi32>,
                                            %idx: index, %rhs: i32)
      -> (none, i32) {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %sum = arith.addi %data, %rhs : i32
    dataflow.graph.return %done, %sum : none, i32
  }

  fabric.module @switch_forbidden_adg(%mgr : memref<?x!fabric.bits<32>>,
                                      %addr : !fabric.bits<32>,
                                      %ctrl : !fabric.bits<0>,
                                      %rhs : !fabric.bits<32>) {
    %rhs_to_switch, %rhs_to_pe = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    %to_add, %unused = fabric.switch [spatial] %rhs_to_switch, %data
         [{connectivity_table = ["01", "10"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.pe [spatial] (%lhs = %to_add : !fabric.bits<32>,
                         %right = %rhs_to_pe : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %right : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.yield
  }

  fabric.module @switch_allowed_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %addr : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>,
                                    %rhs : !fabric.bits<32>) {
    %rhs_to_switch, %rhs_to_pe = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    %to_add, %unused = fabric.switch [spatial] %rhs_to_switch, %data
         [{connectivity_table = ["10", "01"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.pe [spatial] (%lhs = %to_add : !fabric.bits<32>,
                         %right = %rhs_to_pe : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %right : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.yield
  }
}

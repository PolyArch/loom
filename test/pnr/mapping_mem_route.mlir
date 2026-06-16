// RUN: loom-pnr-map --dfg-mlir %s --graph mem_route --hardware-mlir %s --hardware mem_route_adg --workload mem_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: mem_route,mem_route_adg,mem_route__mem_route__mem_route_adg,3,2,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "mem_route_adg::mem.load#0"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result0"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#0.operand0"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result1"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#1.operand0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_store_route --hardware-mlir %s --hardware mem_store_route_adg --workload mem_store_route --output %t.store.mapping.csv --artifact %t.store.mapping.json
// RUN: FileCheck %s --check-prefix=STORE-CSV < %t.store.mapping.csv
// RUN: FileCheck %s --check-prefix=STORE-JSON < %t.store.mapping.json

// STORE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// STORE-CSV-NEXT: mem_store_route,mem_store_route_adg,mem_store_route__mem_store_route__mem_store_route_adg,2,2,0,0,pass

// STORE-JSON-DAG: "status": "pass"
// STORE-JSON-DAG: "hardware": "mem_store_route_adg::mem.load#0"
// STORE-JSON-DAG: "hardware": "mem_store_route_adg::mem.store#0"
// STORE-JSON-DAG: "source_endpoint": "mem_store_route_adg::mem.load#0.result0"
// STORE-JSON-DAG: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand1"
// STORE-JSON-DAG: "source_endpoint": "mem_store_route_adg::mem.load#0.result1"
// STORE-JSON-DAG: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand2"
// STORE-JSON-NOT: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand3"
// STORE-JSON-NOT: ".out"
// STORE-JSON-NOT: ".in"

// RUN: %python %S/mapping_summary.py --dfg-mlir %s --graph mem_gep_store --hardware-mlir %s --hardware mem_store_route_adg --workload mem_gep_store --output %t.gep.mapping.csv --artifact %t.gep.mapping.json
// RUN: FileCheck %s --check-prefix=GEP-CSV < %t.gep.mapping.csv
// RUN: FileCheck %s --check-prefix=GEP-JSON < %t.gep.mapping.json

// GEP-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// GEP-CSV-NEXT: mem_gep_store,mem_store_route_adg,mem_gep_store__mem_gep_store__mem_store_route_adg,,,,,unsupported,unsupported PnR graph operation: llvm.getelementptr

// GEP-JSON-DAG: "status": "unsupported"
// GEP-JSON-DAG: "unsupported PnR graph operation: llvm.getelementptr"
// GEP-JSON-DAG: "placements": []
// GEP-JSON-DAG: "routes": []

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_pointer_bookkeeping --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_bookkeeping --output %t.ptr.mapping.csv --artifact %t.ptr.mapping.json
// RUN: FileCheck %s --check-prefix=PTR-CSV < %t.ptr.mapping.csv
// RUN: FileCheck %s --check-prefix=PTR-JSON < %t.ptr.mapping.json

// PTR-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTR-CSV-NEXT: mem_pointer_bookkeeping,shared_reduction_adg,mem_pointer_bookkeeping__mem_pointer_bookkeeping__shared_reduction_adg,5,6,0,0,pass

// PTR-JSON-DAG: "status": "pass"
// PTR-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// PTR-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PTR-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// PTR-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// PTR-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PTR-JSON-NOT: "operation": "llvm.getelementptr"
// PTR-JSON-NOT: "operation": "dataflow.carry"
// PTR-JSON-NOT: ".out"
// PTR-JSON-NOT: ".in"

// RUN: %python %S/mapping_summary.py --dfg-mlir %s --graph mem_pointer_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_return --output %t.ptrret.mapping.csv --artifact %t.ptrret.mapping.json
// RUN: FileCheck %s --check-prefix=PTRRET-CSV < %t.ptrret.mapping.csv
// RUN: FileCheck %s --check-prefix=PTRRET-JSON < %t.ptrret.mapping.json

// PTRRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTRRET-CSV-NEXT: mem_pointer_return,shared_reduction_adg,mem_pointer_return__mem_pointer_return__shared_reduction_adg,,,,,unsupported,graph returns unsupported pointer value for PnR mapping
// PTRRET-JSON-DAG: "status": "unsupported"
// PTRRET-JSON-DAG: "graph returns unsupported pointer value for PnR mapping"

module {
  dataflow.graph.func private @mem_route(%ctrl: none, %mem: memref<?xi32>,
                                         %idx: index, %rhs: i32)
      -> (none, i32) {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %sum = arith.addi %data, %rhs : i32
    %synced = dataflow.sync %done : (none) -> none
    dataflow.graph.return %synced, %sum : none, i32
  }

  fabric.module @mem_route_adg(%mgr : memref<?x!fabric.bits<32>>,
                               %addr : !fabric.bits<32>,
                               %ctrl : !fabric.bits<0>,
                               %rhs : !fabric.bits<32>) {
    %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    fabric.pe [spatial] (%lhs = %data : !fabric.bits<32>,
                         %right = %rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %right : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.pe [spatial] (%pc = %done : !fabric.bits<0>)
        -> !fabric.bits<0> {
      fabric.fu(%fc = %pc : !fabric.bits<0>) -> () {
        %synced = fabric.op [@dataflow.sync] (%fc)
                  {sw_configs = {bitmask = "1"}}
                  : (!fabric.bits<0>) -> !fabric.bits<0>
        fabric.yield
      }
    }
    fabric.yield
  }

  dataflow.graph.func private @mem_store_route(%ctrl: none, %mem: memref<?xi32>,
                                               %idx: index)
      -> (none) {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %stored = dataflow.store %mem[%idx] %data %done : memref<?xi32>
    dataflow.graph.return %stored : none
  }

  dataflow.graph.func private @mem_gep_store(%ctrl: none, %src: !llvm.ptr,
                                             %dst: !llvm.ptr, %idx: index)
      -> (none) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xi32>
    %dst_next = llvm.getelementptr inbounds|nuw %dst[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %dst_mem = builtin.unrealized_conversion_cast %dst_next : !llvm.ptr to memref<?xi32>
    %data, %done = dataflow.load %src_mem[%idx] %ctrl : memref<?xi32>
    %stored = dataflow.store %dst_mem[%idx] %data %done : memref<?xi32>
    dataflow.graph.return %stored : none
  }

  dataflow.graph.func private @mem_pointer_bookkeeping(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xf32>
    %dst_mem = builtin.unrealized_conversion_cast %dst : !llvm.ptr to memref<?xf32>
    %idx, %rwc = dataflow.stream %lb, %ub, %step {cont_cond = "<", step_op = "+="} : i32
    %addr = arith.index_cast %idx : i32 to index
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data, %done = dataflow.load %src_mem[%addr] %ctrl : memref<?xf32>
    %sum = arith.addf %data, %bias : f32
    %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %stored = dataflow.store %dst_mem[%addr] %sum %ctrl : memref<?xf32>
    %synced:2 = dataflow.sync %done, %stored : (none, none) -> (none, none)
    dataflow.graph.return %synced#0 : none
  }

  dataflow.graph.func private @mem_pointer_return(%ctrl: none, %ptr: !llvm.ptr)
      -> (none, !llvm.ptr) {
    dataflow.graph.return %ctrl, %ptr : none, !llvm.ptr
  }

  fabric.module @mem_store_route_adg(%mgr : memref<?x!fabric.bits<32>>,
                                     %addr : !fabric.bits<32>,
                                     %ctrl : !fabric.bits<0>) {
    %sub, %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
    %stored =
        fabric.mem [spatial] mgr(%sub) load() store(%addr, %data, %done)
          [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<32>,
             !fabric.bits<0>) -> !fabric.bits<0>
    fabric.yield
  }
}

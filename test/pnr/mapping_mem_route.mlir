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

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_two_loads_one_port --hardware-mlir %s --hardware mem_route_adg --workload mem_two_loads_one_port --output %t.twoload.mapping.csv --artifact %t.twoload.mapping.json
// RUN: FileCheck %s --check-prefix=TWOLOAD-CSV < %t.twoload.mapping.csv
// RUN: FileCheck %s --check-prefix=TWOLOAD-JSON < %t.twoload.mapping.json

// TWOLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// TWOLOAD-CSV-NEXT: mem_two_loads_one_port,mem_route_adg,mem_two_loads_one_port__mem_two_loads_one_port__mem_route_adg,2,1,0,1,fail,missing hardware resource for software op dataflow.load

// TWOLOAD-JSON-DAG: "status": "fail"
// TWOLOAD-JSON-DAG: "missing hardware resource for software op dataflow.load"
// TWOLOAD-JSON-DAG: "operation": "dataflow.load"
// TWOLOAD-JSON-DAG: "resource_kind": "fabric.mem.load"
// TWOLOAD-JSON-DAG: "unplaced_records": 1
// TWOLOAD-JSON-NOT: "fabric.mem.copy"
// TWOLOAD-JSON-NOT: "memory_copy_binding"

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

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_pointer_bookkeeping_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_bookkeeping_return --output %t.ptrbookret.mapping.csv --artifact %t.ptrbookret.mapping.json
// RUN: FileCheck %s --check-prefix=PTRBOOKRET-CSV < %t.ptrbookret.mapping.csv
// RUN: FileCheck %s --check-prefix=PTRBOOKRET-JSON < %t.ptrbookret.mapping.json

// PTRBOOKRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTRBOOKRET-CSV-NEXT: mem_pointer_bookkeeping_return,shared_reduction_adg,mem_pointer_bookkeeping_return__mem_pointer_bookkeeping_return__shared_reduction_adg,5,6,0,0,pass

// PTRBOOKRET-JSON-DAG: "status": "pass"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// PTRBOOKRET-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PTRBOOKRET-JSON-NOT: "operation": "llvm.getelementptr"
// PTRBOOKRET-JSON-NOT: "operation": "dataflow.carry"
// PTRBOOKRET-JSON-NOT: ".out"
// PTRBOOKRET-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_gep_bookkeeping_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_gep_bookkeeping_return --output %t.gepbookret.mapping.csv --artifact %t.gepbookret.mapping.json
// RUN: FileCheck %s --check-prefix=GEPBOOKRET-CSV < %t.gepbookret.mapping.csv
// RUN: FileCheck %s --check-prefix=GEPBOOKRET-JSON < %t.gepbookret.mapping.json

// GEPBOOKRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// GEPBOOKRET-CSV-NEXT: mem_gep_bookkeeping_return,shared_reduction_adg,mem_gep_bookkeeping_return__mem_gep_bookkeeping_return__shared_reduction_adg,5,6,0,0,pass

// GEPBOOKRET-JSON-DAG: "status": "pass"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// GEPBOOKRET-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// GEPBOOKRET-JSON-NOT: "operation": "llvm.getelementptr"
// GEPBOOKRET-JSON-NOT: "operation": "dataflow.carry"
// GEPBOOKRET-JSON-NOT: ".out"
// GEPBOOKRET-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph llvm_load_pointer --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload llvm_load_pointer --output %t.llvmload.mapping.csv --artifact %t.llvmload.mapping.json
// RUN: FileCheck %s --check-prefix=LLVMLOAD-CSV < %t.llvmload.mapping.csv
// RUN: FileCheck %s --check-prefix=LLVMLOAD-JSON < %t.llvmload.mapping.json

// LLVMLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LLVMLOAD-CSV-NEXT: llvm_load_pointer,shared_reduction_adg,llvm_load_pointer__llvm_load_pointer__shared_reduction_adg,2,1,0,0,pass

// LLVMLOAD-JSON-DAG: "operation": "llvm.load"
// LLVMLOAD-JSON-DAG: "resource_kind": "fabric.mem.load"
// LLVMLOAD-JSON-DAG: "edge_ref": "llvm.load#0.result0->arith.addi#0.operand0"
// LLVMLOAD-JSON-NOT: "operation": "llvm.getelementptr"
// LLVMLOAD-JSON-NOT: ".out"
// LLVMLOAD-JSON-NOT: ".in"

// RUN: %python %S/mapping_summary.py --dfg-mlir %s --graph mem_pointer_semantic_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_semantic_return --output %t.ptrsemantic.mapping.csv --artifact %t.ptrsemantic.mapping.json
// RUN: FileCheck %s --check-prefix=PTRSEM-CSV < %t.ptrsemantic.mapping.csv
// RUN: FileCheck %s --check-prefix=PTRSEM-JSON < %t.ptrsemantic.mapping.json

// PTRSEM-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTRSEM-CSV-NEXT: mem_pointer_semantic_return,shared_reduction_adg,mem_pointer_semantic_return__mem_pointer_semantic_return__shared_reduction_adg,,,,,unsupported,graph returns unsupported pointer value for PnR mapping
// PTRSEM-JSON-DAG: "status": "unsupported"
// PTRSEM-JSON-DAG: "graph returns unsupported pointer value for PnR mapping"

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

  dataflow.graph.func private @mem_two_loads_one_port(
      %ctrl: none, %mem: memref<?xi32>, %lhs_idx: index, %rhs_idx: index)
      -> (none, i32) {
    %lhs, %lhs_done = dataflow.load %mem[%lhs_idx] %ctrl : memref<?xi32>
    %rhs, %rhs_done = dataflow.load %mem[%rhs_idx] %ctrl : memref<?xi32>
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
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

  dataflow.graph.func private @mem_pointer_bookkeeping_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none, !llvm.ptr) {
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
    dataflow.graph.return %synced#0, %dst_cur : none, !llvm.ptr
  }

  dataflow.graph.func private @mem_gep_bookkeeping_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none, !llvm.ptr) {
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
    dataflow.graph.return %synced#0, %dst_next : none, !llvm.ptr
  }

  dataflow.graph.func private @llvm_load_pointer(%ctrl: none, %ptr: !llvm.ptr,
                                                 %rhs: i32) -> (none, i32) {
    %next = llvm.getelementptr inbounds|nuw %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %next {alignment = 4 : i64} : !llvm.ptr -> i32
    %sum = arith.addi %data, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @mem_pointer_semantic_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %src: !llvm.ptr)
      -> (none, !llvm.ptr, i32) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step {cont_cond = "<", step_op = "+="} : i32
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %bits = builtin.unrealized_conversion_cast %src_cur : !llvm.ptr to i32
    %sum = arith.addi %bits, %lb : i32
    dataflow.graph.return %ctrl, %src_cur, %sum : none, !llvm.ptr, i32
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

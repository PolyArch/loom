// RUN: loom-pnr-map --dfg-mlir %s --graph shared_reduction_conditional_store_tail --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload shared_reduction_conditional_store_tail --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: shared_reduction_conditional_store_tail,shared_reduction_adg,shared_reduction_conditional_store_tail__shared_reduction_conditional_store_tail__shared_reduction_adg,12,15,0,1,fail,missing hardware resource for software op dataflow.stream

// JSON-DAG: "status": "fail"
// JSON-DAG: "placed_records": 12
// JSON-DAG: "unplaced_records": 1
// JSON-DAG: "routed_edges": 15
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "missing hardware resource for software op dataflow.stream
// JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.select#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.constant#2.result0->dataflow.store#0.operand1"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.store#0.operand0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @shared_reduction_conditional_store_tail(
      %ctrl: none, %ub: i16, %lb: i16, %step: i16, %zero: i8,
      %ptr: !llvm.ptr) -> none {
    %mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xi8>
    %index, %rwc = dataflow.stream %ub, %lb, %step step add while sgt : i16
    %stable_zero = dataflow.invariant %rwc, %zero : i8
    %ptr_carried = dataflow.carry %rwc, %ptr, %next_ptr : !llvm.ptr
    %after_cond, %after_value = dataflow.gate %rwc, %ptr_carried : !llvm.ptr
    %zero_i16 = dataflow.constant %ctrl {const_value = 0 : i16} : i16
    %one_i16 = dataflow.constant %ctrl {const_value = 1 : i16} : i16
    %stable_one = dataflow.invariant %rwc, %one_i16 : i16
    %idx_carried = dataflow.carry %rwc, %zero_i16, %idx_next : i16
    %idx_next = arith.addi %idx_carried, %stable_one : i16
    %load_idx = arith.index_cast %idx_carried : i16 to index
    %data, %load_done = dataflow.load %mem[%load_idx] %ctrl : memref<?xi8>
    %is_negative = arith.cmpi slt, %data, %stable_zero : i8
    %store_mem = builtin.unrealized_conversion_cast %after_value : !llvm.ptr to memref<?xi8>
    %store_idx = dataflow.constant %ctrl {const_value = 0 : index} : index
    %selected = arith.select %is_negative, %stable_zero, %data : i8
    %store_done = dataflow.store %store_mem[%store_idx] %selected %ctrl : memref<?xi8>
    %next_ptr = llvm.getelementptr inbounds|nuw %after_value[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %done:2 = dataflow.sync %load_done, %store_done : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }
}

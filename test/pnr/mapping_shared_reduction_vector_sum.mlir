// RUN: loom-pnr-map --dfg-mlir %s --graph cmsis_vector_sum_s8 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cmsis_vector_sum_s8 --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: cmsis_vector_sum_s8,shared_reduction_adg,cmsis_vector_sum_s8__cmsis_vector_sum_s8__shared_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "cmsis_vector_sum_s8"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand0"
// JSON-DAG: "edge_ref": "arith.addi#1.result0->arith.muli#0.operand0"
// JSON-DAG: "edge_ref": "arith.muli#0.result0->arith.addi#2.operand1"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.addi#2.operand0"
// JSON-DAG: "edge_ref": "arith.addi#2.result0->dataflow.store#0.operand2"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @cmsis_vector_sum_s8(
      %ctrl: none, %lb: i32, %outer_ub: i32, %step: i32, %inner_ub: i32,
      %active: i1, %bias: i32, %scale: i32, %dst: !llvm.ptr,
      %src: !llvm.ptr) -> none {
    %0:2 = scf.for %outer = %lb to %outer_ub step %step
        iter_args(%dst_cur = %dst, %src_cur = %src)
        -> (!llvm.ptr, !llvm.ptr) : i32 {
      %1:2 = scf.if %active -> (!llvm.ptr, i32) {
        %11:2 = scf.for %inner = %lb to %inner_ub step %step
            iter_args(%acc = %lb, %src_inner = %src_cur)
            -> (i32, !llvm.ptr) : i32 {
          %src_next = llvm.getelementptr inbounds|nuw %src_inner[1]
              : (!llvm.ptr) -> !llvm.ptr, i8
          %src_mem = builtin.unrealized_conversion_cast %src_inner
              : !llvm.ptr to memref<?xi8>
          %zero_i8 = dataflow.constant %ctrl {const_value = 0 : index} : index
          %loaded_i8, %load_i8_done =
              dataflow.load %src_mem[%zero_i8] %ctrl : memref<?xi8>
          %loaded_i32 = llvm.sext %loaded_i8 : i8 to i32
          %next_acc = arith.addi %acc, %loaded_i32 : i32
          scf.yield %next_acc, %src_next : i32, !llvm.ptr
        }
        %after_inner = llvm.getelementptr %src_cur[%inner_ub]
            : (!llvm.ptr, i32) -> !llvm.ptr, i8
        scf.yield %after_inner, %11#0 : !llvm.ptr, i32
      } else {
        scf.yield %src_cur, %lb : !llvm.ptr, i32
      }
      %biased = arith.addi %1#1, %bias : i32
      %scaled = arith.muli %biased, %scale : i32
      %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4]
          : (!llvm.ptr) -> !llvm.ptr, i8
      %dst_load_mem = builtin.unrealized_conversion_cast %dst_cur
          : !llvm.ptr to memref<?xi32>
      %zero_load = dataflow.constant %ctrl {const_value = 0 : index} : index
      %old, %old_done = dataflow.load %dst_load_mem[%zero_load] %ctrl
          : memref<?xi32>
      %updated = arith.addi %old, %scaled : i32
      %dst_store_mem = builtin.unrealized_conversion_cast %dst_cur
          : !llvm.ptr to memref<?xi32>
      %zero_store = dataflow.constant %ctrl {const_value = 0 : index} : index
      %stored = dataflow.store %dst_store_mem[%zero_store] %updated %ctrl
          : memref<?xi32>
      scf.yield %dst_next, %1#0 : !llvm.ptr, !llvm.ptr
    } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
    dataflow.graph.return %ctrl : none
  }
}

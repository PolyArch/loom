// RUN: %python %S/mapping_summary.py --dfg-mlir %s --graph pointer_memcpy_stream --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pointer_memcpy_stream --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: pointer_memcpy_stream,shared_reduction_adg,pointer_memcpy_stream__pointer_memcpy_stream__shared_reduction_adg,,,,,unsupported,unsupported PnR graph operation: llvm.intr.memcpy

// JSON-DAG: "status": "unsupported"
// JSON-DAG: "unsupported PnR graph operation: llvm.intr.memcpy"
// JSON-DAG: "placements": []
// JSON-DAG: "routes": []

module {
  dataflow.graph.func private @pointer_memcpy_stream(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %copy_bytes: i32,
      %dst_stride: i32, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %idx, %rwc = dataflow.stream %lb, %ub, %step {cont_cond = "<", step_op = "+="} : i32
    %bytes = dataflow.invariant %rwc, %copy_bytes : i32
    %stride = dataflow.invariant %rwc, %dst_stride : i32
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %src_live_cond, %src_live = dataflow.gate %rwc, %src_cur : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %dst_live_cond, %dst_live = dataflow.gate %rwc, %dst_cur : !llvm.ptr
    "llvm.intr.memcpy"(%dst_live, %src_live, %bytes)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %src_next = llvm.getelementptr inbounds|nuw %src_live[%bytes]
      : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %dst_next = llvm.getelementptr inbounds|nuw %dst_live[%stride]
      : (!llvm.ptr, i32) -> !llvm.ptr, i8
    dataflow.graph.return %ctrl : none
  }
}

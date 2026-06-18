// RUN: loom-dfg-sim %s --graph pointer_memcpy_stream --arg 0=none --arg 1=0 --arg 2=2 --arg 3=1 --arg 4=2 --arg 5=2 --memref 6=1,2,3,4 --memref 7=0,0,0,0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --arg 0=none --memref 1=1,2,3,4 --memref 2=0,0,0,0 --arg 3=2 --arg 3=2 --output %t.direct.json
// RUN: FileCheck %s --check-prefix=DIRECT < %t.direct.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_stream --arg 0=none --arg 1=0 --arg 2=2 --arg 3=1 --arg 4=2 --arg 5=2 --memref 6=1,2,3,4 --memref 7=0,0,0 --output %t.oob.json
// RUN: FileCheck %s --check-prefix=OOB < %t.oob.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --memref 1=1,2 --memref 2=0,0 --arg 3=-1 --output %t.negative.json
// RUN: FileCheck %s --check-prefix=NEGATIVE < %t.negative.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --memref 1=1,2 --memref 2=0,0,0 --arg 3=3 --output %t.srcoob.json
// RUN: FileCheck %s --check-prefix=SRCOOB < %t.srcoob.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_overlap --arg 0=none --memref 1=1,2,3 --output %t.overlap.json
// RUN: FileCheck %s --check-prefix=OVERLAP < %t.overlap.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_non_i8 --arg 0=none --memref 1=1.000000e+00,2.000000e+00 --memref 2=0,0 --output %t.type.json
// RUN: FileCheck %s --check-prefix=TYPE < %t.type.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_volatile --arg 0=none --memref 1=1,2 --memref 2=0,0 --output %t.volatile.json
// RUN: FileCheck %s --check-prefix=VOLATILE < %t.volatile.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "graph": "pointer_memcpy_stream"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 2
// CHECK-DAG: "llvm.intr.memcpy": 2
// CHECK-DAG: "arg6": [
// CHECK-DAG: "i8:1"
// CHECK-DAG: "i8:2"
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i8:4"
// CHECK-DAG: "arg7": [
// CHECK-DAG: "i8:1"
// CHECK-DAG: "i8:2"
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i8:4"
// CHECK-DAG: "i8:0"
// CHECK-DAG: "i8:0"

// DIRECT: "dynamic_work_items": 2
// DIRECT: "final_memory_state": {
// DIRECT: "arg2": [
// DIRECT-NEXT: "i8:1",
// DIRECT-NEXT: "i8:2",
// DIRECT-NEXT: "i8:0",
// DIRECT-NEXT: "i8:0"
// DIRECT: "llvm.intr.memcpy": 2
// DIRECT: "status": "pass"

// OOB-DAG: "status": "blocked"
// OOB-DAG: "llvm.intr.memcpy destination range is out of range"
// OOB-DAG: "llvm.intr.memcpy": 1
// OOB-DAG: "arg7": [
// OOB-DAG: "i8:1"
// OOB-DAG: "i8:2"
// OOB-DAG: "i8:0"

// NEGATIVE-DAG: "status": "blocked"
// NEGATIVE-DAG: "llvm.intr.memcpy length is negative"
// NEGATIVE-NOT: "llvm.intr.memcpy": 1

// SRCOOB-DAG: "status": "blocked"
// SRCOOB-DAG: "llvm.intr.memcpy source range is out of range"
// SRCOOB-NOT: "llvm.intr.memcpy": 1

// OVERLAP-DAG: "status": "blocked"
// OVERLAP-DAG: "llvm.intr.memcpy overlapping ranges are unsupported"
// OVERLAP-NOT: "llvm.intr.memcpy": 1

// TYPE-DAG: "status": "blocked"
// TYPE-DAG: "memory fixture type mismatch: existing f32, requested i8"
// TYPE-NOT: "llvm.intr.memcpy": 1

// VOLATILE-DAG: "status": "blocked"
// VOLATILE-DAG: "volatile llvm.intr.memcpy is unsupported"
// VOLATILE-NOT: "llvm.intr.memcpy": 1

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

  dataflow.graph.func private @pointer_memcpy_direct(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr, %copy_bytes: i32)
      -> none {
    "llvm.intr.memcpy"(%dst, %src, %copy_bytes)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_overlap(
      %ctrl: none, %base: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %dst = llvm.getelementptr inbounds|nuw %base[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst, %base, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_non_i8(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %loaded = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_volatile(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }
}

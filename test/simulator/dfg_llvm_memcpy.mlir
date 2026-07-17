// RUN: loom-dfg-sim %s --graph pointer_memcpy_stream --arg 0=none --arg 1=0 --arg 2=2 --arg 3=1 --arg 4=2 --arg 5=2 --memref 6=1,2,3,4 --memref 7=0,0,0,0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --arg 0=none --memref 1=1,2,3,4 --memref 2=0,0,0,0 --arg 3=2 --arg 3=2 --output %t.direct.json
// RUN: FileCheck %s --check-prefix=DIRECT < %t.direct.json
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered-direct.mlir
// RUN: FileCheck %s --check-prefix=DIRECT-LOWERED-IR < %t.lowered-direct.mlir
// RUN: loom-dfg-sim %t.lowered-direct.mlir --graph pointer_memcpy_direct --arg 0=none --arg 0=none --memref 1=1,2,3,4 --memref 2=0,0,0,0 --arg 3=2 --output %t.direct-lowered.json
// RUN: FileCheck %s --check-prefix=DIRECT-LOWERED < %t.direct-lowered.json
// RUN: FileCheck %s --check-prefix=DIRECT-OFFSET-LOWERED-IR < %t.lowered-direct.mlir
// RUN: loom-dfg-sim %t.lowered-direct.mlir --graph pointer_memcpy_direct_offset --arg 0=none --arg 0=none --memref 1=1,2,3,4 --memref 2=0,0,0,0,0,0 --arg 3=2 --arg 4=2 --output %t.direct-offset-lowered.json
// RUN: FileCheck %s --check-prefix=DIRECT-OFFSET-LOWERED < %t.direct-offset-lowered.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_structured_if --arg 0=none --memref 1=5,6,7 --memref 2=0,0,0 --arg 3=2 --arg 4=true --output %t.structured-if.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-IF < %t.structured-if.json
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_structured_i32_gep --arg 0=none --arg 1=2 --arg 2=true --arg 3=1 --memref 4=10,11,12,13,20,21,22,23 --memref 5=0,0,0,0 --output %t.structured-i32-gep.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-I32-GEP < %t.structured-i32-gep.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_stream --arg 0=none --arg 1=0 --arg 2=2 --arg 3=1 --arg 4=2 --arg 5=2 --memref 6=1,2,3,4 --memref 7=0,0,0 --output %t.oob.json
// RUN: FileCheck %s --check-prefix=OOB < %t.oob.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --memref 1=1,2 --memref 2=0,0 --arg 3=-1 --output %t.negative.json
// RUN: FileCheck %s --check-prefix=NEGATIVE < %t.negative.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_direct --arg 0=none --memref 1=1,2 --memref 2=0,0,0 --arg 3=3 --output %t.srcoob.json
// RUN: FileCheck %s --check-prefix=SRCOOB < %t.srcoob.json

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

// DIRECT-LOWERED-IR-LABEL: dataflow.graph.func private @pointer_memcpy_direct
// DIRECT-LOWERED-IR: dataflow.stream
// DIRECT-LOWERED-IR: dataflow.load
// DIRECT-LOWERED-IR: dataflow.store
// DIRECT-LOWERED-IR-NOT: llvm.intr.memcpy
// DIRECT-LOWERED-IR: dataflow.graph.return

// DIRECT-LOWERED: "dynamic_work_items": 2
// DIRECT-LOWERED: "final_memory_state": {
// DIRECT-LOWERED: "arg2": [
// DIRECT-LOWERED-NEXT: "i8:1",
// DIRECT-LOWERED-NEXT: "i8:2",
// DIRECT-LOWERED-NEXT: "i8:0",
// DIRECT-LOWERED-NEXT: "i8:0"
// DIRECT-LOWERED: "dataflow.load": 2
// DIRECT-LOWERED: "dataflow.store": 2
// DIRECT-LOWERED-NOT: "llvm.intr.memcpy"
// DIRECT-LOWERED: "status": "pass"

// DIRECT-OFFSET-LOWERED-IR-LABEL: dataflow.graph.func private @pointer_memcpy_direct_offset
// DIRECT-OFFSET-LOWERED-IR: dataflow.stream
// DIRECT-OFFSET-LOWERED-IR: dataflow.invariant
// DIRECT-OFFSET-LOWERED-IR: arith.addi
// DIRECT-OFFSET-LOWERED-IR: dataflow.load
// DIRECT-OFFSET-LOWERED-IR: dataflow.store
// DIRECT-OFFSET-LOWERED-IR-NOT: llvm.intr.memcpy
// DIRECT-OFFSET-LOWERED-IR: dataflow.graph.return

// DIRECT-OFFSET-LOWERED: "dynamic_work_items": 2
// DIRECT-OFFSET-LOWERED: "final_memory_state": {
// DIRECT-OFFSET-LOWERED: "arg2": [
// DIRECT-OFFSET-LOWERED-NEXT: "i8:0",
// DIRECT-OFFSET-LOWERED-NEXT: "i8:0",
// DIRECT-OFFSET-LOWERED-NEXT: "i8:1",
// DIRECT-OFFSET-LOWERED-NEXT: "i8:2",
// DIRECT-OFFSET-LOWERED-NEXT: "i8:0",
// DIRECT-OFFSET-LOWERED-NEXT: "i8:0"
// DIRECT-OFFSET-LOWERED: "dataflow.load": 2
// DIRECT-OFFSET-LOWERED: "dataflow.store": 2
// DIRECT-OFFSET-LOWERED-NOT: "llvm.intr.memcpy"
// DIRECT-OFFSET-LOWERED: "status": "pass"

// STRUCTURED-IF: "dynamic_work_items": 1
// STRUCTURED-IF: "final_memory_state": {
// STRUCTURED-IF: "arg2": [
// STRUCTURED-IF-NEXT: "i8:5",
// STRUCTURED-IF-NEXT: "i8:6",
// STRUCTURED-IF-NEXT: "i8:0"
// STRUCTURED-IF: "llvm.intr.memcpy": 1
// STRUCTURED-IF: "scf.if": 1
// STRUCTURED-IF: "status": "pass"

// STRUCTURED-I32-GEP: "final_memory_state": {
// STRUCTURED-I32-GEP: "arg5": [
// STRUCTURED-I32-GEP-NEXT: "i8:20",
// STRUCTURED-I32-GEP-NEXT: "i8:21",
// STRUCTURED-I32-GEP-NEXT: "i8:0",
// STRUCTURED-I32-GEP-NEXT: "i8:0"
// STRUCTURED-I32-GEP-NOT: "llvm.intr.memcpy"
// STRUCTURED-I32-GEP: "status": "pass"

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

module {
  dataflow.graph.func private @pointer_memcpy_stream(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %copy_bytes: i32,
      %dst_stride: i32, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
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

  dataflow.graph.func private @pointer_memcpy_direct_offset(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr, %copy_bytes: i32,
      %dst_offset: i32) -> none {
    %dst_at = llvm.getelementptr inbounds|nuw %dst[%dst_offset]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst_at, %src, %copy_bytes)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_structured_if(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr, %copy_bytes: i32,
      %do_copy: i1) -> none {
    scf.if %do_copy {
      "llvm.intr.memcpy"(%dst, %src, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_structured_i32_gep(
      %ctrl: none, %copy_bytes: i32, %do_copy: i1, %elem_offset: i32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    scf.if %do_copy {
      %src_at = llvm.getelementptr %src[%elem_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i32
      "llvm.intr.memcpy"(%dst, %src_at, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }

}

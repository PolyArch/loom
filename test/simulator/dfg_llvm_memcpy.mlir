// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_stream --arg 0=0 --arg 1=2 --arg 2=1 --arg 3=2 --arg 4=2 --memref 5=1,2,3,4 --memref 6=0,0,0,0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_direct --arg 0=2 --memref 1=1,2,3,4 --memref 2=0,0,0,0 --output %t.direct.json
// RUN: FileCheck %s --check-prefix=DIRECT < %t.direct.json
// RUN: FileCheck %s --check-prefix=DIRECT-LOWERED-IR < %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=DIRECT-OFFSET-LOWERED-IR < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_direct_offset --arg 0=2 --arg 1=2 --memref 2=1,2,3,4 --memref 3=0,0,0,0,0,0 --output %t.direct-offset-lowered.json
// RUN: FileCheck %s --check-prefix=DIRECT-OFFSET-LOWERED < %t.direct-offset-lowered.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_structured_if --arg 0=2 --arg 1=true --memref 2=5,6,7 --memref 3=0,0,0 --output %t.structured-if.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-IF < %t.structured-if.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_structured_i32_gep --arg 0=2 --arg 1=true --arg 2=1 --memref 3=10,11,12,13,20,21,22,23 --memref 4=0,0,0,0 --output %t.structured-i32-gep.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-I32-GEP < %t.structured-i32-gep.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_stream --arg 0=0 --arg 1=2 --arg 2=1 --arg 3=2 --arg 4=2 --memref 5=1,2,3,4 --memref 6=0,0,0 --output %t.oob.json
// RUN: FileCheck %s --check-prefix=OOB < %t.oob.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_direct --arg 0=-1 --memref 1=1,2 --memref 2=0,0 --output %t.negative.json
// RUN: FileCheck %s --check-prefix=NEGATIVE < %t.negative.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_memcpy_direct --arg 0=3 --memref 1=1,2 --memref 2=0,0,0 --output %t.srcoob.json
// RUN: FileCheck %s --check-prefix=SRCOOB < %t.srcoob.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "graph": "pointer_memcpy_stream"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 2
// CHECK-DAG: "dataflow.load": 4
// CHECK-DAG: "dataflow.store": 4
// CHECK-DAG: "arg5": [
// CHECK-DAG: "i8:1"
// CHECK-DAG: "i8:2"
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i8:4"
// CHECK-DAG: "arg6": [
// CHECK-DAG: "i8:1"
// CHECK-DAG: "i8:2"
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i8:4"
// CHECK-DAG: "i8:0"
// CHECK-DAG: "i8:0"

// DIRECT: "dynamic_work_items": 1
// DIRECT: "final_memory_state": {
// DIRECT: "arg2": [
// DIRECT-NEXT: "i8:1",
// DIRECT-NEXT: "i8:2",
// DIRECT-NEXT: "i8:0",
// DIRECT-NEXT: "i8:0"
// DIRECT: "dataflow.load": 2
// DIRECT: "dataflow.store": 2
// DIRECT: "status": "pass"

// DIRECT-LOWERED-IR-LABEL: dataflow.graph.func private @pointer_memcpy_direct
// DIRECT-LOWERED-IR: arith.cmpi ult
// DIRECT-LOWERED-IR: dataflow.carry
// DIRECT-LOWERED-IR: dataflow.load
// DIRECT-LOWERED-IR: dataflow.store
// DIRECT-LOWERED-IR-NOT: llvm.intr.memcpy
// DIRECT-LOWERED-IR-NOT: scf.while
// DIRECT-LOWERED-IR: dataflow.graph.return

// DIRECT-OFFSET-LOWERED-IR-LABEL: dataflow.graph.func private @pointer_memcpy_direct_offset
// DIRECT-OFFSET-LOWERED-IR: arith.cmpi ult
// DIRECT-OFFSET-LOWERED-IR: dataflow.carry
// DIRECT-OFFSET-LOWERED-IR: dataflow.invariant
// DIRECT-OFFSET-LOWERED-IR: arith.addi
// DIRECT-OFFSET-LOWERED-IR: dataflow.load
// DIRECT-OFFSET-LOWERED-IR: dataflow.store
// DIRECT-OFFSET-LOWERED-IR-NOT: llvm.intr.memcpy
// DIRECT-OFFSET-LOWERED-IR: dataflow.graph.return

// DIRECT-OFFSET-LOWERED: "dynamic_work_items": 1
// DIRECT-OFFSET-LOWERED: "final_memory_state": {
// DIRECT-OFFSET-LOWERED: "arg3": [
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
// STRUCTURED-IF: "arg3": [
// STRUCTURED-IF-NEXT: "i8:5",
// STRUCTURED-IF-NEXT: "i8:6",
// STRUCTURED-IF-NEXT: "i8:0"
// STRUCTURED-IF: "dataflow.load": 2
// STRUCTURED-IF: "dataflow.store": 2
// STRUCTURED-IF-NOT: "llvm.intr.memcpy"
// STRUCTURED-IF-NOT: "scf.if"
// STRUCTURED-IF: "status": "pass"

// STRUCTURED-I32-GEP: "final_memory_state": {
// STRUCTURED-I32-GEP: "arg4": [
// STRUCTURED-I32-GEP-NEXT: "i8:20",
// STRUCTURED-I32-GEP-NEXT: "i8:21",
// STRUCTURED-I32-GEP-NEXT: "i8:0",
// STRUCTURED-I32-GEP-NEXT: "i8:0"
// STRUCTURED-I32-GEP-NOT: "llvm.intr.memcpy"
// STRUCTURED-I32-GEP: "status": "pass"

// OOB-DAG: "status": "blocked"
// OOB-DAG: "dataflow.store address is out of range"
// OOB-DAG: "dataflow.load": 4
// OOB-DAG: "dataflow.store": 3
// OOB-DAG: "arg6": [
// OOB-DAG: "i8:1"
// OOB-DAG: "i8:2"
// OOB-DAG: "i8:3"

// NEGATIVE-DAG: "status": "blocked"
// NEGATIVE-DAG: "graph did not fire its retirement frontier"
// NEGATIVE-DAG: "dataflow.load address is out of range"
// NEGATIVE-DAG: "dataflow.load": 2
// NEGATIVE-DAG: "dataflow.store": 2
// NEGATIVE-NOT: maximum event steps reached

// SRCOOB-DAG: "status": "blocked"
// SRCOOB-DAG: "dataflow.load address is out of range"
// SRCOOB-DAG: "dataflow.load": 2
// SRCOOB-DAG: "dataflow.store": 2

module {
  dataflow.graph.func private @pointer_memcpy_stream(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %copy_bytes: i32,
      %dst_stride: i32, %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 5, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.for %i = %lb to %ub step %step : i32 {
      %src_offset = arith.muli %i, %copy_bytes : i32
      %dst_offset = arith.muli %i, %dst_stride : i32
      %src_at = llvm.getelementptr inbounds|nuw %src[%src_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      %dst_at = llvm.getelementptr inbounds|nuw %dst[%dst_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      "llvm.intr.memcpy"(%dst_at, %src_at, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_direct(
      %ctrl: none, %copy_bytes: i32, %src: !llvm.ptr, %dst: !llvm.ptr)
      -> none attributes {input_segments = array<i32: 1, 0, 2>,
                          result_segments = array<i32: 0, 0, 0>} {
    "llvm.intr.memcpy"(%dst, %src, %copy_bytes)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_direct_offset(
      %ctrl: none, %copy_bytes: i32, %dst_offset: i32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %dst_at = llvm.getelementptr inbounds|nuw %dst[%dst_offset]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst_at, %src, %copy_bytes)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_structured_if(
      %ctrl: none, %copy_bytes: i32, %do_copy: i1,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.if %do_copy {
      "llvm.intr.memcpy"(%dst, %src, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_structured_i32_gep(
      %ctrl: none, %copy_bytes: i32, %do_copy: i1, %elem_offset: i32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 3, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
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

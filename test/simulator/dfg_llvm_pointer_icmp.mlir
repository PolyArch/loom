// RUN: loom-dfg-sim %s --graph pointer_icmp_top --memref 0=1 --memref 1=2 --output %t.top.json
// RUN: FileCheck %s --check-prefix=TOP < %t.top.json
// RUN: loom-dfg-sim %s --graph pointer_icmp_structured --arg 0=true --memref 1=1 --memref 2=2 --output %t.structured.json
// RUN: FileCheck %s --check-prefix=STRUCTURED < %t.structured.json

// TOP-DAG: "graph": "pointer_icmp_top"
// TOP-DAG: "status": "pass"
// TOP-DAG: "llvm.mlir.zero": 1
// TOP-DAG: "llvm.icmp": 6
// TOP-DAG: "final_outputs": [
// TOP-DAG: "none"
// TOP-DAG: "i1:true"
// TOP-DAG: "i1:false"
// TOP-DAG: "i1:false"
// TOP-DAG: "i1:true"
// TOP-DAG: "i1:true"
// TOP-DAG: "i1:false"

// STRUCTURED-DAG: "graph": "pointer_icmp_structured"
// STRUCTURED-DAG: "status": "pass"
// STRUCTURED-DAG: "llvm.mlir.zero": 1
// STRUCTURED-DAG: "llvm.icmp": 3
// STRUCTURED-DAG: "final_outputs": [
// STRUCTURED-DAG: "none"
// STRUCTURED-DAG: "i1:true"
// STRUCTURED-DAG: "i1:false"
// STRUCTURED-DAG: "i1:true"

module {
  dataflow.graph.func private @pointer_icmp_top(
      %ctrl: none, %lhs: !llvm.ptr, %rhs: !llvm.ptr)
      -> (none, i1, i1, i1, i1, i1, i1)
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 6, 0, 0>} {
    %null = llvm.mlir.zero : !llvm.ptr
    %same_eq = llvm.icmp "eq" %lhs, %lhs : !llvm.ptr
    %diff_eq = llvm.icmp "eq" %lhs, %rhs : !llvm.ptr
    %same_ne = llvm.icmp "ne" %lhs, %lhs : !llvm.ptr
    %diff_ne = llvm.icmp "ne" %lhs, %rhs : !llvm.ptr
    %null_eq = llvm.icmp "eq" %null, %null : !llvm.ptr
    %arg_null_eq = llvm.icmp "eq" %lhs, %null : !llvm.ptr
    %published:7 = dataflow.sync %ctrl, %same_eq, %diff_eq, %same_ne,
        %diff_ne, %null_eq, %arg_null_eq
        : (none, i1, i1, i1, i1, i1, i1)
          -> (none, i1, i1, i1, i1, i1, i1)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5, %published#6
        : none, i1, i1, i1, i1, i1, i1
  }

  dataflow.graph.func private @pointer_icmp_structured(
      %ctrl: none, %cond: i1, %lhs: !llvm.ptr, %rhs: !llvm.ptr)
      -> (none, i1, i1, i1)
      attributes {input_segments = array<i32: 1, 0, 2>,
                  result_segments = array<i32: 3, 0, 0>} {
    %null = llvm.mlir.zero : !llvm.ptr
    %same_eq = llvm.icmp "eq" %lhs, %lhs : !llvm.ptr
    %arg_null_eq = llvm.icmp "eq" %lhs, %null : !llvm.ptr
    %diff_ne = llvm.icmp "ne" %lhs, %rhs : !llvm.ptr
    %true = arith.constant true
    %diff_eq = arith.xori %diff_ne, %true : i1
    %selected = arith.select %cond, %diff_ne, %diff_eq : i1
    %published:4 = dataflow.sync %ctrl, %same_eq, %arg_null_eq, %selected
        : (none, i1, i1, i1) -> (none, i1, i1, i1)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3 : none, i1, i1, i1
  }
}

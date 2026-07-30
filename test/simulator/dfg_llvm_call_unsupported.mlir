// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/direct.mlir --graph calls_external --output %t.json 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: not loom-dfg-sim %t.dir/intrinsic.mlir --graph generic_intrinsic --output %t.intrinsic.json 2>&1 | FileCheck %s --check-prefix=INTRINSIC
// RUN: not loom-dfg-sim %t.dir/asm.mlir --graph generic_inline_asm --output %t.asm.json 2>&1 | FileCheck %s --check-prefix=ASM

// DIRECT: finalized graph contains unregistered actor 'llvm.call'

// INTRINSIC: finalized graph contains unregistered actor 'llvm.call_intrinsic'

// ASM: finalized graph contains unregistered actor 'llvm.inline_asm'

//--- direct.mlir
module {
  llvm.func @opaque_callee(i32) -> i32

  dataflow.graph private @calls_external(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = llvm.call @opaque_callee(%value) : (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- intrinsic.mlir
module {
  dataflow.graph private @generic_intrinsic(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %result = llvm.call_intrinsic "llvm.arm.qadd16"(%lhs, %rhs)
        : (i32, i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- asm.mlir
module {
  dataflow.graph private @generic_inline_asm(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %amount = dataflow.constant %ctrl {const_value = 16 : i32} : i32
    %result = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "pkhbt $0, $1, $2, lsl $3", "=r,r,r,I" %lhs, %rhs, %amount
        : (i32, i32, i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

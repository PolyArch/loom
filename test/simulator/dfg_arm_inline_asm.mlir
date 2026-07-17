// RUN: loom-dfg-sim %s --graph arm_inline_asm --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "arm_inline_asm"
// CHECK-DAG: "graph": "arm_inline_asm"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "llvm.arm.pkhbt": 1
// CHECK-DAG: "llvm.arm.pkhtb": 1
// CHECK-DAG: "llvm.arm.sxtab16": 1
// CHECK-DAG: "llvm.arm.sxtb16": 1
// CHECK-DAG: "i32:262146"
// CHECK-DAG: "i32:305463278"
// CHECK-DAG: "i32:65539"
// CHECK-DAG: "i32:-65535"

module {
  dataflow.graph.func private @arm_inline_asm(%ctrl: none)
      -> (none, i32, i32, i32, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 4, 0, 0>} {
    %pack_low = dataflow.constant %ctrl {const_value = 65538 : i32} : i32
    %pack_high = dataflow.constant %ctrl {const_value = 196612 : i32} : i32
    %shift16 = dataflow.constant %ctrl {const_value = 16 : i32} : i32
    %pkhbt = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "pkhbt $0, $1, $2, lsl $3", "=r,r,r,I" %pack_low, %pack_high, %shift16
        : (i32, i32, i32) -> i32

    %top_half = dataflow.constant %ctrl {const_value = 305419896 : i32} : i32
    %bottom_shifted = dataflow.constant %ctrl {const_value = -1179390 : i32} : i32
    %pkhtb = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "pkhtb $0, $1, $2, asr $3", "=r,r,r,I" %top_half, %bottom_shifted, %shift16
        : (i32, i32, i32) -> i32

    %offset = dataflow.constant %ctrl {const_value = 131074 : i32} : i32
    %bytes = dataflow.constant %ctrl {const_value = 16711681 : i32} : i32
    %sxtab16 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "sxtab16 $0, $1, $2", "=r,r,r" %offset, %bytes
        : (i32, i32) -> i32
    %sxtb16 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "sxtb16 $0, $1", "=r,r" %bytes : (i32) -> i32

    %published:5 = dataflow.sync %ctrl, %pkhbt, %pkhtb, %sxtab16, %sxtb16
        : (none, i32, i32, i32, i32) -> (none, i32, i32, i32, i32)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4 : none, i32, i32, i32, i32
  }
}

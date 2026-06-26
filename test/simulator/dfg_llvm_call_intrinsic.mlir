// RUN: loom-dfg-sim %s --graph arm_qsub16 --arg 0=none --output %t.qsub16.json
// RUN: FileCheck %s --check-prefix=QSUB16 < %t.qsub16.json
// RUN: loom-dfg-sim %s --graph arm_qsub16_saturate --arg 0=none --output %t.qsub16.saturate.json
// RUN: FileCheck %s --check-prefix=QSUB16-SAT < %t.qsub16.saturate.json
// RUN: loom-dfg-sim %s --graph arm_qsub8_saturate --arg 0=none --output %t.qsub8.saturate.json
// RUN: FileCheck %s --check-prefix=QSUB8-SAT < %t.qsub8.saturate.json
// RUN: loom-dfg-sim %s --graph arm_qadd16_saturate --arg 0=none --output %t.qadd16.saturate.json
// RUN: FileCheck %s --check-prefix=QADD16-SAT < %t.qadd16.saturate.json

// QSUB16-DAG: "workload": "arm_qsub16"
// QSUB16-DAG: "graph": "arm_qsub16"
// QSUB16-DAG: "status": "pass"
// QSUB16-DAG: "optimistic_cycles": 5
// QSUB16-DAG: "event_count": 3
// QSUB16-DAG: "llvm.arm.qsub16": 1
// QSUB16-DAG: "i32:-1"

// QSUB16-SAT-DAG: "workload": "arm_qsub16_saturate"
// QSUB16-SAT-DAG: "graph": "arm_qsub16_saturate"
// QSUB16-SAT-DAG: "status": "pass"
// QSUB16-SAT-DAG: "llvm.arm.qsub16": 1
// QSUB16-SAT-DAG: "i32:2147450880"

// QSUB8-SAT-DAG: "workload": "arm_qsub8_saturate"
// QSUB8-SAT-DAG: "graph": "arm_qsub8_saturate"
// QSUB8-SAT-DAG: "status": "pass"
// QSUB8-SAT-DAG: "llvm.arm.qsub8": 1
// QSUB8-SAT-DAG: "i32:2139029888"

// QADD16-SAT-DAG: "workload": "arm_qadd16_saturate"
// QADD16-SAT-DAG: "graph": "arm_qadd16_saturate"
// QADD16-SAT-DAG: "status": "pass"
// QADD16-SAT-DAG: "llvm.arm.qadd16": 1
// QADD16-SAT-DAG: "i32:2147450879"

module {
  dataflow.graph.func private @arm_qsub16(%ctrl: none) -> (none, i32) {
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %value = dataflow.constant %ctrl {const_value = 65537 : i32} : i32
    %packed = llvm.call_intrinsic "llvm.arm.qsub16"(%zero, %value)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_qsub16_saturate(%ctrl: none) -> (none, i32) {
    %lhs = dataflow.constant %ctrl {const_value = 2147450880 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = -65535 : i32} : i32
    %packed = llvm.call_intrinsic "llvm.arm.qsub16"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_qsub8_saturate(%ctrl: none) -> (none, i32) {
    %lhs = dataflow.constant %ctrl {const_value = 25100416 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = -2130706687 : i32} : i32
    %packed = llvm.call_intrinsic "llvm.arm.qsub8"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_qadd16_saturate(%ctrl: none) -> (none, i32) {
    %lhs = dataflow.constant %ctrl {const_value = 2147450879 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 65537 : i32} : i32
    %packed = llvm.call_intrinsic "llvm.arm.qadd16"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }
}

// RUN: loom-dfg-sim %s --graph carry_address_relu_tail --arg 0=none --arg 1=3 --arg 2=0 --arg 3=-1 --arg 4=0 --memref 5=-1,2,-3 --output %t.blocked.json
// RUN: FileCheck %s --check-prefix=BLOCKED < %t.blocked.json
// RUN: loom-dfg-sim %s --graph carry_address_relu_tail --arg 0=none --arg 0=none --arg 0=none --arg 1=3 --arg 2=0 --arg 3=-1 --arg 4=0 --memref 5=-1,2,-3 --output %t.pass.json
// RUN: FileCheck %s --check-prefix=PASS < %t.pass.json

// BLOCKED-DAG: "dataflow.load consumed 1 of 3 true stream indices"
// BLOCKED: "dynamic_work_items": 3
// BLOCKED: "final_memory_state": {
// BLOCKED-NEXT: "arg5": [
// BLOCKED-NEXT: "i8:0",
// BLOCKED-NEXT: "i8:2",
// BLOCKED-NEXT: "i8:-3"
// BLOCKED-NEXT: ]
// BLOCKED-NEXT: }
// BLOCKED-DAG: "dataflow.load": 1
// BLOCKED-DAG: "dataflow.store": 1
// BLOCKED-DAG: "status": "blocked"

// PASS: "dynamic_work_items": 3
// PASS: "final_memory_state": {
// PASS-NEXT: "arg5": [
// PASS-NEXT: "i8:0",
// PASS-NEXT: "i8:2",
// PASS-NEXT: "i8:0"
// PASS-NEXT: ]
// PASS-NEXT: }
// PASS-DAG: "dataflow.load": 3
// PASS-DAG: "arith.cmpi": 3
// PASS-DAG: "arith.select": 3
// PASS-DAG: "dataflow.store": 3
// PASS-DAG: "status": "pass"

module {
  dataflow.graph.func private @carry_address_relu_tail(
      %ctrl: none, %ub: i16, %lb: i16, %step: i16, %zero: i8,
      %mem: memref<?xi8>) -> none {
    %index, %rwc = dataflow.stream %ub, %lb, %step {cont_cond = ">", step_op = "+="} : i16
    %stable_zero = dataflow.invariant %rwc, %zero : i8
    %zero_i16 = dataflow.constant %ctrl {const_value = 0 : i16} : i16
    %one_i16 = dataflow.constant %ctrl {const_value = 1 : i16} : i16
    %stable_one = dataflow.invariant %rwc, %one_i16 : i16
    %idx_carried = dataflow.carry %rwc, %zero_i16, %idx_next : i16
    %idx_next = arith.addi %idx_carried, %stable_one : i16
    %idx = arith.index_cast %idx_carried : i16 to index
    %data, %load_done = dataflow.load %mem[%idx] %ctrl : memref<?xi8>
    %is_negative = arith.cmpi slt, %data, %stable_zero : i8
    %selected = arith.select %is_negative, %stable_zero, %data : i8
    %store_done = dataflow.store %mem[%idx] %selected %ctrl : memref<?xi8>
    %done:2 = dataflow.sync %load_done, %store_done : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }
}

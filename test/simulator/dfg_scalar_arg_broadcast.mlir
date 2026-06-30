// RUN: loom-dfg-sim %s --graph scalar_arg_broadcast_store --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 1=1 --arg 1=2 --memref 2=0,0,0 --arg 3=1 --arg 3=2 --arg 3=3 --arg 4=10 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph scalar_carry_seed_not_broadcast --arg 0=none --arg 0=none --arg 0=none --arg 1=1 --arg 1=1 --arg 1=1 --arg 2=1 --arg 2=2 --arg 2=3 --arg 3=0 --output %t.carry.json
// RUN: FileCheck %s --check-prefix=CARRY < %t.carry.json

// CHECK-DAG: "graph": "scalar_arg_broadcast_store"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 3
// CHECK-DAG: "arith.addi": 3
// CHECK-DAG: "dataflow.store": 3
// CHECK-DAG: "arg2": [
// CHECK-DAG: "i32:11"
// CHECK-DAG: "i32:12"
// CHECK-DAG: "i32:13"

// CARRY-DAG: "graph": "scalar_carry_seed_not_broadcast"
// CARRY-DAG: "status": "pass"
// CARRY-DAG: "dynamic_work_items": 3
// CARRY-DAG: "arith.addi": 3
// CARRY-DAG: "dataflow.carry": 4
// CARRY-DAG: "final_outputs":
// CARRY-DAG: "none",
// CARRY-DAG: "i32:6"

module {
  dataflow.graph.func private @scalar_arg_broadcast_store(
      %ctrl: none, %slot: index, %mem: memref<?xi32>, %value: i32,
      %bias: i32) -> none {
    %sum = arith.addi %value, %bias : i32
    %done = dataflow.store %mem[%slot] %sum %ctrl : memref<?xi32>
    dataflow.graph.return %done : none
  }

  dataflow.graph.func private @scalar_carry_seed_not_broadcast(
      %ctrl: none, %cond: i1, %value: i32, %init: i32) -> (none, i32) {
    %carry = dataflow.carry %cond, %init, %next : i32
    %next = arith.addi %carry, %value : i32
    dataflow.graph.return %ctrl, %carry : none, i32
  }
}

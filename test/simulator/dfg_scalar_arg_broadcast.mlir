// RUN: loom-dfg-sim %s --graph scalar_arg_broadcast_store --arg 0=0 --arg 0=1 --arg 0=2 --arg 1=1 --arg 1=2 --arg 1=3 --arg 2=10 --arg 2=10 --arg 2=10 --arg 3=none --arg 3=none --arg 3=none --arg 4=false --arg 4=false --arg 4=true --memref 5=0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph scalar_carry_seed_not_broadcast --arg 0=0 --arg 1=true --arg 1=true --arg 1=true --arg 1=false --arg 2=1 --arg 2=2 --arg 2=3 --arg 3=none --arg 3=none --arg 3=none --arg 3=none --arg 4=false --arg 4=false --arg 4=false --arg 4=true --output %t.carry.json
// RUN: FileCheck %s --check-prefix=CARRY < %t.carry.json

// CHECK-DAG: "graph": "scalar_arg_broadcast_store"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 3
// CHECK-DAG: "arith.addi": 3
// CHECK-DAG: "dataflow.store": 3
// CHECK-DAG: "arg5": [
// CHECK-DAG: "i32:11"
// CHECK-DAG: "i32:12"
// CHECK-DAG: "i32:13"

// CARRY-DAG: "graph": "scalar_carry_seed_not_broadcast"
// CARRY-DAG: "status": "pass"
// CARRY-DAG: "dynamic_work_items": 4
// CARRY-DAG: "arith.addi": 3
// CARRY-DAG: "dataflow.carry": 5
// CARRY-DAG: "final_outputs":
// CARRY-DAG: "none",
// CARRY-DAG: "i32:6"

module {
  dataflow.graph private @scalar_arg_broadcast_store(
      %ctrl: none, %slot: index, %value: i32, %bias: i32,
      %unit: none, %last: i1, %mem: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 0, 5, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %sum = arith.addi %value, %bias : i32
    %done = dataflow.store %mem[%slot] %sum %unit : memref<?xi32>
    %complete:2 = dataflow.demux %last, %done
        : (i1, none) -> (none, none)
    dataflow.graph.return %complete#1 : none
  }

  dataflow.graph private @scalar_carry_seed_not_broadcast(
      %ctrl: none, %init: i32, %cond: i1, %value: i32,
      %phase_unit: none, %last: i1) -> (i32)
      attributes {input_segments = array<i32: 1, 4, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %carry = dataflow.carry %cond, %init, %next : i32
    %next = arith.addi %carry, %value : i32
    %closed:2 = dataflow.demux %cond, %phase_unit
        : (i1, none) -> (none, none)
    %final:2 = dataflow.demux %last, %carry
        : (i1, i32) -> (i32, i32)
    %published:2 = dataflow.sync %closed#0, %final#1
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

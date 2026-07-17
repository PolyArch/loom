// RUN: loom-dfg-sim %s --graph carry_address_relu_tail --arg 0=3 --arg 1=0 --arg 2=-1 --arg 3=0 --memref 4=-1,2,-3 --output %t.json
// RUN: FileCheck %s --check-prefix=PASS < %t.json

// PASS: "dynamic_work_items": 3
// PASS: "final_memory_state": {
// PASS-NEXT: "arg4": [
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
  dataflow.graph private @carry_address_relu_tail(
      %ctrl: none, %ub: i16, %lb: i16, %step: i16, %zero: i8,
      %mem: memref<?xi8>) -> ()
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %index, %rwc = dataflow.stream %ub, %lb, %step step add while sgt : i16
    %execution = dataflow.carry %rwc, %ctrl, %execution_lane#1 : none
    %execution_lane:2 = dataflow.demux %rwc, %execution
        : (i1, none) -> (none, none)
    %memory_frontier = dataflow.carry %rwc, %ctrl, %store_done : none
    %memory_lane:2 = dataflow.demux %rwc, %memory_frontier
        : (i1, none) -> (none, none)
    %activation:2 = dataflow.sync %execution_lane#1, %memory_lane#1
        : (none, none) -> (none, none)
    %stable_zero = dataflow.invariant %rwc, %zero : i8
    %zero_i16 = dataflow.constant %ctrl {const_value = 0 : i16} : i16
    %one_i16 = dataflow.constant %ctrl {const_value = 1 : i16} : i16
    %stable_one = dataflow.invariant %rwc, %one_i16 : i16
    %idx_carried = dataflow.carry %rwc, %zero_i16, %idx_next : i16
    %idx_next = arith.addi %idx_carried, %stable_one : i16
    %idx = arith.index_cast %idx_carried : i16 to index
    %data, %load_done = dataflow.load %mem[%idx] %activation#0
        : memref<?xi8>
    %is_negative = arith.cmpi slt, %data, %stable_zero : i8
    %selected = arith.select %is_negative, %stable_zero, %data : i8
    %store_done = dataflow.store %mem[%idx] %selected %load_done
        : memref<?xi8>
    %retired:2 = dataflow.sync %execution_lane#0, %memory_lane#0
        : (none, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%retired#0 : none)
  }
}

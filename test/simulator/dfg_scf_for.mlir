// RUN: loom-dfg-sim %s --graph structured_for_sum --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --arg 5=3 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph structured_if_scalar --arg 0=none --arg 1=true --arg 2=7 --arg 3=5 --output %t.if.json
// RUN: FileCheck %s --check-prefix=IF < %t.if.json
// RUN: loom-dfg-sim %s --graph structured_if_waits_for_delayed_capture --arg 0=none --arg 1=true --arg 2=7 --output %t.if_wait.json
// RUN: FileCheck %s --check-prefix=IF-WAIT < %t.if_wait.json
// RUN: loom-dfg-sim %s --graph structured_for_nested_if --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --arg 5=3 --arg 6=true --output %t.nested_if.json
// RUN: FileCheck %s --check-prefix=NESTED-IF < %t.nested_if.json
// RUN: loom-dfg-sim %s --graph structured_for_scalar_with_parallel_stream --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --arg 5=3 --output %t.parallel.json
// RUN: FileCheck %s --check-prefix=PARALLEL < %t.parallel.json
// RUN: loom-dfg-sim %s --graph structured_for_captures_top_level_constant --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --output %t.capture.json
// RUN: FileCheck %s --check-prefix=CAPTURE < %t.capture.json
// RUN: loom-dfg-sim %s --graph structured_for_captures_dynamic_arg --arg 0=none --arg 0=none --arg 1=0 --arg 1=1 --arg 2=0 --arg 2=0 --arg 3=1 --arg 3=1 --arg 4=1 --arg 4=1 --arg 5=10 --arg 5=20 --arg 6=3 --arg 6=5 --memref 7=0,0 --output %t.dynamic_capture.json
// RUN: FileCheck %s --check-prefix=DYNAMIC-CAPTURE < %t.dynamic_capture.json
// RUN: loom-dfg-sim %s --graph structured_for_batched_return --arg 0=none --arg 0=none --arg 1=0 --arg 1=0 --arg 2=4 --arg 2=4 --arg 3=1 --arg 3=1 --arg 4=10 --arg 4=20 --output %t.batched_return.json
// RUN: FileCheck %s --check-prefix=BATCHED-RETURN < %t.batched_return.json
// RUN: loom-dfg-sim %s --graph structured_for_rejects_memref_operand_cast --arg 0=none --arg 1=0 --arg 2=1 --arg 3=1 --memref 4=0 --arg 5=7 --output %t.memref_cast.json
// RUN: FileCheck %s --check-prefix=MEMREF-CAST < %t.memref_cast.json
// RUN: loom-dfg-sim %s --graph structured_for_blocks_partial_dynamic_capture --arg 0=none --arg 0=none --arg 1=0 --arg 1=1 --arg 2=0 --arg 2=0 --arg 3=1 --arg 3=1 --arg 4=1 --arg 4=1 --arg 5=10 --arg 5=20 --arg 6=3 --arg 6=5 --arg 7=0 --memref 8=0,0 --output %t.partial_capture.json
// RUN: FileCheck %s --check-prefix=PARTIAL-CAPTURE < %t.partial_capture.json
// RUN: loom-dfg-sim %s --graph structured_for_pointer_memory --arg 0=none --memref 1=1,2,3 --arg 2=0 --arg 3=3 --arg 4=1 --arg 5=0 --output %t.pointer_memory.json
// RUN: FileCheck %s --check-prefix=POINTER-MEMORY < %t.pointer_memory.json
// RUN: loom-dfg-sim %s --graph structured_for_carried_pointer_memory --arg 0=none --memref 1=1 --arg 2=0 --arg 3=1 --arg 4=1 --arg 5=0 --output %t.carried_pointer_memory.json
// RUN: FileCheck %s --check-prefix=CARRIED-POINTER-MEMORY < %t.carried_pointer_memory.json

// CHECK-DAG: "graph": "structured_for_sum"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "arith.addi": 4
// CHECK-DAG: "final_outputs": [
// CHECK-DAG: "none"
// CHECK-DAG: "i32:22"

// IF-DAG: "graph": "structured_if_scalar"
// IF-DAG: "status": "pass"
// IF-DAG: "scf.if": 1
// IF-DAG: "arith.addi": 1
// IF-DAG: "final_outputs": [
// IF-DAG: "none"
// IF-DAG: "i32:12"

// IF-WAIT-DAG: "graph": "structured_if_waits_for_delayed_capture"
// IF-WAIT-DAG: "status": "pass"
// IF-WAIT-DAG: "dataflow.constant": 1
// IF-WAIT-DAG: "arith.addi": 1
// IF-WAIT-DAG: "scf.if": 1
// IF-WAIT-DAG: "final_outputs": [
// IF-WAIT-DAG: "none"
// IF-WAIT-DAG: "i32:8"

// NESTED-IF-DAG: "graph": "structured_for_nested_if"
// NESTED-IF-DAG: "status": "pass"
// NESTED-IF-DAG: "dynamic_work_items": 4
// NESTED-IF-DAG: "scf.if": 4
// NESTED-IF-DAG: "arith.addi": 4
// NESTED-IF-DAG: "final_outputs": [
// NESTED-IF-DAG: "none"
// NESTED-IF-DAG: "i32:22"

// PARALLEL-DAG: "graph": "structured_for_scalar_with_parallel_stream"
// PARALLEL-DAG: "status": "blocked"
// PARALLEL-DAG: "dynamic_work_items": 4
// PARALLEL-DAG: "dataflow.graph.return value produced 1 of 4 dynamic work items"

// CAPTURE-DAG: "graph": "structured_for_captures_top_level_constant"
// CAPTURE-DAG: "status": "pass"
// CAPTURE-DAG: "dynamic_work_items": 4
// CAPTURE-DAG: "i32:18"
// CAPTURE-NOT: "structured scf.for operand is unavailable"

// DYNAMIC-CAPTURE-DAG: "graph": "structured_for_captures_dynamic_arg"
// DYNAMIC-CAPTURE-DAG: "status": "pass"
// DYNAMIC-CAPTURE-DAG: "arg7": [
// DYNAMIC-CAPTURE-DAG: "i32:13"
// DYNAMIC-CAPTURE-DAG: "i32:25"
// DYNAMIC-CAPTURE-NOT: "i32:15"

// BATCHED-RETURN-DAG: "graph": "structured_for_batched_return"
// BATCHED-RETURN-DAG: "status": "pass"
// BATCHED-RETURN-DAG: "dynamic_work_items": 4
// BATCHED-RETURN-DAG: "i32:20"
// BATCHED-RETURN-DAG: "i32:28"
// BATCHED-RETURN-NOT: "dataflow.graph.return value produced 2 of 4 dynamic work items"

// MEMREF-CAST-DAG: "graph": "structured_for_rejects_memref_operand_cast"
// MEMREF-CAST-DAG: "status": "unsupported"
// MEMREF-CAST-DAG: "unsupported op: builtin.unrealized_conversion_cast"

// PARTIAL-CAPTURE-DAG: "graph": "structured_for_blocks_partial_dynamic_capture"
// PARTIAL-CAPTURE-DAG: "status": "blocked"
// PARTIAL-CAPTURE-DAG: "dataflow.graph.return value produced 1 of 2 dynamic work items"
// PARTIAL-CAPTURE-DAG: "arg8": [
// PARTIAL-CAPTURE-DAG: "i32:13"
// PARTIAL-CAPTURE-DAG: "i32:0"
// PARTIAL-CAPTURE-NOT: "i32:23"

// POINTER-MEMORY-DAG: "graph": "structured_for_pointer_memory"
// POINTER-MEMORY-DAG: "status": "pass"
// POINTER-MEMORY-DAG: "dynamic_work_items": 3
// POINTER-MEMORY-DAG: "llvm.getelementptr": 3
// POINTER-MEMORY-DAG: "dataflow.load": 3
// POINTER-MEMORY-DAG: "dataflow.store": 3
// POINTER-MEMORY-DAG: "final_outputs": [
// POINTER-MEMORY-DAG: "none"
// POINTER-MEMORY-DAG: "i32:6"
// POINTER-MEMORY-DAG: "arg1": [
// POINTER-MEMORY-DAG: "i32:2"
// POINTER-MEMORY-DAG: "i32:3"
// POINTER-MEMORY-DAG: "i32:4"

// CARRIED-POINTER-MEMORY-DAG: "graph": "structured_for_carried_pointer_memory"
// CARRIED-POINTER-MEMORY-DAG: "status": "pass"
// CARRIED-POINTER-MEMORY-DAG: "dynamic_work_items": 1
// CARRIED-POINTER-MEMORY-DAG: "dataflow.load": 1
// CARRIED-POINTER-MEMORY-DAG: "dataflow.store": 1
// CARRIED-POINTER-MEMORY-DAG: "final_outputs": [
// CARRIED-POINTER-MEMORY-DAG: "none"
// CARRIED-POINTER-MEMORY-DAG: "i32:1"
// CARRIED-POINTER-MEMORY-DAG: "arg1": [
// CARRIED-POINTER-MEMORY-DAG: "i32:2"
// CARRIED-POINTER-MEMORY-NOT: "i32:3"

module {
  dataflow.graph.func private @structured_if_scalar(
      %ctrl: none, %cond: i1, %lhs: i32, %rhs: i32) -> (none, i32) {
    %value = scf.if %cond -> (i32) {
      %sum = arith.addi %lhs, %rhs : i32
      scf.yield %sum : i32
    } else {
      %diff = arith.subi %lhs, %rhs : i32
      scf.yield %diff : i32
    }
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @structured_if_waits_for_delayed_capture(
      %ctrl: none, %cond: i1, %lhs: i32) -> (none, i32) {
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %sum = arith.addi %lhs, %one : i32
    %value = scf.if %cond -> (i32) {
      scf.yield %sum : i32
    } else {
      scf.yield %lhs : i32
    }
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @structured_for_sum(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %init: i32, %addend: i32)
      -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_nested_if(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %init: i32, %addend: i32,
      %cond: i1) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = scf.if %cond -> (i32) {
        %added = arith.addi %acc, %addend : i32
        scf.yield %added : i32
      } else {
        scf.yield %acc : i32
      }
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_scalar_with_parallel_stream(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %init: i32, %addend: i32)
      -> (none, i32) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i64
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_captures_top_level_constant(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %init: i32)
      -> (none, i32) {
    %addend = arith.constant 2 : i32
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_captures_dynamic_arg(
      %ctrl: none, %slot: index, %lb: i64, %ub: i64, %step: i64,
      %init: i32, %addend: i32, %mem: memref<?xi32>) -> none {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    %done = dataflow.store %mem[%slot] %sum %ctrl : memref<?xi32>
    dataflow.graph.return %done : none
  }

  dataflow.graph.func private @structured_for_batched_return(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %init: i32)
      -> (none, i32, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %two = arith.constant 2 : i32
      %next = arith.addi %acc, %two : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %init, %sum : none, i32, i32
  }

  dataflow.graph.func private @structured_for_rejects_memref_operand_cast(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %mem: memref<?xi32>,
      %init: i32) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %ptr = builtin.unrealized_conversion_cast %mem : memref<?xi32> to !llvm.ptr
      %next = arith.addi %acc, %acc : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_blocks_partial_dynamic_capture(
      %ctrl: none, %slot: index, %lb: i64, %ub: i64, %step: i64,
      %init: i32, %lhs: i32, %rhs: i32, %mem: memref<?xi32>) -> none {
    %addend = arith.addi %lhs, %rhs : i32
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    %done = dataflow.store %mem[%slot] %sum %ctrl : memref<?xi32>
    dataflow.graph.return %done : none
  }

  dataflow.graph.func private @structured_for_pointer_memory(
      %ctrl: none, %mem: !llvm.ptr, %lb: i64, %ub: i64, %step: i64,
      %init: i32) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %ptr = llvm.getelementptr inbounds|nuw %mem[%i]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %view = builtin.unrealized_conversion_cast %ptr
          : !llvm.ptr to memref<?xi32>
      %slot = dataflow.constant %ctrl {const_value = 0 : index} : index
      %value, %load_done = dataflow.load %view[%slot] %ctrl : memref<?xi32>
      %one = arith.constant 1 : i32
      %stored = arith.addi %value, %one : i32
      %store_done = dataflow.store %view[%slot] %stored %ctrl : memref<?xi32>
      %next = arith.addi %acc, %value : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_carried_pointer_memory(
      %ctrl: none, %mem: !llvm.ptr, %lb: i64, %ub: i64, %step: i64,
      %init: i32) -> (none, i32) {
    %view = builtin.unrealized_conversion_cast %mem
        : !llvm.ptr to memref<?xi32>
    %sum, %view_out = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %init, %carried = %view)
        -> (i32, memref<?xi32>) : i64 {
      %slot = dataflow.constant %ctrl {const_value = 0 : index} : index
      %value, %load_done = dataflow.load %carried[%slot] %ctrl
          : memref<?xi32>
      %one = arith.constant 1 : i32
      %stored = arith.addi %value, %one : i32
      %store_done = dataflow.store %carried[%slot] %stored %ctrl
          : memref<?xi32>
      %next = arith.addi %acc, %value : i32
      scf.yield %next, %carried : i32, memref<?xi32>
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }
}

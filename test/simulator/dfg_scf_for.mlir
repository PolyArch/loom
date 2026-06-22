// RUN: loom-dfg-sim %s --graph structured_for_sum --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --arg 5=3 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph structured_for_nested_if --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=10 --arg 5=3 --arg 6=true --output %t.unsupported.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED < %t.unsupported.json
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

// CHECK-DAG: "graph": "structured_for_sum"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "arith.addi": 4
// CHECK-DAG: "final_outputs": [
// CHECK-DAG: "none"
// CHECK-DAG: "i32:22"

// UNSUPPORTED-DAG: "graph": "structured_for_nested_if"
// UNSUPPORTED-DAG: "status": "unsupported"
// UNSUPPORTED-DAG: "unsupported op: scf.if"

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

module {
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
}

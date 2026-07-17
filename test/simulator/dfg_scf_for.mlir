// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_sum --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=10 --arg 4=3 --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_if_scalar --arg 0=true --arg 1=7 --arg 2=5 --output %t.if.json
// RUN: FileCheck %s --check-prefix=IF < %t.if.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_if_waits_for_delayed_capture --arg 0=true --arg 1=7 --output %t.if_wait.json
// RUN: FileCheck %s --check-prefix=IF-WAIT < %t.if_wait.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_if_effect_batched --arg 0=true --arg 1=0 --arg 2=10 --memref 3=0,0 --output %t.if_effect_first.json
// RUN: FileCheck %s --check-prefix=IF-EFFECT-FIRST < %t.if_effect_first.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_if_effect_batched --arg 0=true --arg 1=1 --arg 2=20 --memref 3=10,0 --output %t.if_effect_second.json
// RUN: FileCheck %s --check-prefix=IF-EFFECT-SECOND < %t.if_effect_second.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_nested_if --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=10 --arg 4=3 --arg 5=true --output %t.nested_if.json
// RUN: FileCheck %s --check-prefix=NESTED-IF < %t.nested_if.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_scalar_with_parallel_stream --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=10 --arg 4=3 --output %t.parallel.json
// RUN: FileCheck %s --check-prefix=PARALLEL < %t.parallel.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_captures_top_level_constant --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=10 --output %t.capture.json
// RUN: FileCheck %s --check-prefix=CAPTURE < %t.capture.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_captures_dynamic_arg --arg 0=0 --arg 1=0 --arg 2=1 --arg 3=1 --arg 4=10 --arg 5=3 --memref 6=0,0 --output %t.dynamic_capture_first.json
// RUN: FileCheck %s --check-prefix=DYNAMIC-CAPTURE-FIRST < %t.dynamic_capture_first.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_captures_dynamic_arg --arg 0=1 --arg 1=0 --arg 2=1 --arg 3=1 --arg 4=20 --arg 5=5 --memref 6=13,0 --output %t.dynamic_capture_second.json
// RUN: FileCheck %s --check-prefix=DYNAMIC-CAPTURE-SECOND < %t.dynamic_capture_second.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_batched_return --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=10 --output %t.return_first.json
// RUN: FileCheck %s --check-prefix=RETURN-FIRST < %t.return_first.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_batched_return --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=20 --output %t.return_second.json
// RUN: FileCheck %s --check-prefix=RETURN-SECOND < %t.return_second.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_memref_capture --arg 0=0 --arg 1=1 --arg 2=1 --arg 3=7 --memref 4=0 --output %t.memref_capture.json
// RUN: FileCheck %s --check-prefix=MEMREF-CAPTURE < %t.memref_capture.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_blocks_partial_dynamic_capture --arg 0=0 --arg 1=0 --arg 2=1 --arg 3=1 --arg 4=10 --arg 5=3 --arg 6=0 --memref 7=0,0 --output %t.partial_capture_first.json
// RUN: FileCheck %s --check-prefix=PARTIAL-CAPTURE-FIRST < %t.partial_capture_first.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_blocks_partial_dynamic_capture --arg 0=1 --arg 1=0 --arg 2=1 --arg 3=1 --arg 4=20 --arg 5=5 --arg 6=0 --memref 7=13,0 --output %t.partial_capture_second.json
// RUN: FileCheck %s --check-prefix=PARTIAL-CAPTURE-SECOND < %t.partial_capture_second.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_pointer_memory --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0 --memref 4=1,2,3 --output %t.pointer_memory.json
// RUN: FileCheck %s --check-prefix=POINTER-MEMORY < %t.pointer_memory.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_pointer_memory --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0 --memref 4=1 --output %t.pointer_memory_oob.json
// RUN: FileCheck %s --check-prefix=POINTER-MEMORY-OOB < %t.pointer_memory_oob.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_pointer_select_memory --arg 0=false --arg 1=0 --memref 2=11 --memref 3=22 --output %t.pointer_select_memory.json
// RUN: FileCheck %s --check-prefix=POINTER-SELECT-MEMORY < %t.pointer_select_memory.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_if_nested_for_pointer_memory --arg 0=true --arg 1=0 --arg 2=3 --arg 3=1 --arg 4=0.000000e+00 --memref 5=1.000000e+00,2.000000e+00,3.000000e+00 --output %t.if_nested_for_memory.json
// RUN: FileCheck %s --check-prefix=IF-NESTED-FOR-MEMORY < %t.if_nested_for_memory.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_for_autocorr_slice --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0 --arg 4=0.000000e+00 --arg 5=2 --arg 6=3 --arg 7=0 --arg 8=0 --memref 9=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00 --memref 10=0.000000e+00,0.000000e+00,0.000000e+00 --output %t.autocorr_slice.json
// RUN: FileCheck %s --check-prefix=AUTOCORR-SLICE < %t.autocorr_slice.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_forall_store --arg 0=0 --arg 1=3 --arg 2=10 --memref 3=1,2,3 --output %t.forall_store.json
// RUN: FileCheck %s --check-prefix=FORALL-STORE < %t.forall_store.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph structured_forall_pointer_capture_memory --arg 0=1 --arg 1=1 --memref 2=10,20,30,40,50 --memref 3=0,0,0,0,0 --output %t.forall_pointer_capture.json
// RUN: FileCheck %s --check-prefix=FORALL-POINTER-CAPTURE < %t.forall_pointer_capture.json

// CHECK-DAG: "graph": "structured_for_sum"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "arith.addi": 4
// CHECK-DAG: "final_outputs": [
// CHECK-DAG: "none"
// CHECK-DAG: "i32:22"

// IF-DAG: "graph": "structured_if_scalar"
// IF-DAG: "status": "pass"
// IF-DAG: "dataflow.demux": 3
// IF-DAG: "dataflow.mux": 2
// IF-DAG: "arith.addi": 1
// IF-DAG: "final_outputs": [
// IF-DAG: "none"
// IF-DAG: "i32:12"

// IF-WAIT-DAG: "graph": "structured_if_waits_for_delayed_capture"
// IF-WAIT-DAG: "status": "pass"
// IF-WAIT-DAG: "dataflow.constant": 1
// IF-WAIT-DAG: "arith.addi": 1
// IF-WAIT-DAG: "dataflow.demux": 3
// IF-WAIT-DAG: "dataflow.mux": 2
// IF-WAIT-DAG: "final_outputs": [
// IF-WAIT-DAG: "none"
// IF-WAIT-DAG: "i32:8"

// IF-EFFECT-FIRST-DAG: "graph": "structured_if_effect_batched"
// IF-EFFECT-FIRST-DAG: "status": "pass"
// IF-EFFECT-FIRST-DAG: "dataflow.demux": 5
// IF-EFFECT-FIRST-DAG: "dataflow.mux": 2
// IF-EFFECT-FIRST-DAG: "dataflow.store": 1
// IF-EFFECT-FIRST-DAG: "arg3": [
// IF-EFFECT-FIRST-DAG: "i32:10"
// IF-EFFECT-FIRST-DAG: "i32:0"

// IF-EFFECT-SECOND-DAG: "graph": "structured_if_effect_batched"
// IF-EFFECT-SECOND-DAG: "status": "pass"
// IF-EFFECT-SECOND-DAG: "dataflow.demux": 5
// IF-EFFECT-SECOND-DAG: "dataflow.mux": 2
// IF-EFFECT-SECOND-DAG: "dataflow.store": 1
// IF-EFFECT-SECOND-DAG: "arg3": [
// IF-EFFECT-SECOND-DAG: "i32:10"
// IF-EFFECT-SECOND-DAG: "i32:20"

// NESTED-IF-DAG: "graph": "structured_for_nested_if"
// NESTED-IF-DAG: "status": "pass"
// NESTED-IF-DAG: "dynamic_work_items": 4
// NESTED-IF-DAG: "dataflow.demux": 22
// NESTED-IF-DAG: "dataflow.mux": 8
// NESTED-IF-DAG: "arith.addi": 4
// NESTED-IF-DAG: "final_outputs": [
// NESTED-IF-DAG: "none"
// NESTED-IF-DAG: "i32:22"

// PARALLEL-DAG: "graph": "structured_for_scalar_with_parallel_stream"
// PARALLEL-DAG: "status": "pass"
// PARALLEL-DAG: "dynamic_work_items": 4
// PARALLEL-DAG: "i32:22"
// PARALLEL-NOT: "dataflow.graph.return value produced"

// CAPTURE-DAG: "graph": "structured_for_captures_top_level_constant"
// CAPTURE-DAG: "status": "pass"
// CAPTURE-DAG: "dynamic_work_items": 4
// CAPTURE-DAG: "i32:18"
// CAPTURE-NOT: "structured scf.for operand is unavailable"

// DYNAMIC-CAPTURE-FIRST-DAG: "graph": "structured_for_captures_dynamic_arg"
// DYNAMIC-CAPTURE-FIRST-DAG: "status": "pass"
// DYNAMIC-CAPTURE-FIRST-DAG: "arith.addi": 1
// DYNAMIC-CAPTURE-FIRST-DAG: "dataflow.store": 1
// DYNAMIC-CAPTURE-FIRST-DAG: "arg6": [
// DYNAMIC-CAPTURE-FIRST-DAG: "i32:13"
// DYNAMIC-CAPTURE-FIRST-DAG: "i32:0"

// DYNAMIC-CAPTURE-SECOND-DAG: "graph": "structured_for_captures_dynamic_arg"
// DYNAMIC-CAPTURE-SECOND-DAG: "status": "pass"
// DYNAMIC-CAPTURE-SECOND-DAG: "arith.addi": 1
// DYNAMIC-CAPTURE-SECOND-DAG: "dataflow.store": 1
// DYNAMIC-CAPTURE-SECOND-DAG: "arg6": [
// DYNAMIC-CAPTURE-SECOND-DAG: "i32:13"
// DYNAMIC-CAPTURE-SECOND-DAG: "i32:25"
// DYNAMIC-CAPTURE-SECOND-NOT: "i32:15"

// RETURN-FIRST-DAG: "graph": "structured_for_batched_return"
// RETURN-FIRST-DAG: "status": "pass"
// RETURN-FIRST-DAG: "dynamic_work_items": 4
// RETURN-FIRST-DAG: "i32:10"
// RETURN-FIRST-DAG: "i32:18"

// RETURN-SECOND-DAG: "graph": "structured_for_batched_return"
// RETURN-SECOND-DAG: "status": "pass"
// RETURN-SECOND-DAG: "dynamic_work_items": 4
// RETURN-SECOND-DAG: "i32:20"
// RETURN-SECOND-DAG: "i32:28"

// MEMREF-CAPTURE-DAG: "graph": "structured_for_memref_capture"
// MEMREF-CAPTURE-DAG: "status": "pass"
// MEMREF-CAPTURE-DAG: "arith.addi": 1
// MEMREF-CAPTURE-DAG: "dataflow.carry": 6
// MEMREF-CAPTURE-DAG: "dataflow.demux": 4
// MEMREF-CAPTURE-DAG: "i32:14"

// PARTIAL-CAPTURE-FIRST-DAG: "graph": "structured_for_blocks_partial_dynamic_capture"
// PARTIAL-CAPTURE-FIRST-DAG: "status": "pass"
// PARTIAL-CAPTURE-FIRST-DAG: "arith.addi": 2
// PARTIAL-CAPTURE-FIRST-DAG: "dataflow.store": 1
// PARTIAL-CAPTURE-FIRST-DAG: "arg7": [
// PARTIAL-CAPTURE-FIRST-DAG: "i32:13"
// PARTIAL-CAPTURE-FIRST-DAG: "i32:0"

// PARTIAL-CAPTURE-SECOND-DAG: "graph": "structured_for_blocks_partial_dynamic_capture"
// PARTIAL-CAPTURE-SECOND-DAG: "status": "pass"
// PARTIAL-CAPTURE-SECOND-DAG: "arith.addi": 2
// PARTIAL-CAPTURE-SECOND-DAG: "dataflow.store": 1
// PARTIAL-CAPTURE-SECOND-DAG: "arg7": [
// PARTIAL-CAPTURE-SECOND-DAG: "i32:13"
// PARTIAL-CAPTURE-SECOND-DAG: "i32:25"
// PARTIAL-CAPTURE-SECOND-NOT: "dataflow.graph.return value produced"

// POINTER-MEMORY-DAG: "graph": "structured_for_pointer_memory"
// POINTER-MEMORY-DAG: "status": "pass"
// POINTER-MEMORY-DAG: "dynamic_work_items": 3
// POINTER-MEMORY-DAG: "arith.index_cast": 3
// POINTER-MEMORY-DAG: "dataflow.load": 3
// POINTER-MEMORY-DAG: "dataflow.store": 3
// POINTER-MEMORY-DAG: "final_outputs": [
// POINTER-MEMORY-DAG: "none"
// POINTER-MEMORY-DAG: "i32:6"
// POINTER-MEMORY-DAG: "arg4": [
// POINTER-MEMORY-DAG: "i32:2"
// POINTER-MEMORY-DAG: "i32:3"
// POINTER-MEMORY-DAG: "i32:4"

// POINTER-MEMORY-OOB-DAG: "graph": "structured_for_pointer_memory"
// POINTER-MEMORY-OOB-DAG: "status": "blocked"
// POINTER-MEMORY-OOB-DAG: "graph did not fire its retirement frontier"
// POINTER-MEMORY-OOB-DAG: "dataflow.load address is out of range"
// POINTER-MEMORY-OOB-DAG: "dataflow.load consumed 1 of 3 true stream indices"

// POINTER-SELECT-MEMORY-DAG: "graph": "structured_for_pointer_select_memory"
// POINTER-SELECT-MEMORY-DAG: "status": "pass"
// POINTER-SELECT-MEMORY-DAG: "llvm.select": 1
// POINTER-SELECT-MEMORY-DAG: "dataflow.load": 1
// POINTER-SELECT-MEMORY-DAG: "final_outputs": [
// POINTER-SELECT-MEMORY-DAG: "none"
// POINTER-SELECT-MEMORY-DAG: "i32:22"

// IF-NESTED-FOR-MEMORY-DAG: "graph": "structured_if_nested_for_pointer_memory"
// IF-NESTED-FOR-MEMORY-DAG: "status": "pass"
// IF-NESTED-FOR-MEMORY-DAG: "dynamic_work_items": 3
// IF-NESTED-FOR-MEMORY-DAG: "dataflow.demux": 23
// IF-NESTED-FOR-MEMORY-DAG: "dataflow.mux": 3
// IF-NESTED-FOR-MEMORY-DAG: "dataflow.load": 3
// IF-NESTED-FOR-MEMORY-DAG: "final_outputs": [
// IF-NESTED-FOR-MEMORY-DAG: "none"
// IF-NESTED-FOR-MEMORY-DAG: "f32:6"

// AUTOCORR-SLICE-DAG: "graph": "structured_for_autocorr_slice"
// AUTOCORR-SLICE-DAG: "status": "pass"
// AUTOCORR-SLICE-DAG: "dynamic_work_items": 4
// AUTOCORR-SLICE-DAG: "dataflow.demux": 70
// AUTOCORR-SLICE-DAG: "dataflow.mux": 12
// AUTOCORR-SLICE-DAG: "llvm.intr.umax": 2
// AUTOCORR-SLICE-DAG: "llvm.intr.fmuladd": 4
// AUTOCORR-SLICE-DAG: "dataflow.load": 8
// AUTOCORR-SLICE-DAG: "dataflow.store": 3
// AUTOCORR-SLICE-DAG: "final_outputs": [
// AUTOCORR-SLICE-DAG: "none"
// AUTOCORR-SLICE-DAG: "i32:0"
// AUTOCORR-SLICE-DAG: "arg10": [
// AUTOCORR-SLICE-DAG: "f32:0"
// AUTOCORR-SLICE-DAG: "f32:8"
// AUTOCORR-SLICE-DAG: "f32:11"

// FORALL-STORE-DAG: "graph": "structured_forall_store"
// FORALL-STORE-DAG: "status": "pass"
// FORALL-STORE-DAG: "dynamic_work_items": 3
// FORALL-STORE-DAG: "dataflow.stream": 4
// FORALL-STORE-DAG: "dataflow.carry": 15
// FORALL-STORE-DAG: "dataflow.demux": 12
// FORALL-STORE-DAG: "dataflow.load": 3
// FORALL-STORE-DAG: "dataflow.store": 3
// FORALL-STORE-DAG: "arith.addi": 3
// FORALL-STORE-DAG: "final_outputs": [
// FORALL-STORE-DAG: "none"
// FORALL-STORE-DAG: "arg3": [
// FORALL-STORE-DAG: "i32:11"
// FORALL-STORE-DAG: "i32:12"
// FORALL-STORE-DAG: "i32:13"

// FORALL-POINTER-CAPTURE-DAG: "graph": "structured_forall_pointer_capture_memory"
// FORALL-POINTER-CAPTURE-DAG: "status": "pass"
// FORALL-POINTER-CAPTURE-DAG: "dynamic_work_items": 3
// FORALL-POINTER-CAPTURE-DAG: "dataflow.stream": 4
// FORALL-POINTER-CAPTURE-DAG: "dataflow.carry": 15
// FORALL-POINTER-CAPTURE-DAG: "dataflow.demux": 12
// FORALL-POINTER-CAPTURE-DAG: "dataflow.load": 3
// FORALL-POINTER-CAPTURE-DAG: "dataflow.store": 3
// FORALL-POINTER-CAPTURE-DAG: "arg3": [
// FORALL-POINTER-CAPTURE-DAG: "i32:0"
// FORALL-POINTER-CAPTURE-DAG: "i32:20"
// FORALL-POINTER-CAPTURE-DAG: "i32:30"
// FORALL-POINTER-CAPTURE-DAG: "i32:40"
// FORALL-POINTER-CAPTURE-DAG: "i32:0"

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

  dataflow.graph.func private @structured_if_effect_batched(
      %ctrl: none, %cond: i1, %slot: index, %mem: memref<?xi32>,
      %value: i32) -> none {
    scf.if %cond {
      %stored = dataflow.store %mem[%slot] %value %ctrl : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
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
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i64
    %parallel_tokens = dataflow.invariant %rwc, %ctrl : none
    %parallel_close:2 = dataflow.demux %rwc, %parallel_tokens
        : (i1, none) -> (none, none)
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %parallel_close#0, %sum : none, i32
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

  dataflow.graph.func private @structured_for_memref_capture(
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
    %view = builtin.unrealized_conversion_cast %mem
        : !llvm.ptr to memref<?xi32>
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) : i64 {
      %slot = arith.index_cast %i : i64 to index
      %value, %load_done = dataflow.load %view[%slot] %ctrl : memref<?xi32>
      %one = arith.constant 1 : i32
      %stored = arith.addi %value, %one : i32
      %store_done = dataflow.store %view[%slot] %stored %ctrl : memref<?xi32>
      %next = arith.addi %acc, %value : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_pointer_select_memory(
      %ctrl: none, %cond: i1, %lhs: !llvm.ptr, %rhs: !llvm.ptr,
      %slot: index) -> (none, i32) {
    %selected = llvm.select %cond, %lhs, %rhs : i1, !llvm.ptr
    %view = builtin.unrealized_conversion_cast %selected
        : !llvm.ptr to memref<?xi32>
    %value, %done = dataflow.load %view[%slot] %ctrl : memref<?xi32>
    dataflow.graph.return %done, %value : none, i32
  }

  dataflow.graph.func private @structured_if_nested_for_pointer_memory(
      %ctrl: none, %cond: i1, %lb: i64, %ub: i64, %step: i64,
      %mem: !llvm.ptr, %init: f32) -> (none, f32) {
    %sum = scf.if %cond -> (f32) {
      %inner = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
          -> (f32) : i64 {
        %view = builtin.unrealized_conversion_cast %mem
            : !llvm.ptr to memref<?xf32>
        %slot = arith.index_cast %i : i64 to index
        %value, %load_done = dataflow.load %view[%slot] %ctrl
            : memref<?xf32>
        %next = arith.addf %acc, %value : f32
        scf.yield %next : f32
      }
      scf.yield %inner : f32
    } else {
      scf.yield %init : f32
    }
    dataflow.graph.return %ctrl, %sum : none, f32
  }

  dataflow.graph.func private @structured_for_autocorr_slice(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %skip_lag: i64,
      %zero: f32, %min_bound: i32, %input: !llvm.ptr, %mask: i64,
      %output: !llvm.ptr, %dec: i32, %remaining_init: i32) -> (none, i32) {
    %remaining = scf.for %lag = %lb to %ub step %step
        iter_args(%remaining_arg = %remaining_init) -> (i32) : i64 {
      %skip = arith.cmpi eq, %lag, %skip_lag : i64
      %sum = scf.if %skip -> (f32) {
        scf.yield %zero : f32
      } else {
        %inner_bound_i32 = llvm.intr.umax(%remaining_arg, %min_bound)
            : (i32, i32) -> i32
        %inner_bound = llvm.zext %inner_bound_i32 : i32 to i64
        %inner = scf.for %i = %lb to %inner_bound step %step
            iter_args(%acc = %zero) -> (f32) : i64 {
          %lhs_view = builtin.unrealized_conversion_cast %input
              : !llvm.ptr to memref<?xf32>
          %lhs_slot = arith.index_cast %i : i64 to index
          %lhs_value, %lhs_done = dataflow.load %lhs_view[%lhs_slot] %ctrl
              : memref<?xf32>
          %rhs_view = builtin.unrealized_conversion_cast %input
              : !llvm.ptr to memref<?xf32>
          %rhs_base = arith.addi %i, %lag : i64
          %rhs_masked = arith.andi %rhs_base, %mask : i64
          %rhs_slot = arith.index_cast %rhs_masked : i64 to index
          %rhs_value, %rhs_done = dataflow.load %rhs_view[%rhs_slot] %ctrl
              : memref<?xf32>
          %next = llvm.intr.fmuladd(%lhs_value, %rhs_value, %acc)
              : (f32, f32, f32) -> f32
          scf.yield %next : f32
        }
        scf.yield %inner : f32
      }
      %out_view = builtin.unrealized_conversion_cast %output
          : !llvm.ptr to memref<?xf32>
      %out_slot = arith.index_cast %lag : i64 to index
      %stored = dataflow.store %out_view[%out_slot] %sum %ctrl
          : memref<?xf32>
      %next_remaining = arith.addi %remaining_arg, %dec : i32
      scf.yield %next_remaining : i32
    }
    dataflow.graph.return %ctrl, %remaining : none, i32
  }

  dataflow.graph.func private @structured_forall_store(
      %ctrl: none, %lb: index, %ub: index, %mem: memref<?xi32>, %addend: i32)
      -> none {
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    scf.for %i = %lb to %ub step %one {
      %value, %done = dataflow.load %mem[%i] %ctrl : memref<?xi32>
      %stored = arith.addi %value, %addend : i32
      %store_done = dataflow.store %mem[%i] %stored %ctrl : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @structured_forall_pointer_capture_memory(
      %ctrl: none, %stride: i64, %src: !llvm.ptr, %dst: !llvm.ptr,
      %row: index) -> none {
    %row64 = arith.index_cast %row : index to i64
    %base = arith.muli %row64, %stride : i64
    %base_index = arith.index_cast %base : i64 to index
    %src_view = builtin.unrealized_conversion_cast %src
        : !llvm.ptr to memref<?xi32>
    %dst_view = builtin.unrealized_conversion_cast %dst
        : !llvm.ptr to memref<?xi32>
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %three = dataflow.constant %ctrl {const_value = 3 : index} : index
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    scf.for %i = %zero to %three step %one {
      %slot = arith.addi %base_index, %i : index
      %value, %load_done = dataflow.load %src_view[%slot] %ctrl
          : memref<?xi32>
      %store_done = dataflow.store %dst_view[%slot] %value %ctrl
          : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }
}

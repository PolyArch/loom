// RUN: loom-dfg-sim %s --graph stream_zero_trip --arg 0=none --arg 1=5 --arg 2=5 --arg 3=1 --output %t.zero.json
// RUN: FileCheck %s --check-prefix=ZERO < %t.zero.json
// RUN: loom-dfg-sim %s --graph stream_finite --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --output %t.finite.json
// RUN: FileCheck %s --check-prefix=FINITE < %t.finite.json
// RUN: loom-dfg-sim %s --graph stream_repeated --arg 0=none --arg 1=0 --arg 1=10 --arg 2=2 --arg 2=13 --arg 3=1 --arg 3=1 --output %t.repeated.json
// RUN: FileCheck %s --check-prefix=REPEATED < %t.repeated.json
// RUN: loom-dfg-sim %s --graph stream_i8_signed_cont --arg 0=none --arg 1=255 --arg 2=1 --arg 3=1 --output %t.i8-cont.json
// RUN: FileCheck %s --check-prefix=I8-CONT < %t.i8-cont.json
// RUN: loom-dfg-sim %s --graph stream_i8_unsigned_cont --arg 0=none --arg 1=255 --arg 2=1 --arg 3=1 --output %t.i8-unsigned.json
// RUN: FileCheck %s --check-prefix=I8-UNSIGNED < %t.i8-unsigned.json
// RUN: loom-dfg-sim %s --graph stream_i8_signed_index_cast --arg 0=none --arg 1=255 --arg 2=0 --arg 3=1 --output %t.i8-index.json
// RUN: FileCheck %s --check-prefix=I8-INDEX < %t.i8-index.json
// RUN: loom-dfg-sim %s --graph stream_static_signed_cast --arg 0=none --arg 1=254 --output %t.static-cast.json
// RUN: FileCheck %s --check-prefix=STATIC-CAST < %t.static-cast.json
// RUN: loom-dfg-sim %s --graph stream_i128_unsupported --arg 0=none --arg 1=0 --arg 2=1 --arg 3=1 --output %t.i128.json
// RUN: FileCheck %s --check-prefix=I128 < %t.i128.json
// RUN: loom-dfg-sim %s --graph stream_divide_by_zero --arg 0=none --arg 1=8 --arg 2=0 --arg 3=0 --output %t.divzero.json
// RUN: FileCheck %s --check-prefix=DIVZERO < %t.divzero.json
// RUN: loom-dfg-sim %s --graph carry_zero_trip_reentry --arg 0=none --arg 1=false --arg 1=false --arg 2=7 --arg 2=20 --output %t.carry-zero.json
// RUN: FileCheck %s --check-prefix=CARRY-ZERO < %t.carry-zero.json
// RUN: loom-dfg-sim %s --graph carry_finite_reentry --arg 0=none --arg 1=true --arg 1=false --arg 1=true --arg 1=false --arg 2=7 --arg 2=20 --arg 3=8 --arg 3=21 --output %t.carry-finite.json
// RUN: FileCheck %s --check-prefix=CARRY-FINITE < %t.carry-finite.json
// RUN: loom-dfg-sim %s --graph invariant_reentry --arg 0=none --arg 1=true --arg 1=false --arg 1=true --arg 1=false --arg 2=7 --arg 2=9 --output %t.invariant.json
// RUN: FileCheck %s --check-prefix=INVARIANT < %t.invariant.json
// RUN: loom-dfg-sim %s --graph gate_reentry_and_fanout --arg 0=none --arg 1=false --arg 1=true --arg 1=false --arg 1=true --arg 1=true --arg 1=false --arg 2=0 --arg 2=10 --arg 2=11 --arg 2=20 --arg 2=21 --arg 2=22 --output %t.gate.json
// RUN: FileCheck %s --check-prefix=GATE < %t.gate.json
// RUN: loom-dfg-sim %s --graph structured_gate_reentry --arg 0=none --output %t.structured-gate.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-GATE < %t.structured-gate.json
// RUN: loom-dfg-sim %s --graph structured_gate_reentry_no_phase --arg 0=none --output %t.structured-gate-no-phase.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-GATE-NO-PHASE < %t.structured-gate-no-phase.json
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph lowered_memory_exact_k --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --memref 4=1,2,3,99 --output %t.memory.json
// RUN: FileCheck %s --check-prefix=MEMORY < %t.memory.json

// Actor close/reset anchors below intentionally check token values and fire
// counts only. They do not define graph retirement for zero-output close
// transitions; that protocol remains pending.

// ZERO-DAG: "graph": "stream_zero_trip"
// ZERO-DAG: "dataflow.stream": 1
// ZERO-DAG: "i1:false"
// ZERO-NOT: "arith.addi"

// FINITE-DAG: "graph": "stream_finite"
// FINITE-DAG: "status": "pass"
// FINITE-DAG: "dynamic_work_items": 3
// FINITE-DAG: "dataflow.stream": 4
// FINITE-DAG: "i64:2"
// FINITE-DAG: "i1:false"
// FINITE-NOT: "i64:3"

// REPEATED-DAG: "graph": "stream_repeated"
// REPEATED-DAG: "status": "pass"
// REPEATED-DAG: "dynamic_work_items": 5
// REPEATED-DAG: "dataflow.stream": 7
// REPEATED-DAG: "i64:12"
// REPEATED-DAG: "i1:false"

// I8-CONT-DAG: "graph": "stream_i8_signed_cont"
// I8-CONT-DAG: "status": "pass"
// I8-CONT-DAG: "dynamic_work_items": 2
// I8-CONT-DAG: "dataflow.stream": 3
// I8-CONT-DAG: "dataflow.constant": 2
// I8-CONT-DAG: "arith.addi": 2
// I8-CONT-DAG: "i8:1"
// I8-CONT-DAG: "i1:false"

// I8-UNSIGNED-DAG: "graph": "stream_i8_unsigned_cont"
// I8-UNSIGNED-DAG: "status": "pass"
// I8-UNSIGNED-DAG: "dynamic_work_items": 1
// I8-UNSIGNED-DAG: "dataflow.stream": 2
// I8-UNSIGNED-DAG: "i8:255"
// I8-UNSIGNED-DAG: "i1:false"

// I8-INDEX-DAG: "graph": "stream_i8_signed_index_cast"
// I8-INDEX-DAG: "status": "pass"
// I8-INDEX-DAG: "dataflow.stream": 2
// I8-INDEX-DAG: "arith.index_cast": 1
// I8-INDEX-DAG: "index:-1"
// I8-INDEX-DAG: "i1:false"

// STATIC-CAST-DAG: "graph": "stream_static_signed_cast"
// STATIC-CAST-DAG: "status": "pass"
// STATIC-CAST-DAG: "dynamic_work_items": 2
// STATIC-CAST-DAG: "arith.index_cast": 2
// STATIC-CAST-DAG: "dataflow.stream": 3
// STATIC-CAST-DAG: "dataflow.constant": 2
// STATIC-CAST-DAG: "arith.addi": 2
// STATIC-CAST-DAG: "i64:0"
// STATIC-CAST-DAG: "i1:false"

// I128-DAG: "graph": "stream_i128_unsupported"
// I128-DAG: "status": "blocked"
// I128-DAG: "dataflow.stream integer bit width must be in [1, 64], got 128"

// DIVZERO-DAG: "graph": "stream_divide_by_zero"
// DIVZERO-DAG: "status": "blocked"
// DIVZERO-DAG: "arith.divsi divisor must be non-zero"

// CARRY-ZERO-DAG: "graph": "carry_zero_trip_reentry"
// CARRY-ZERO-DAG: "dataflow.carry": 4
// CARRY-ZERO-DAG: "dataflow.gate": 2
// CARRY-ZERO-DAG: "i32:20"

// CARRY-FINITE-DAG: "graph": "carry_finite_reentry"
// CARRY-FINITE-DAG: "dataflow.carry": 6
// CARRY-FINITE-DAG: "i32:21"

// INVARIANT-DAG: "graph": "invariant_reentry"
// INVARIANT-DAG: "dataflow.invariant": 6
// INVARIANT-DAG: "i32:9"

// GATE-DAG: "graph": "gate_reentry_and_fanout"
// GATE-DAG: "dataflow.gate": 6
// GATE-DAG: "dataflow.demux": 6

// STRUCTURED-GATE-DAG: "graph": "structured_gate_reentry"
// STRUCTURED-GATE-DAG: "status": "pass"
// STRUCTURED-GATE-DAG: "dataflow.gate": 6
// STRUCTURED-GATE-DAG: "arith.addi": 6
// STRUCTURED-GATE-DAG: "arith.select": 3
// STRUCTURED-GATE-DAG: "index:128"

// STRUCTURED-GATE-NO-PHASE-DAG: "graph": "structured_gate_reentry_no_phase"
// STRUCTURED-GATE-NO-PHASE-DAG: "status": "blocked"
// STRUCTURED-GATE-NO-PHASE-DAG: "structured scf.index_switch failed to execute arith.index_castui"

// MEMORY-DAG: "graph": "lowered_memory_exact_k"
// MEMORY-DAG: "dataflow.load": 3

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @stream_zero_trip(
      %ctrl: none, %init: i64, %limit: i64, %step: i64) -> (none, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i64
    %unused = arith.addi %iv, %iv : i64
    dataflow.graph.return %ctrl, %phase : none, i1
  }

  dataflow.graph.func private @stream_finite(
      %ctrl: none, %init: i64, %limit: i64, %step: i64)
      -> (none, i64, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i64
    dataflow.graph.return %ctrl, %iv, %phase : none, i64, i1
  }

  dataflow.graph.func private @stream_repeated(
      %ctrl: none, %init: i64, %limit: i64, %step: i64)
      -> (none, i64, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i64
    dataflow.graph.return %ctrl, %iv, %phase : none, i64, i1
  }

  dataflow.graph.func private @stream_i8_signed_cont(
      %ctrl: none, %init: i8, %limit: i8, %step: i8)
      -> (none, i8, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i8
    %one = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %value = arith.addi %iv, %one : i8
    dataflow.graph.return %ctrl, %value, %phase : none, i8, i1
  }

  dataflow.graph.func private @stream_i8_unsigned_cont(
      %ctrl: none, %init: i8, %limit: i8, %step: i8)
      -> (none, i8, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while ugt : i8
    dataflow.graph.return %ctrl, %iv, %phase : none, i8, i1
  }

  dataflow.graph.func private @stream_i8_signed_index_cast(
      %ctrl: none, %init: i8, %limit: i8, %step: i8)
      -> (none, index, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i8
    %index = arith.index_cast %iv : i8 to index
    dataflow.graph.return %ctrl, %index, %phase : none, index, i1
  }

  dataflow.graph.func private @stream_static_signed_cast(
      %ctrl: none, %byte: i8) -> (none, i64, i1) {
    %index = arith.index_cast %byte : i8 to index
    %init = arith.index_cast %index : index to i64
    %limit = arith.constant 0 : i64
    %step = arith.constant 1 : i64
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i64
    %one = dataflow.constant %ctrl {const_value = 1 : i64} : i64
    %value = arith.addi %iv, %one : i64
    dataflow.graph.return %ctrl, %value, %phase : none, i64, i1
  }

  dataflow.graph.func private @stream_i128_unsupported(
      %ctrl: none, %init: i128, %limit: i128, %step: i128) -> (none, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i128
    dataflow.graph.return %ctrl, %phase : none, i1
  }

  dataflow.graph.func private @stream_divide_by_zero(
      %ctrl: none, %init: i64, %limit: i64, %step: i64) -> (none, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step sdiv while sgt : i64
    dataflow.graph.return %ctrl, %phase : none, i1
  }

  dataflow.graph.func private @carry_zero_trip_reentry(
      %ctrl: none, %phase: i1, %init: i32) -> (none, i32) {
    %unused_phase, %no_next = dataflow.gate %phase, %init : i32
    %value = dataflow.carry %phase, %init, %no_next : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @carry_finite_reentry(
      %ctrl: none, %phase: i1, %init: i32, %next: i32) -> (none, i32) {
    %value = dataflow.carry %phase, %init, %next : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @invariant_reentry(
      %ctrl: none, %phase: i1, %init: i32) -> (none, i32) {
    %value = dataflow.invariant %phase, %init : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @gate_reentry_and_fanout(
      %ctrl: none, %phase: i1, %value: i32) -> none {
    %child_phase, %child_value = dataflow.gate %phase, %value : i32
    %left:2 = dataflow.demux %child_phase, %child_value
        : (i1, i32) -> (i32, i32)
    %right:2 = dataflow.demux %child_phase, %child_value
        : (i1, i32) -> (i32, i32)
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @structured_gate_reentry(%ctrl: none)
      -> (none, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %result = scf.for %i = %c0 to %c6 step %c1
        iter_args(%acc = %c0) -> (index) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %is1 = arith.cmpi eq, %i, %c1 : index
      %is3 = arith.cmpi eq, %i, %c3 : index
      %is4 = arith.cmpi eq, %i, %c4 : index
      %is1or3 = arith.ori %is1, %is3 : i1
      %phase = arith.ori %is1or3, %is4 : i1
      %after_cond, %after_value = dataflow.gate %phase, %i : index
      %next = scf.index_switch %i -> index
      case 0 {
        scf.yield %acc : index
      }
      case 1 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 2 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      case 3 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 4 {
        %with_value = arith.addi %acc, %after_value : index
        %bonus = arith.select %after_cond, %c100, %c0 : index
        %sum = arith.addi %with_value, %bonus : index
        scf.yield %sum : index
      }
      case 5 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      default {
        scf.yield %acc : index
      }
      scf.yield %next : index
    }
    dataflow.graph.return %ctrl, %result : none, index
  }

  dataflow.graph.func private @structured_gate_reentry_no_phase(%ctrl: none)
      -> (none, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %i = %c0 to %c3 step %c1
        iter_args(%acc = %c0) -> (index) {
      %phase = arith.cmpi ne, %i, %c1 : index
      %after_cond, %after_value = dataflow.gate %phase, %i : index
      %next = scf.index_switch %i -> index
      case 0 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 1 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      case 2 {
        %with_value = arith.addi %acc, %after_value : index
        %unexpected_phase = arith.index_castui %after_cond : i1 to index
        %sum = arith.addi %with_value, %unexpected_phase : index
        scf.yield %sum : index
      }
      default {
        scf.yield %acc : index
      }
      scf.yield %next : index
    }
    dataflow.graph.return %ctrl, %result : none, index
  }

  dataflow.graph.func private @lowered_memory_exact_k(
      %ctrl: none, %init: i32, %limit: i32, %step: i32, %ptr: !llvm.ptr)
      -> none {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %iv, %phase = dataflow.stream %c0, %limit, %c1
        step add while slt : i32
    %raw = dataflow.carry %phase, %ptr, %next : !llvm.ptr
    %body_phase, %body_ptr = dataflow.gate %phase, %raw : !llvm.ptr
    %exit:2 = dataflow.demux %phase, %raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %value = llvm.load %body_ptr : !llvm.ptr -> i32
    %next = llvm.getelementptr %body_ptr[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    dataflow.graph.return %ctrl : none
  }
}

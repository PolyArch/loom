// RUN: not loom-raise-opt --split-input-file --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// LLVM pointer capabilities cannot become dynamic dataflow carry state. The
// pass must reject each recurrence before partially rewriting its memory ops.

// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_i8_f32
// CHECK: dataflow.carry
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-NOT: dataflow.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @g_pointer_carry_i8_f32(
      %start: none, %lower: i32, %upper: i32, %step: i32, %bias: f32,
      %source: !llvm.ptr, %destination: !llvm.ptr)
      -> (none, !llvm.ptr, !llvm.ptr)
      attributes {input_segments = array<i32: 4, 0, 2>,
                  result_segments = array<i32: 0, 0, 2>} {
    %stream_init = arith.constant 0 : i32
    %stream_step = arith.constant 1 : i32
    %index, %phase = dataflow.stream %stream_init, %upper, %stream_step
        step add while slt : i32
    %source_raw = dataflow.carry %phase, %source, %source_next : !llvm.ptr
    %source_phase, %source_current = dataflow.gate %phase, %source_raw
        : !llvm.ptr
    %source_exit:2 = dataflow.demux %phase, %source_raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %destination_raw = dataflow.carry %phase, %destination, %destination_next
        : !llvm.ptr
    %destination_phase, %destination_current =
        dataflow.gate %phase, %destination_raw : !llvm.ptr
    %destination_exit:2 = dataflow.demux %phase, %destination_raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %source_next = llvm.getelementptr %source_current[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %source_current : !llvm.ptr -> f32
    %sum = arith.addf %data, %bias : f32
    %destination_next = llvm.getelementptr %destination_current[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %sum, %destination_current : f32, !llvm.ptr
    dataflow.graph.return values() streams()
        memories(%source_exit#0, %destination_exit#0 : !llvm.ptr, !llvm.ptr)
        complete(%start : none)
  }
}

// -----

// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32
// CHECK: dataflow.carry
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-NOT: dataflow.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32(
      %start: none, %lower: i32, %upper: i32, %step: i32, %bias: f32,
      %source: !llvm.ptr, %destination: !llvm.ptr)
      -> (none, !llvm.ptr, !llvm.ptr)
      attributes {input_segments = array<i32: 4, 0, 2>,
                  result_segments = array<i32: 0, 0, 2>} {
    %stream_init = arith.constant 0 : i32
    %stream_step = arith.constant 1 : i32
    %index, %phase = dataflow.stream %stream_init, %upper, %stream_step
        step add while slt : i32
    %source_raw = dataflow.carry %phase, %source, %source_next : !llvm.ptr
    %source_phase, %source_current = dataflow.gate %phase, %source_raw
        : !llvm.ptr
    %source_exit:2 = dataflow.demux %phase, %source_raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %destination_raw = dataflow.carry %phase, %destination, %destination_next
        : !llvm.ptr
    %destination_phase, %destination_current =
        dataflow.gate %phase, %destination_raw : !llvm.ptr
    %destination_exit:2 = dataflow.demux %phase, %destination_raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %source_next = llvm.getelementptr %source_current[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %source_next : !llvm.ptr -> f32
    %sum = arith.addf %data, %bias : f32
    %destination_next = llvm.getelementptr %destination_current[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %sum, %destination_next : f32, !llvm.ptr
    dataflow.graph.return values() streams()
        memories(%source_exit#0, %destination_exit#0 : !llvm.ptr, !llvm.ptr)
        complete(%start : none)
  }
}

// -----

// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_nonordinal_init
// CHECK: dataflow.carry
// CHECK: llvm.load
// CHECK-NOT: dataflow.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @g_pointer_carry_nonordinal_init(
      %start: none, %limit: i32, %step: i32, %bias: f32,
      %source: !llvm.ptr) -> (none, !llvm.ptr)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    %stream_init = arith.constant 4 : i32
    %stream_step = arith.constant 1 : i32
    %index, %phase = dataflow.stream %stream_init, %limit, %stream_step
        step add while slt : i32
    %raw = dataflow.carry %phase, %source, %next : !llvm.ptr
    %body_phase, %current = dataflow.gate %phase, %raw : !llvm.ptr
    %exit:2 = dataflow.demux %phase, %raw
        : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
    %data = llvm.load %current : !llvm.ptr -> f32
    %sum = arith.addf %data, %bias : f32
    %next = llvm.getelementptr %current[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    dataflow.graph.return values() streams() memories(%exit#0 : !llvm.ptr)
        complete(%start : none)
  }
}

// -----

// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_widening_i8
// CHECK: dataflow.carry
// CHECK: llvm.load
// CHECK-NOT: dataflow.load
dataflow.graph.func private @g_pointer_carry_widening_i8(
    %start: none, %limit: i8, %source: !llvm.ptr) -> (none, !llvm.ptr)
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %stream_init = arith.constant 0 : i8
  %stream_step = arith.constant 1 : i8
  %index, %phase = dataflow.stream %stream_init, %limit, %stream_step
      step add while ne : i8
  %raw = dataflow.carry %phase, %source, %next : !llvm.ptr
  %body_phase, %current = dataflow.gate %phase, %raw : !llvm.ptr
  %exit:2 = dataflow.demux %phase, %raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %value = llvm.load %current : !llvm.ptr -> i8
  %next = llvm.getelementptr %current[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
  dataflow.graph.return values() streams() memories(%exit#0 : !llvm.ptr)
      complete(%start : none)
}

// -----

// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_dynamic_offset_i8_f32
// CHECK: dataflow.carry
// CHECK: llvm.load
// CHECK-NOT: dataflow.load
dataflow.graph.func private @g_pointer_carry_dynamic_offset_i8_f32(
    %start: none, %lower: i32, %upper: i32, %step: i32,
    %bias: f32, %dynamic_offset: i32, %source: !llvm.ptr)
    -> (none, !llvm.ptr)
    attributes {input_segments = array<i32: 5, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %index, %phase = dataflow.stream %lower, %upper, %step
      step add while slt : i32
  %current = dataflow.carry %phase, %source, %next : !llvm.ptr
  %next = llvm.getelementptr %current[4]
      : (!llvm.ptr) -> !llvm.ptr, i8
  %offset = arith.addi %index, %dynamic_offset : i32
  %dynamic = llvm.getelementptr %current[%offset]
      : (!llvm.ptr, i32) -> !llvm.ptr, i8
  %data = llvm.load %dynamic : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  dataflow.graph.return values() streams() memories(%current : !llvm.ptr)
      complete(%start : none)
}

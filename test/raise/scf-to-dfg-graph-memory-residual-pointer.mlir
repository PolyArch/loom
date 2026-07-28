// RUN: not loom-raise-opt --split-input-file --loom-lower-graph-memory %s 2>&1 | FileCheck %s

// CHECK-COUNT-5: loom-lower-graph-memory: residual memory operation 'llvm.load' has no explicit completion event

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[8, 8, 8, 8]> : vector<4xi64>>
>} {
  dataflow.graph private @wrapped_pointer_index(
      %start: none, %lower: i8, %upper: i8, %step: i8, %base: !llvm.ptr)
      -> (i8)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i8
    %ptr = llvm.getelementptr %base[%index]
        : (!llvm.ptr, i8) -> !llvm.ptr, !llvm.array<256 x i8>
    %value = llvm.load %ptr : !llvm.ptr -> i8
    dataflow.graph.return %start, %value : none, i8
  }
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>,
  #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 64, 64]> : vector<4xi64>>
>} {
  dataflow.graph private @narrow_index(
      %start: none, %lower: i64, %upper: i64, %step: i64, %base: !llvm.ptr)
      -> (i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i64
    %ptr = llvm.getelementptr %base[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %ptr : !llvm.ptr -> i32
    dataflow.graph.return %start, %value : none, i32
  }
}

// -----

module attributes {
  llvm.data_layout = "e-p:64:64-p1:64:64-ni:1",
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr<1>, dense<[64, 64, 64, 64]> : vector<4xi64>>
  >
} {
  dataflow.graph private @non_integral_pointer(
      %start: none, %lower: i64, %upper: i64, %step: i64,
      %base: !llvm.ptr<1>) -> (i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i64
    %ptr = llvm.getelementptr %base[%index]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %value = llvm.load %ptr : !llvm.ptr<1> -> i32
    dataflow.graph.return %start, %value : none, i32
  }
}

// -----

module attributes {llvm.data_layout = "e-p:32:32"} {
  dataflow.graph private @llvm_layout_pointer_index(
      %start: none, %lower: i64, %upper: i64, %step: i64, %base: !llvm.ptr)
      -> (i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %phase = dataflow.stream %lower, %upper, %step
        step add while slt : i64
    %ptr = llvm.getelementptr %base[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %ptr : !llvm.ptr -> i32
    dataflow.graph.return %start, %value : none, i32
  }
}

// -----

// The recurrence itself is structured, but a two-byte update cannot address
// an f32 memory element without changing the source access semantics.
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @unaligned_byte_recurrence(
      %start: none, %limit: i64, %base: !llvm.ptr) -> f32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = arith.constant 0 : i64
    %step = arith.constant 2 : i64
    %initial = arith.constant 0.0 : f32
    %result:2 = scf.while (%sum = %initial, %offset = %zero)
        : (f32, i64) -> (f32, i64) {
      %ptr = llvm.getelementptr inbounds|nuw %base[%offset]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %value = llvm.load %ptr : !llvm.ptr -> f32
      %next_sum = arith.addf %sum, %value : f32
      %next_offset = arith.addi %offset, %step : i64
      %more = arith.cmpi ult, %next_offset, %limit : i64
      scf.condition(%more) %next_sum, %next_offset : f32, i64
    } do {
    ^bb0(%sum: f32, %offset: i64):
      scf.yield %sum, %offset : f32, i64
    }
    dataflow.graph.return values(%result#0 : f32) streams() memories()
        complete(%start : none)
  }
}

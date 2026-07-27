// RUN: env LOOM_INDEX_WIDTH=32 loom-raise-opt --loom-lower-graph-memory \
// RUN:   -split-input-file -verify-diagnostics %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// A configured width is validated exactly like a declared one, so an
// unrepresentable one is reported instead of building an invalid type.
// RUN: env LOOM_INDEX_WIDTH=33554432 not loom-raise-opt \
// RUN:   --loom-lower-graph-memory -split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONFIGURED-OVER-LIMIT
// An override wider than a host integer keeps its own value instead of
// narrowing into a legal one, whether it exceeds the unsigned range or the
// decimal parse itself.
// RUN: env LOOM_INDEX_WIDTH=4294967297 not loom-raise-opt \
// RUN:   --loom-lower-graph-memory -split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONFIGURED-OVER-UNSIGNED
// RUN: env LOOM_INDEX_WIDTH=18446744073709551617 not loom-raise-opt \
// RUN:   --loom-lower-graph-memory -split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONFIGURED-OVER-PARSE

// CONFIGURED-OVER-LIMIT: loom-lower-graph-memory: index bit width 33554432 has no fixed representation

// CONFIGURED-OVER-UNSIGNED: loom-lower-graph-memory: index bit width 4294967297 has no fixed representation

// CONFIGURED-OVER-PARSE: loom-lower-graph-memory: index bit width 18446744073709551617 has no fixed representation

// The configured width owns the canonical index of a program that declares
// none, so a 32-bit index cannot materialize this i64 ordinal and the access
// stays residual.
dataflow.graph private @configured_index_width(
    %ctrl: none, %init: i64, %limit: i64, %step: i64, %base: !llvm.ptr)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i64
  %addr = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // expected-error @+1 {{residual memory operation 'llvm.load' has no explicit completion event}}
  %value = llvm.load %addr : !llvm.ptr -> f32
  llvm.store %value, %addr : f32, !llvm.ptr
  dataflow.graph.return %ctrl : none
}

// -----

// An explicit declaration overrides the configured width, so the same ordinal
// is admitted and every access is normalized.

// CHECK-LABEL: dataflow.graph private @declared_index_width
// CHECK: %[[IDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK: dataflow.load %{{.*}}[%[[IDX]]]
// CHECK-NOT: llvm.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @declared_index_width(
      %ctrl: none, %init: i64, %limit: i64, %step: i64, %base: !llvm.ptr)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i64
    %addr = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %addr : !llvm.ptr -> f32
    llvm.store %value, %addr : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}

// -----

// The Structured candidate owns its canonical address-index width. LLVM's
// pointer index size still determines source GEP semantics, but it cannot
// override an already materialized, narrower GEP operand and fixed index
// declaration during mechanical graph-memory lowering.

// CHECK-LABEL: dataflow.graph private @canonical_i32_on_pointer64
// CHECK: %[[I32:.*]] = arith.extsi %{{.*}} : i16 to i32
// CHECK: %[[ADDR:.*]] = arith.index_cast %[[I32]] : i32 to index
// CHECK: %[[DATA:.*]], %[[DONE:.*]] = dataflow.load %{{.*}}[%[[ADDR]]]
// CHECK: dataflow.store %{{.*}}[%[[ADDR]]] %[[DATA]] %[[DONE]]
// CHECK-NOT: llvm.getelementptr
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @canonical_i32_on_pointer64(
      %ctrl: none, %address: i16, %base: !llvm.ptr)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%address]
        : (!llvm.ptr, i16) -> !llvm.ptr, !llvm.array<4 x i8>
    %value = llvm.load %ptr : !llvm.ptr -> f32
    llvm.store %value, %ptr : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}

// -----

// Region lowering consumes the width the pass boundary resolved, so an
// index-typed loop becomes a 64-bit ordinal stream even though the configured
// width is 32.

// CHECK-LABEL: dataflow.graph private @declared_index_loop
// CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step add while slt : i64
// CHECK-NOT: scf.for
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @declared_index_loop(
      %ctrl: none, %lb: index, %ub: index, %step: index, %mem: memref<?xi32>)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.for %i = %lb to %ub step %step : index {
      %value = memref.load %mem[%i] : memref<?xi32>
      memref.store %value, %mem[%i] : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }
}

// -----

// Index-domain rewriting of a memory address consumes the same resolved width.
// A zero extension is redundant only when its source already spans the
// canonical index, so under a declared 64-bit index this 32-bit source keeps
// its extension instead of becoming a signed cast of the narrow value.

// CHECK-LABEL: dataflow.graph private @declared_index_zext_address
// CHECK-NOT: arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[WIDE:.*]] = arith.extui %{{.*}} : i32 to i64
// CHECK: %[[ZIDX:.*]] = arith.index_cast %[[WIDE]] : i64 to index
// CHECK: dataflow.load %{{.*}}[%[[ZIDX]]]
// CHECK-NOT: llvm.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @declared_index_zext_address(
      %ctrl: none, %raw: i32, %base: !llvm.ptr) -> i8
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide = arith.extui %raw : i32 to i64
    %ptr = llvm.getelementptr %base[%wide] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %value = llvm.load %ptr : !llvm.ptr -> i8
    dataflow.graph.return %ctrl, %value : none, i8
  }
}

// -----

// The width belongs to the scope that declares it, so a graph under a nested
// shared layout owner uses that owner rather than the outermost one.

// CHECK-LABEL: dataflow.graph private @nested_declared_index
// CHECK: %[[NIDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK: dataflow.load %{{.*}}[%[[NIDX]]]
// CHECK-NOT: llvm.load
module {
  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
  } {
    dataflow.graph private @nested_declared_index(
        %ctrl: none, %init: i64, %limit: i64, %step: i64, %base: !llvm.ptr)
        attributes {input_segments = array<i32: 3, 0, 1>,
                    result_segments = array<i32: 0, 0, 0>} {
      %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i64
      %addr = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %value = llvm.load %addr : !llvm.ptr -> f32
      llvm.store %value, %addr : f32, !llvm.ptr
      dataflow.graph.return %ctrl : none
    }
  }
}

// -----

// An index-domain constant keeps its exact value, so a canonical index wider
// than a host integer neither narrows nor aborts. This one exceeds the index
// attribute's own storage, so the extended constant and its cast remain.

// CHECK-LABEL: dataflow.graph private @wide_index_constant_address
// CHECK: %[[WIDEC:.*]] = dataflow.constant %{{.*}} {const_value = 18446744073709551616 : i128} : i128
// CHECK: %[[WIDEIDX:.*]] = arith.index_cast %[[WIDEC]] : i128 to index
// CHECK: dataflow.load %{{.*}}[%[[WIDEIDX]]]
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 128>>
} {
  dataflow.graph private @wide_index_constant_address(
      %ctrl: none, %mem: memref<?xi8>) -> i8
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %c = dataflow.constant %ctrl
        {const_value = 18446744073709551616 : i128} : i128
    %addr = arith.index_cast %c : i128 to index
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xi8>
    dataflow.graph.return %done, %data : none, i8
  }
}

// -----

// A declaration the canonical index width cannot resolve is reported at the
// pass boundary, with its own reason, before anything is rewritten.
// expected-error @+1 {{loom-lower-graph-memory: index bit width must be nonzero}}
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 0>>
} {
  dataflow.graph private @zero_index_width(
      %ctrl: none, %init: i64, %limit: i64, %step: i64, %base: !llvm.ptr)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i64
    %addr = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %addr : !llvm.ptr -> f32
    llvm.store %value, %addr : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}

// -----

// A width above the fixed integer representation is rejected exactly, instead
// of silently becoming its low bits.
// expected-error @+1 {{loom-lower-graph-memory: index bit width 4294967328 has no fixed representation}}
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 4294967328>>
} {
  dataflow.graph private @unrepresentable_index_width(
      %ctrl: none, %init: i64, %limit: i64, %step: i64, %base: !llvm.ptr)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i64
    %addr = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %addr : !llvm.ptr -> f32
    llvm.store %value, %addr : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}

// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/declared.mlir \
// RUN:   | FileCheck %s --check-prefix=DECLARED
// RUN: env LOOM_INDEX_WIDTH=33554432 not loom-raise-opt \
// RUN:   --loom-lower-graph-memory %t.dir/configured-invalid.mlir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONFIGURED-INVALID
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/zero.mlir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZERO

// The Structured candidate's declared index width remains independent of LLVM
// pointer arithmetic. The exact LLVM DataLayout owns the retained GEP.

// DECLARED-LABEL: dataflow.graph private @declared_index_graph(
// DECLARED-SAME: %[[BASE:[^, )]+]]: !llvm.ptr
// DECLARED-SAME: [[MEM:%[^, )]+]]: memref<?xf32>)
// DECLARED: %[[ADDRESS:.*]] = llvm.getelementptr inbounds %[[BASE]]
// DECLARED: %[[DATA:.*]], %[[READ:.*]] = dataflow.load [[MEM]][%[[ADDRESS]]]
// DECLARED: dataflow.store [[MEM]][%[[ADDRESS]]] %[[DATA]] %[[READ]]
// DECLARED-NOT: builtin.unrealized_conversion_cast

//--- declared.mlir
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.thread private @declared_index
      domain(#dataflow.thread_domain<dense>)(%base: !llvm.ptr, %address: i16)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%address, %base)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%offset: i16, %memory: !llvm.ptr):
        %ptr = llvm.getelementptr inbounds %memory[%offset]
            : (!llvm.ptr, i16) -> !llvm.ptr, !llvm.array<4 x i8>
        %value = llvm.load %ptr : !llvm.ptr -> f32
        llvm.store %value, %ptr : f32, !llvm.ptr
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "declared_index_graph", source_maps = []} :
        (i16, !llvm.ptr) -> ()
    dataflow.thread.yield
  }
}

// Process configuration is validated without narrowing into a host integer.

// CONFIGURED-INVALID: loom-lower-graph-memory: index bit width 33554432 has no fixed representation

//--- configured-invalid.mlir
module {
  dataflow.graph private @configured_invalid(
      %start: none, %memory: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
}

// An owner declaration must provide a positive fixed index width.

// ZERO: loom-lower-graph-memory: index bit width must be nonzero

//--- zero.mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 0>>
} {
  dataflow.graph private @zero_index(
      %start: none, %memory: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
}

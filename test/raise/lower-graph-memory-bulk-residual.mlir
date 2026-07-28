// RUN: env LOOM_INDEX_WIDTH=64 loom-raise-opt --loom-lower-graph-memory \
// RUN:   -split-input-file -verify-diagnostics %s -o /dev/null

// Bulk-memory semantics must be expanded before Spatial ownership selection.
// The graph-memory pass has no competing late lowering path.
dataflow.graph private @residual_memcpy(
    %start: none, %len: i64, %src: !llvm.ptr, %dst: !llvm.ptr) -> ()
    attributes {input_segments = array<i32: 1, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{residual memory operation 'llvm.intr.memcpy' has no explicit completion event}}
  "llvm.intr.memcpy"(%dst, %src, %len)
      <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  dataflow.graph.return %start : none
}

// -----

dataflow.graph private @residual_memmove(
    %start: none, %len: i64, %src: !llvm.ptr, %dst: !llvm.ptr) -> ()
    attributes {input_segments = array<i32: 1, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{residual memory operation 'llvm.intr.memmove' has no explicit completion event}}
  "llvm.intr.memmove"(%dst, %src, %len)
      <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  dataflow.graph.return %start : none
}

// -----

dataflow.graph private @residual_memset(
    %start: none, %len: i64, %fill: i8, %dst: !llvm.ptr) -> ()
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{residual memory operation 'llvm.intr.memset' has no explicit completion event}}
  "llvm.intr.memset"(%dst, %fill, %len)
      <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  dataflow.graph.return %start : none
}

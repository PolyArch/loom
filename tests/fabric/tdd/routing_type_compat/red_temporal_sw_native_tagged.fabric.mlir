// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must be !dataflow.tagged

// Temporal switch requires all ports to be tagged; i32 is not tagged.
fabric.module @test_tsw_native_rejected(
  %a: i32, %b: i32
) -> (i32) {
  %o0, %o1 = fabric.temporal_sw [num_route_table = 4]
    %a, %b : i32 -> i32, i32
  fabric.yield %o0 : i32
}

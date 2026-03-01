// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_SW_PORT_LIMIT

// 33 inputs exceed the 32-port limit.
fabric.module @test(
    %i0: !dataflow.tagged<!dataflow.bits<32>, i4>, %i1: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i2: !dataflow.tagged<!dataflow.bits<32>, i4>, %i3: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i4: !dataflow.tagged<!dataflow.bits<32>, i4>, %i5: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i6: !dataflow.tagged<!dataflow.bits<32>, i4>, %i7: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i8: !dataflow.tagged<!dataflow.bits<32>, i4>, %i9: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i10: !dataflow.tagged<!dataflow.bits<32>, i4>, %i11: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i12: !dataflow.tagged<!dataflow.bits<32>, i4>, %i13: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i14: !dataflow.tagged<!dataflow.bits<32>, i4>, %i15: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i16: !dataflow.tagged<!dataflow.bits<32>, i4>, %i17: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i18: !dataflow.tagged<!dataflow.bits<32>, i4>, %i19: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i20: !dataflow.tagged<!dataflow.bits<32>, i4>, %i21: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i22: !dataflow.tagged<!dataflow.bits<32>, i4>, %i23: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i24: !dataflow.tagged<!dataflow.bits<32>, i4>, %i25: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i26: !dataflow.tagged<!dataflow.bits<32>, i4>, %i27: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i28: !dataflow.tagged<!dataflow.bits<32>, i4>, %i29: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i30: !dataflow.tagged<!dataflow.bits<32>, i4>, %i31: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %i32_: !dataflow.tagged<!dataflow.bits<32>, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %o = fabric.temporal_sw [num_route_table = 1] %i0, %i1, %i2, %i3, %i4, %i5, %i6, %i7, %i8, %i9, %i10, %i11, %i12, %i13, %i14, %i15, %i16, %i17, %i18, %i19, %i20, %i21, %i22, %i23, %i24, %i25, %i26, %i27, %i28, %i29, %i30, %i31, %i32_ : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %o : !dataflow.tagged<!dataflow.bits<32>, i4>
}

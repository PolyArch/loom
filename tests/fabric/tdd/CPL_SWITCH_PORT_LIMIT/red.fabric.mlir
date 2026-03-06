// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_SWITCH_PORT_LIMIT

// 33 inputs exceeds the 32-port limit.
fabric.module @test(
  %i0: !dataflow.bits<32>, %i1: !dataflow.bits<32>, %i2: !dataflow.bits<32>, %i3: !dataflow.bits<32>,
  %i4: !dataflow.bits<32>, %i5: !dataflow.bits<32>, %i6: !dataflow.bits<32>, %i7: !dataflow.bits<32>,
  %i8: !dataflow.bits<32>, %i9: !dataflow.bits<32>, %i10: !dataflow.bits<32>, %i11: !dataflow.bits<32>,
  %i12: !dataflow.bits<32>, %i13: !dataflow.bits<32>, %i14: !dataflow.bits<32>, %i15: !dataflow.bits<32>,
  %i16: !dataflow.bits<32>, %i17: !dataflow.bits<32>, %i18: !dataflow.bits<32>, %i19: !dataflow.bits<32>,
  %i20: !dataflow.bits<32>, %i21: !dataflow.bits<32>, %i22: !dataflow.bits<32>, %i23: !dataflow.bits<32>,
  %i24: !dataflow.bits<32>, %i25: !dataflow.bits<32>, %i26: !dataflow.bits<32>, %i27: !dataflow.bits<32>,
  %i28: !dataflow.bits<32>, %i29: !dataflow.bits<32>, %i30: !dataflow.bits<32>, %i31: !dataflow.bits<32>,
  %i32: !dataflow.bits<32>
) -> (!dataflow.bits<32>) {
  %o = fabric.switch %i0, %i1, %i2, %i3, %i4, %i5, %i6, %i7, %i8, %i9, %i10, %i11, %i12, %i13, %i14, %i15, %i16, %i17, %i18, %i19, %i20, %i21, %i22, %i23, %i24, %i25, %i26, %i27, %i28, %i29, %i30, %i31, %i32 : !dataflow.bits<32> -> !dataflow.bits<32>
  fabric.yield %o : !dataflow.bits<32>
}

// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_SWITCH_PORT_LIMIT

// 33 inputs exceeds the 32-port limit.
fabric.module @test(
  %i0: i32, %i1: i32, %i2: i32, %i3: i32,
  %i4: i32, %i5: i32, %i6: i32, %i7: i32,
  %i8: i32, %i9: i32, %i10: i32, %i11: i32,
  %i12: i32, %i13: i32, %i14: i32, %i15: i32,
  %i16: i32, %i17: i32, %i18: i32, %i19: i32,
  %i20: i32, %i21: i32, %i22: i32, %i23: i32,
  %i24: i32, %i25: i32, %i26: i32, %i27: i32,
  %i28: i32, %i29: i32, %i30: i32, %i31: i32,
  %i32: i32
) -> (i32) {
  %o = fabric.switch %i0, %i1, %i2, %i3, %i4, %i5, %i6, %i7, %i8, %i9, %i10, %i11, %i12, %i13, %i14, %i15, %i16, %i17, %i18, %i19, %i20, %i21, %i22, %i23, %i24, %i25, %i26, %i27, %i28, %i29, %i30, %i31, %i32 : i32 -> i32
  fabric.yield %o : i32
}
